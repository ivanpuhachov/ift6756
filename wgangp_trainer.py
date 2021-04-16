import torch

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.utils import make_grid

import os
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
from shutil import copyfile

from simple_gan import SimpleGAN
from vector_gan import SimpleGANBezier


class Trainer:
    def __init__(self,
                 model: SimpleGAN, dataloader,
                 lr=0.00005, clip_value=0.01, gp_weight=10.0,
                 discriminator_steps=5,
                 files_to_backup=list(),
                 save_freq=100):
        self.model = model
        self.dataloader = dataloader
        self.lr = lr
        self.clip_value = clip_value
        self.gp_weight = gp_weight
        self.batch_size = self.dataloader.batch_size
        self.discriminator_steps = discriminator_steps
        self.opt_generator = torch.optim.RMSprop(self.model.generator.parameters(), lr=self.lr)
        self.opt_discriminator = torch.optim.RMSprop(self.model.discriminator.parameters(), lr=self.lr)

        self.files_to_backup = ['wgangp_trainer.py']
        self.files_to_backup.extend(files_to_backup)

        self.save_freq = save_freq

        self.log_dir = self.create_logs_folder()
        print(f"log dir: {self.log_dir}")
        self.tb_writer = SummaryWriter(self.log_dir+"tensorboard/")
        print(f"TensorBoard: tensorboard --logdir={self.log_dir+'tensorboard/'}")
        self.plot_dir = self.log_dir + "plots/"
        self.checkpoint_dir = self.log_dir + "checkpoints/"

        self.backup_files()

    def create_logs_folder(self):
        folder_log = "logs/run" + datetime.now().strftime("%d%H%M") + "/"
        if os.path.exists(folder_log):
            for f in os.listdir(folder_log):
                os.remove(os.path.join(folder_log, f))
        else:
            os.mkdir(folder_log)
        os.mkdir(folder_log + "tensorboard/")
        os.mkdir(folder_log + "plots/")
        os.mkdir(folder_log + "checkpoints/")
        return folder_log

    def backup_files(self):
        for f in self.files_to_backup:
            copyfile(f, self.log_dir + f)

    def report_loss(self, g_loss, d_loss, epoch):
        self.tb_writer.add_scalar(tag="gen loss", scalar_value=g_loss, global_step=epoch)
        self.tb_writer.add_scalar(tag="disc loss", scalar_value=d_loss, global_step=epoch)

    def report_generation(self, epoch, save=True):
        generated = self.model.generator.generate_batch(batch_size=24).detach().cpu()
        image = make_grid(generated, nrow=4)
        self.tb_writer.add_image('generation', image, global_step=epoch)
        if save:
            plt.imshow(image.permute(1,2,0).numpy(), cmap='Greys')
            plt.title(f"epoch {epoch}")
            plt.axis("off")
            plt.savefig(self.plot_dir+f"{epoch}.png", bbox_inches='tight')
            plt.close()

    def save_checkpoint(self, name):
        path = self.checkpoint_dir + name
        print(f"Saving model to {path}")
        torch.save({
            'generator_state_dict': self.model.generator.state_dict(),
            'discriminator_state_dict': self.model.discriminator.state_dict(),
            'generator_opt': self.opt_generator.state_dict(),
            'discriminator_opt': self.opt_discriminator.state_dict(),
        }, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.model.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.model.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.opt_generator.load_state_dict(checkpoint['generator_opt'])
        self.opt_discriminator.load_state_dict(checkpoint['discriminator_opt'])
        print(f"Model loaded from {path}")

    def gradien_penalty(self, data, fake_data):
        batch_size = data.shape[0]
        eps = torch.rand(size=(batch_size, 1, 1, 1)).cuda()
        data_interpolated = eps * data + (1-eps)*fake_data
        interpolated = torch.autograd.Variable(data_interpolated, requires_grad=True)
        decision_interpolated = self.model.discriminator(interpolated)
        grad_outputs = torch.ones_like(decision_interpolated, requires_grad=False)
        gradients = torch.autograd.grad(
            outputs=decision_interpolated,
            inputs=interpolated,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
        )
        gradients = gradients[0].view(batch_size, -1)
        gradient_norms = torch.sqrt(torch.sum(gradients**2, dim=1) + 1e-7)
        # Original WGAN-GP penalty https://arxiv.org/abs/1704.00028
        # penalty = torch.mean((gradient_norms - 1)**2)
        # Zero-centered WGAN-GP penalty from https://arxiv.org/abs/1902.03984
        penalty = torch.mean((gradient_norms - 0)**2)
        return penalty

    def step_discriminator(self, data):
        batch_size = data.shape[0]
        fake_data = self.model.generator.generate_batch(batch_size=batch_size).detach()
        decision_true = self.model.discriminator(data)
        decision_fake = self.model.discriminator(fake_data)
        loss_true = -torch.mean(decision_true)
        loss_fake = torch.mean(decision_fake)
        loss_gp = self.gradien_penalty(data, fake_data)
        loss = loss_true + loss_fake + self.gp_weight*loss_gp
        self.opt_discriminator.zero_grad()
        loss.backward()
        self.opt_discriminator.step()
        return loss.item(), loss_true.item(), loss_fake.item(), loss_gp.item()

    def step_generator(self):
        fake_data = self.model.generator.generate_batch(batch_size=self.batch_size)
        loss = -torch.mean(self.model.discriminator(fake_data))
        self.opt_generator.zero_grad()
        loss.backward()
        self.opt_generator.step()
        return loss.item()

    def clip_discriminator(self):
        # WGAN does weight clipping to enforce a Lipschitz constraint
        for param in self.model.discriminator.parameters():
            param.data.clamp_(-self.clip_value, self.clip_value)

    def train(self, n_epochs):
        self.save_checkpoint(name="init.pth")
        for i_epoch in range(n_epochs):
            print(f"Epoch: {i_epoch}")
            total_discriminator_loss = 0
            total_generator_loss = 0

            for i, data in tqdm(enumerate(self.dataloader), total=len(self.dataloader)):

                losses_discriminator = self.step_discriminator(data=data[0].cuda())
                total_discriminator_loss += losses_discriminator[0]

                self.clip_discriminator()

                if i % self.discriminator_steps == 0:
                    total_generator_loss += self.step_generator()

            total_generator_loss /= len(self.dataloader)
            total_discriminator_loss /= len(self.dataloader)
            self.report_loss(g_loss=total_generator_loss, d_loss=total_discriminator_loss, epoch=i_epoch)
            self.report_generation(epoch=i_epoch)

            self.save_checkpoint(name="last_step.pth")

            if i_epoch%self.save_freq == self.save_freq-1:
                self.save_checkpoint(name=f"checkpoint_{i_epoch}.pth")


def test():
    model = SimpleGAN().to('cuda')
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "data/mnist",
            train=True,
            download=True,
            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]),
        ),
        batch_size=64,
        shuffle=True,
    )

    tr = Trainer(model, dataloader, files_to_backup=['simple_gan.py'])
    tr.train(20)


def test_vector():
    model = SimpleGANBezier(img_size=28).to('cuda')
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "data/mnist",
            train=True,
            download=True,
            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]),
        ),
        batch_size=64,
        shuffle=True,
    )

    tr = Trainer(model, dataloader, files_to_backup=['simple_gan.py'])
    tr.train(5)  # 10 minutes per epoch!


if __name__ == "__main__":
    # test()
    test_vector()
