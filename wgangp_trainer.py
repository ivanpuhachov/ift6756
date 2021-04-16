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


class Trainer:
    def __init__(self,
                 model: SimpleGAN, dataloader,
                 lr=0.00005, clip_value=0.01, discriminator_steps=5,
                 files_to_backup=[]):
        self.model = model
        self.dataloader = dataloader
        self.lr = lr
        self.clip_value = clip_value
        self.batch_size = self.dataloader.batch_size
        self.discriminator_steps = discriminator_steps
        self.opt_generator = torch.optim.RMSprop(self.model.generator.parameters(), lr=self.lr)
        self.opt_discriminator = torch.optim.RMSprop(self.model.discriminator.parameters(), lr=self.lr)

        self.files_to_backup = ['wgangp_trainer.py']
        self.files_to_backup.extend(files_to_backup)

        self.log_dir = self.create_logs_folder()
        print(f"log dir: {self.log_dir}")
        self.tb_writer = SummaryWriter(self.log_dir+"tensorboard/")
        print(f"TensorBoard: tensorboard --logdir={self.log_dir+'tensorboard/'}")
        self.plot_dir = self.log_dir + "plots/"

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

    def step_discriminator(self, data):
        batch_size = data.shape[0]
        fake_data = self.model.generator.generate_batch(batch_size=batch_size).detach()
        decision_true = self.model.discriminator(data)
        decision_fake = self.model.discriminator(fake_data)
        loss = -torch.mean(decision_true) + torch.mean(decision_fake)
        self.opt_discriminator.zero_grad()
        loss.backward()
        self.opt_discriminator.step()
        return loss.item()

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
        for epoch in range(n_epochs):
            total_discriminator_loss = 0
            total_generator_loss = 0

            for i, data in tqdm(enumerate(self.dataloader), total=len(self.dataloader)):

                total_discriminator_loss += self.step_discriminator(data=data[0].cuda())

                self.clip_discriminator()

                if i % self.discriminator_steps == 0:
                    total_generator_loss += self.step_generator()

            total_generator_loss /= len(self.dataloader)
            total_discriminator_loss /= len(self.dataloader)
            self.report_loss(g_loss=total_generator_loss, d_loss=total_discriminator_loss, epoch=epoch)
            self.report_generation(epoch=epoch)


def test():
    model = SimpleGAN().to('cuda')
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "../../data/mnist",
            train=True,
            download=True,
            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]),
        ),
        batch_size=64,
        shuffle=True,
    )

    tr = Trainer(model, dataloader)
    tr.train(2)


if __name__=="__main__":
    test()
