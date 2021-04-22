import torch

from torch.utils.data import DataLoader, TensorDataset
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
from inception_score import PretrainedInception, frechet_inception_distance


class Trainer:
    def __init__(self,
                 model: SimpleGAN, dataloader,
                 lr=0.00005, clip_value=0.01, gp_weight=10.0,
                 discriminator_steps=5,
                 files_to_backup=list(),
                 save_freq=10,
                 log_dir=None):
        self.model = model.cuda()
        self.dataloader = dataloader
        self.lr = lr
        self.clip_value = clip_value
        self.gp_weight = gp_weight
        self.batch_size = self.dataloader.batch_size
        self.discriminator_steps = discriminator_steps
        self.opt_generator = torch.optim.RMSprop(self.model.generator.parameters(), lr=self.lr)
        self.opt_discriminator = torch.optim.RMSprop(self.model.discriminator.parameters(), lr=self.lr)

        self.files_to_backup = ['wgangp_trainer.py', 'diff_rendering.py', 'simple_gan.py', 'vector_gan.py', 'dataset.py']
        self.files_to_backup.extend(files_to_backup)

        self.save_freq = save_freq

        if log_dir is None:
            self.log_dir = "logs/run" + datetime.now().strftime("%d%H%M") + "/"
        else:
            self.log_dir = log_dir
        self.create_logs_folder()
        print(f"log dir: {self.log_dir}")
        self.tb_writer = SummaryWriter(self.log_dir+"tensorboard/")
        print(f"TensorBoard: tensorboard --logdir={self.log_dir+'tensorboard/'}")
        self.plot_dir = self.log_dir + "plots/"
        self.checkpoint_dir = self.log_dir + "checkpoints/"

        self.backup_files()

        self.pretrainedInception = PretrainedInception()
        self.eval_inception_n_samples = 1000

    def create_logs_folder(self):
        if os.path.exists(self.log_dir):
            pass
            # for f in os.listdir(self.log_dir):
            #     os.remove(os.path.join(self.log_dir, f))
        else:
            os.mkdir(self.log_dir)
        os.mkdir(self.log_dir + "tensorboard/")
        os.mkdir(self.log_dir + "plots/")
        os.mkdir(self.log_dir + "checkpoints/")
        return self.log_dir

    def backup_files(self):
        for f in self.files_to_backup:
            copyfile(f, self.log_dir + f)

    def report_loss(self, g_loss, d_loss, epoch):
        self.tb_writer.add_scalar(tag="loss/gen", scalar_value=g_loss, global_step=epoch)
        self.tb_writer.add_scalar(tag="loss/disc", scalar_value=d_loss, global_step=epoch)

    def report_incpetion_scores(self, inc_score, fid, epoch):
        self.tb_writer.add_scalar(tag="inception/score", scalar_value=inc_score, global_step=epoch)
        self.tb_writer.add_scalar(tag="inception/fid", scalar_value=fid, global_step=epoch)

    def report_generation(self, epoch, save=True):
        generated = self.model.generator.generate_batch(batch_size=24).detach().cpu()
        image = make_grid(generated, nrow=4)
        self.tb_writer.add_image('training/fake', image, global_step=epoch)
        if save:
            plt.imshow(image.permute(1,2,0).numpy(), cmap='Greys')
            plt.title(f"epoch {epoch}")
            plt.axis("off")
            plt.savefig(self.plot_dir+f"{epoch}.png", bbox_inches='tight')
            plt.close()

    def report_real_images(self, epoch, save=True):
        real = next(iter(self.dataloader))[0][:24].cpu()
        image = make_grid(real, nrow=4)
        self.tb_writer.add_image('training/real', image, global_step=epoch)
        if save:
            plt.imshow(image.permute(1,2,0).numpy(), cmap='Greys')
            plt.title(f"epoch {epoch}")
            plt.axis("off")
            plt.savefig(self.plot_dir+f"real_{epoch}.png", bbox_inches='tight')
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

    def gradient_penalty(self, data, fake_data):
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
        loss = loss_true + loss_fake
        self.opt_discriminator.zero_grad()
        loss.backward()
        self.opt_discriminator.step()
        return loss.item(), loss_true.item(), loss_fake.item()

    def step_generator(self):
        fake_data = self.model.generator.generate_batch(batch_size=self.batch_size)
        loss = -torch.mean(self.model.discriminator(fake_data))
        self.opt_generator.zero_grad()
        loss.backward()
        self.opt_generator.step()
        return loss.item()

    def step_generator_GD(self):
        """
        Does a LOGAN-GP latent optimization (gradient descent) step of generator.
        Idea from LOGAN: Latent Optimisation for Generative Adversarial Networks https://arxiv.org/abs/1912.00953
        See formula 4 and algorithm 1

        :return: (float) generator loss
        """
        alpha = 0.9
        z_init = self.model.generator.generate_latent(batch_size=self.batch_size)
        z_init = torch.autograd.Variable(z_init, requires_grad=True)
        fake_init = self.model.generator.forward(z=z_init)
        f_init = self.model.discriminator(fake_init)
        grad_outputs = torch.ones_like(f_init, requires_grad=False)
        dfdz = torch.autograd.grad(
            outputs=f_init,
            inputs=z_init,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True
        )
        z_new = z_init + alpha * dfdz[0]
        z_new = torch.clamp(z_new, min=-1.0, max=1.0).detach()

        fake_data = self.model.generator.forward(z=z_new)
        loss = -torch.mean(self.model.discriminator(fake_data))
        self.opt_generator.zero_grad()
        loss.backward()
        self.opt_generator.step()
        return loss.item()

    def step_generator_NGD(self):
        """
        Does a LOGAN-GP latent optimization (Natural Gradient Descent) step of generator.
        Idea from LOGAN: Latent Optimisation for Generative Adversarial Networks https://arxiv.org/abs/1912.00953
        See formula 16 and algorithm 1

        :return: (float) generator loss
        """
        alpha = 0.9
        beta = 0.1
        z_init = self.model.generator.generate_latent(batch_size=self.batch_size)
        z_init = torch.autograd.Variable(z_init, requires_grad=True)
        fake_init = self.model.generator.forward(z=z_init)
        f_init = self.model.discriminator(fake_init)
        grad_outputs = torch.ones_like(f_init, requires_grad=False)
        dfdz = torch.autograd.grad(
            outputs=f_init,
            inputs=z_init,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True
        )
        g = dfdz[0]
        delta_z = alpha * g / (beta + torch.norm(g))
        z_new = torch.clamp(z_init + delta_z, min=-1.0, max=1.0).detach()

        fake_data = self.model.generator.forward(z=z_new)
        loss = -torch.mean(self.model.discriminator(fake_data))
        self.opt_generator.zero_grad()
        loss.backward()
        self.opt_generator.step()
        return loss.item()

    def clip_discriminator(self):
        # WGAN does weight clipping to enforce a Lipschitz constraint
        for param in self.model.discriminator.parameters():
            param.data.clamp_(-self.clip_value, self.clip_value)

    def training_routine(self, iteration, data):
        raise NotImplementedError

    def train(self, n_epochs):
        self.save_checkpoint(name="init.pth")
        for i_epoch in range(n_epochs):
            print(f"Epoch: {i_epoch}")
            self.model.train()
            total_discriminator_loss = 0
            total_generator_loss = 0

            for i, data in tqdm(enumerate(self.dataloader), total=len(self.dataloader)):
                gen_loss, disc_loss = self.training_routine(iteration=i, data=data[0].float().cuda())
                total_generator_loss += gen_loss
                total_discriminator_loss += disc_loss

            self.model.eval()
            total_generator_loss /= len(self.dataloader)
            total_discriminator_loss /= len(self.dataloader)
            self.report_loss(g_loss=total_generator_loss, d_loss=total_discriminator_loss, epoch=i_epoch)
            self.report_generation(epoch=i_epoch)
            self.report_real_images(epoch=i_epoch)

            self.save_checkpoint(name="last_step.pth")

            if i_epoch%self.save_freq == self.save_freq-1:
                self.save_checkpoint(name=f"checkpoint_{i_epoch}.pth")

            inception_score = self.eval_inception_score()
            fid = self.eval_frechet_inception_ditance()
            self.report_incpetion_scores(inc_score=inception_score, fid=fid, epoch=i_epoch)
        self.tb_writer.close()

    def eval_inception_score(self):
        print("Calculating inception score")
        with torch.no_grad():
            generated_batch = self.model.generator.generate_batch(batch_size=self.eval_inception_n_samples).\
                                  repeat(1,3,1,1).cpu().numpy() * 2 - 1
            score = self.pretrainedInception.inception_score(generated_batch)
            print(f"Inception score: {score}")
            return score[0]

    def eval_frechet_inception_ditance(self):
        print("Calculating FID")
        n_iters = self.eval_inception_n_samples // self.batch_size
        real_data = list()
        fake_data = list()
        for i, data in enumerate(self.dataloader):
            real_data.append(data[0].float())
            with torch.no_grad():
                fake_data.append(self.model.generator.generate_batch(batch_size=self.batch_size))
            if i==n_iters:
                break
        real_data = torch.cat(real_data, dim=0).repeat(1,3,1,1) * 2 - 1
        fake_data = torch.cat(fake_data, dim=0).repeat(1,3,1,1) * 2 - 1
        print("Computing Frechet stats")
        mu_real, sigma_real = self.pretrainedInception.compute_frechet_stats(real_data, batch_size=self.batch_size)
        mu_fake, sigma_fake = self.pretrainedInception.compute_frechet_stats(fake_data, batch_size=self.batch_size)
        print("Computing distance")
        fid = frechet_inception_distance(mu_real, sigma_real, mu_fake, sigma_fake)
        print("FID: ", fid)
        return fid


class Vanilla_Trainer(Trainer):
    def __init__(self,
                 model: SimpleGAN, dataloader,
                 lr=0.00005, clip_value=0.01, gp_weight=10.0,
                 discriminator_steps=5,
                 files_to_backup=list(),
                 save_freq=10,
                 log_dir=None):
        super(Vanilla_Trainer, self).__init__(model, dataloader, lr, clip_value, gp_weight, discriminator_steps, files_to_backup, save_freq, log_dir)

    def training_routine(self, iteration, data):
        gen_loss, disc_loss = 0, 0
        losses_discriminator = self.step_discriminator(data=data)
        disc_loss += losses_discriminator[0]

        if iteration % self.discriminator_steps == 0:
            gen_loss += self.step_generator()
        return gen_loss, disc_loss

class WGAN_Trainer(Trainer):
    def __init__(self,
                 model: SimpleGAN, dataloader,
                 lr=0.00005, clip_value=0.01, gp_weight=10.0,
                 discriminator_steps=5,
                 files_to_backup=list(),
                 save_freq=10,
                 log_dir=None):
        super(WGAN_Trainer, self).__init__(model, dataloader, lr, clip_value, gp_weight, discriminator_steps, files_to_backup, save_freq, log_dir)

    def training_routine(self, iteration, data):
        gen_loss, disc_loss = 0, 0
        losses_discriminator = self.step_discriminator(data=data)
        disc_loss += losses_discriminator[0]

        self.clip_discriminator()

        if iteration % self.discriminator_steps == 0:
            gen_loss += self.step_generator()
        return gen_loss, disc_loss


class WGANGP_Trainer(WGAN_Trainer):
    def __init__(self,
                 model: SimpleGAN, dataloader,
                 lr=0.00005, clip_value=0.01, gp_weight=10.0,
                 discriminator_steps=5,
                 files_to_backup=list(),
                 save_freq=10,
                 log_dir=None):
        super(WGANGP_Trainer, self).__init__(model, dataloader, lr, clip_value, gp_weight, discriminator_steps,
                                          files_to_backup, save_freq, log_dir)

    def step_discriminator(self, data):
        batch_size = data.shape[0]
        fake_data = self.model.generator.generate_batch(batch_size=batch_size).detach()
        decision_true = self.model.discriminator(data)
        decision_fake = self.model.discriminator(fake_data)
        loss_true = -torch.mean(decision_true)
        loss_fake = torch.mean(decision_fake)
        loss_gp = self.gradient_penalty(data, fake_data)
        loss = loss_true + loss_fake + self.gp_weight * loss_gp
        self.opt_discriminator.zero_grad()
        loss.backward()
        self.opt_discriminator.step()
        return loss.item(), loss_true.item(), loss_fake.item(), loss_gp.item()


class LOGAN_GD_Trainer(Trainer):
    def __init__(self,
                 model: SimpleGAN, dataloader,
                 lr=0.00005, clip_value=0.01, gp_weight=10.0,
                 discriminator_steps=5,
                 files_to_backup=list(),
                 save_freq=10,
                 log_dir=None):
        super(LOGAN_GD_Trainer, self).__init__(model, dataloader, lr, clip_value, gp_weight, discriminator_steps, files_to_backup, save_freq, log_dir)

    def training_routine(self, iteration, data):
        gen_loss, disc_loss = 0, 0
        losses_discriminator = self.step_discriminator(data=data)
        disc_loss += losses_discriminator[0]

        self.clip_discriminator()

        if iteration % self.discriminator_steps == 0:
            gen_loss += self.step_generator_GD()
        return gen_loss, disc_loss


class LOGAN_NGD_Trainer(Trainer):
    def __init__(self,
                 model: SimpleGAN, dataloader,
                 lr=0.00005, clip_value=0.01, gp_weight=10.0,
                 discriminator_steps=5,
                 files_to_backup=list(),
                 save_freq=10,
                 log_dir=None):
        super(LOGAN_NGD_Trainer, self).__init__(model, dataloader, lr, clip_value, gp_weight, discriminator_steps, files_to_backup, save_freq, log_dir)

    def training_routine(self, iteration, data):
        gen_loss, disc_loss = 0, 0
        losses_discriminator = self.step_discriminator(data=data)
        disc_loss += losses_discriminator[0]

        self.clip_discriminator()

        if iteration % self.discriminator_steps == 0:
            gen_loss += self.step_generator_NGD()
        return gen_loss, disc_loss


def test(n_epochs=20):
    model = SimpleGAN().to('cuda')
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "data/mnist",
            train=True,
            download=True,
            transform=transforms.Compose([transforms.ToTensor()]),
        ),
        batch_size=64,
        shuffle=True,
    )

    # tr = Vanilla_Trainer(model, dataloader)
    # tr = WGAN_Trainer(model, dataloader)
    # tr = WGANGP_Trainer(model, dataloader)
    # tr = LOGAN_GD_Trainer(model, dataloader)
    tr = LOGAN_NGD_Trainer(model, dataloader)
    tr.train(n_epochs)


def test_vector(n_epochs):
    model = SimpleGANBezier(img_size=28).to('cuda')
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "data/mnist",
            train=True,
            download=True,
            transform=transforms.Compose([transforms.ToTensor()]),
        ),
        batch_size=64,
        shuffle=True,
    )

    tr = Trainer(model, dataloader, files_to_backup=['simple_gan.py'])
    tr.train(n_epochs)  # 10 minutes per epoch!


if __name__ == "__main__":
    test(10)
    # test_vector()
