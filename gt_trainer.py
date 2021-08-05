import torch

from wgangp_trainer import Trainer
from simple_gan import SimpleGAN
from vector_gan import SimpleGANBezier
from inception_score import PretrainedInception, frechet_inception_distance

class GTTrainer(Trainer):
    """
    Copying everything from the official repo. This was done to validate the
    """
    def __init__(self,
                 model: SimpleGAN, dataloader,
                 lr=0.0001, clip_value=0.01, gp_weight=10.0,
                 discriminator_steps=4,
                 files_to_backup=list(),
                 save_freq=10,
                 log_dir=None):
        # TODO: Learning rate decay
        super(GTTrainer, self).__init__(model, dataloader, lr, clip_value, gp_weight, discriminator_steps, files_to_backup, save_freq, log_dir)

    def gradient_penalty(self, data, fake_data):
        bs = data.size(0)
        epsilon = torch.rand(bs, 1, 1, 1).cuda()
        epsilon = epsilon.expand_as(data)

        interpolation = epsilon * data.data + (1 - epsilon) * fake_data.data
        interpolation = torch.autograd.Variable(interpolation, requires_grad=True)

        interpolation_logits = self.model.discriminator(interpolation)
        grad_outputs = torch.ones(interpolation_logits.size()).cuda()

        gradients = torch.autograd.grad(outputs=interpolation_logits,
                                     inputs=interpolation,
                                     grad_outputs=grad_outputs,
                                     create_graph=True, retain_graph=True)[0]

        gradients = gradients.view(bs, -1)
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        return self.gp_weight * ((gradients_norm - 0) ** 2).mean()

    def step_discriminator(self, data, iteration):
        """Try to classify fake as 0 and real as 1."""
        batch_size = data.shape[0]
        self.opt_discriminator.zero_grad()

        # no backprop to gen
        fake_gen = self.model.generator.generate_batch(batch_size=batch_size).detach()
        eps = 0.1
        noise_fake = eps * torch.randn_like(fake_gen, requires_grad=False)
        noise_real = eps * torch.randn_like(data, requires_grad=False)
        fake = torch.clamp(fake_gen + noise_fake, min=0, max=1)
        real = torch.clamp(data + noise_real, min=0, max=1)

        fake_pred = self.model.discriminator(fake)
        real_pred = self.model.discriminator(real)

        gradient_penalty = self.gradient_penalty(data, fake_gen)
        loss_true = real_pred.mean()
        loss_fake = fake_pred.mean()
        loss_d = loss_fake - loss_true + gradient_penalty
        gradient_penalty = gradient_penalty.item()

        loss_d.backward()
        #TODO: add this to the main model
        torch.nn.utils.clip_grad_norm_(
            self.model.discriminator.parameters(), self.clip_gradient)

        self.opt_discriminator.step()
        self.report_running_discriminator(real=real,
                                          losses=(loss_true.item(), loss_fake.item(), gradient_penalty),
                                          iteration=iteration)

        return loss_d.item(), gradient_penalty

    def step_generator(self, iteration):
        """Try to classify fake as 1."""
        fake_data = self.model.generator.generate_batch(batch_size=self.batch_size)
        self.opt_generator.zero_grad()

        eps = 0.1
        noise_fake = eps * torch.randn_like(fake_data, requires_grad=False)
        fake = torch.clamp(fake_data+noise_fake, min=0, max=1)
        fake_pred = self.model.discriminator(fake)

        loss_g = -fake_pred.mean()

        loss_g.backward()

        # clip gradients
        nrm = torch.nn.utils.clip_grad_norm_(
            self.model.generator.parameters(), self.clip_gradient)
        # if nrm > self.gradient_clip:
        #     print("Clipped generator gradient (%.5f) to %.2f",
        #               nrm, self.gradient_clip)

        self.opt_generator.step()
        self.report_running_generator(fake=fake.detach(), loss=loss_g.item(), iteration=iteration)
        return loss_g.item()

    def training_routine(self, iteration, data):
        gen_loss, disc_loss = 0, 0
        disc_loss, gp = self.step_discriminator(data, iteration)
        if iteration % self.discriminator_steps == 0:
            gen_loss = self.step_generator(iteration)
        # if iteration % self.discriminator_steps != 0:  # Discriminator update
        #
        #     disc_loss, gp = self.step_discriminator(data, iteration)
        #
        # else:  # Generator update
        #     gen_loss = self.step_generator(iteration)

        return gen_loss, disc_loss
