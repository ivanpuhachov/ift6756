import torch
import torch.nn as nn


class Generator(nn.Module):
    """
    Template class for generator. Here - feed forward flat perceptron.
    """
    def __init__(self, latent_dim=100, img_size=28):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.img_size = img_size

        self.flat_layers = nn.Sequential(
            nn.Linear(self.latent_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, 1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.1),
            nn.Linear(1024, self.img_size**2),
        )

    def forward(self, z):
        x = torch.tanh(self.flat_layers(z))
        x = x.view(x.shape[0], 1, self.img_size, self.img_size)
        return x

    def generate_batch(self, batch_size):
        z = torch.rand(size=(batch_size, self.latent_dim)).cuda()
        return self.forward(z)

    def forward_return_scene(self, z):
        """
        This function generates raster images AND scenes from latent vector

        :param z: (batch_size, latent_dim) - latent vector
        :return:  images, scenes
        """
        raise NotImplementedError

    def generate_return_scene(self, batch_size):
        """
        This function generates batch of raster images AND scenes

        :param batch_size:
        :return: images, scenes
        """
        z = torch.rand(size=(batch_size, self.latent_dim)).cuda()
        return self.forward_return_scene(z)


class Discriminator(nn.Module):
    """
    Tenplate class for discriminator. Here - feed forward flat perceptron.
    """
    def __init__(self, img_size=28):
        super(Discriminator, self).__init__()
        self.img_size = img_size
        self.model = nn.Sequential(
            nn.Linear(self.img_size**2, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)
        y = self.model(x_flat)
        return y


class SimpleGAN(nn.Module):
    """
    This is a simple GAN base class. Generator - feed-forward flat perceptron. Discriminator - feed forward flat perceptron
    """
    def __init__(self, latent_dim=100, img_size=28):
        """
        SimpleGAN initializer

        :param latent_dim: (int) latent dimension of generator
        :param img_size: (int) image size of generated
        """
        super(SimpleGAN, self).__init__()
        self.img_size = img_size
        self.latent_dim = latent_dim
        self.discriminator = Discriminator(img_size=img_size)
        self.generator = Generator(img_size=img_size, latent_dim=latent_dim)


def test():
    model = SimpleGAN().cuda()
    x = model.generator.generate_batch(batch_size=5)
    print("Generated: ", x.shape)
    y = model.discriminator.forward(x)
    print("Discriminator: ", y.shape)


if __name__ == "__main__":
    test()
