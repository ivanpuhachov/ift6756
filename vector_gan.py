import torch
import torch.nn as nn

from simple_gan import SimpleGAN, Generator, Discriminator
from diff_rendering import bezier_render

import matplotlib.pyplot as plt


class VectorGeneratorBezier(Generator):
    """
    Generator with diff rasterizer inside.
    """
    def __init__(self, num_segments=2, n_strokes=16, latent_dim=100, img_size=32):
        """
        Init generator with diff rasterizer inside. Objects: Bezier curves

        :param num_segments: number of segments in Bezier curve
        :param n_strokes: number of curves
        :param latent_dim: latent dimension of input noise
        :param img_size: size of generated image
        """
        super(VectorGeneratorBezier, self).__init__(latent_dim=latent_dim, img_size=img_size)
        self.num_segments = num_segments
        self.n_strokes = n_strokes
        self.width_limits = (0.5, 4.5)

        # each bezier curves takes 3 x segments points + init point (each point = pair of coords)
        self.linear_points = nn.Sequential(
            nn.Linear(self.flat_out,
                      out_features=self.n_strokes*(self.num_segments*2*3 + 2)),
            nn.Tanh()
        )

        # each curve has width
        self.linear_widths = nn.Sequential(
            nn.Linear(self.flat_out,
                      self.n_strokes),
            nn.Sigmoid()
        )

        # each curve has alpha
        self.linear_alphas = nn.Sequential(
            nn.Linear(self.flat_out,
                      self.n_strokes),
            nn.Sigmoid()
        )

        # TODO: colors!

    def forward_return_scene(self, z):
        batch_size = z.shape[0]

        x = self.flat_layers(z)

        points = self.linear_points(x).view(batch_size, self.n_strokes, self.num_segments*3+1, 2)
        widths = self.linear_widths(x) * (self.width_limits[1] - self.width_limits[0]) + self.width_limits[0]
        alphas = self.linear_alphas(x)

        images, scenes = bezier_render(
            all_points=points,
            all_widths=widths,
            all_alphas=alphas,
            colors=None,
            canvas_size=self.img_size)

        return images, scenes

    def forward(self, z):
        images, scenes = self.forward_return_scene(z)
        return images


class ConvDiscriminator(Discriminator):
    def __init__(self, img_size=64):
        super(ConvDiscriminator, self).__init__(img_size=img_size//4)
        self.img_size = img_size
        self.hidden_convs = 10
        self.conv_layers = torch.nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.hidden_convs, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.hidden_convs),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=self.hidden_convs, out_channels=self.hidden_convs, kernel_size=2, stride=2),
            nn.LeakyReLU(0.1),

            nn.Conv2d(in_channels=self.hidden_convs, out_channels=self.hidden_convs, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.hidden_convs),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=self.hidden_convs, out_channels=self.hidden_convs, kernel_size=2, stride=2),
            nn.LeakyReLU(0.1),

            nn.Conv2d(in_channels=self.hidden_convs, out_channels=self.hidden_convs, kernel_size=3, padding=1,
                      bias=False),
            nn.BatchNorm2d(self.hidden_convs),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=self.hidden_convs, out_channels=1, kernel_size=1)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.conv_layers(x).view(batch_size, -1)
        y = self.flat_layers(x)
        return y


class ConvSNDiscriminator(Discriminator):
    def __init__(self, img_size=64):
        super(ConvSNDiscriminator, self).__init__(img_size=img_size//16)
        sn = nn.utils.spectral_norm
        width = img_size
        self.conv_layers = torch.nn.Sequential(
            nn.Conv2d(1, width, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(width, 2 * width, 4, padding=1, stride=2),
            nn.LeakyReLU(0.2, inplace=True),

            sn(nn.Conv2d(2 * width, 2 * width, 3, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            sn(nn.Conv2d(2 * width, 4 * width, 4, padding=1, stride=2)),
            nn.LeakyReLU(0.2, inplace=True),

            sn(nn.Conv2d(4 * width, 4 * width, 3, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            sn(nn.Conv2d(4 * width, width * 4, 4, padding=1, stride=2)),
            nn.LeakyReLU(0.2, inplace=True),

            sn(nn.Conv2d(4 * width, 4 * width, 3, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            sn(nn.Conv2d(4 * width, width * 4, 4, padding=1, stride=2)),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=4*width, out_channels=1, kernel_size=1)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.conv_layers(x).view(batch_size, -1)
        y = self.flat_layers(x)
        return y


class SimpleGANBezier(SimpleGAN):
    def __init__(self, latent_dim=100, img_size=28):
        super(SimpleGANBezier, self).__init__(latent_dim=latent_dim, img_size=img_size)
        self.generator = VectorGeneratorBezier(img_size=img_size, latent_dim=latent_dim)


class BezierGAN(SimpleGAN):
    def __init__(self, latent_dim=100, img_size=28):
        super(BezierGAN, self).__init__(latent_dim=latent_dim, img_size=img_size)
        self.generator = VectorGeneratorBezier(img_size=img_size, latent_dim=latent_dim)
        self.discriminator = ConvDiscriminator(img_size=img_size)

class BezierSNGAN(SimpleGAN):
    def __init__(self, latent_dim=100, img_size=28):
        super(BezierSNGAN, self).__init__(latent_dim=latent_dim, img_size=img_size)
        self.generator = VectorGeneratorBezier(img_size=img_size, latent_dim=latent_dim)
        self.discriminator = ConvSNDiscriminator(img_size=img_size)


def test():
    gen = VectorGeneratorBezier(img_size=96).to('cuda')
    disc = ConvSNDiscriminator(img_size=96).to('cuda')
    img = gen.generate_batch(batch_size=2)
    print(img.shape)
    plt.imshow(img[0][0].detach().cpu().numpy(), cmap='gray_r')
    plt.axis('off')
    plt.colorbar()
    plt.show()
    y = disc(img)
    print(y)


if __name__ == "__main__":
    test()