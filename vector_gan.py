import torch
import torch.nn as nn

from simple_gan import SimpleGAN, Generator, Discriminator
from diff_rendering import bezier_render

import matplotlib.pyplot as plt


class VectorGeneratorBezier(Generator):
    """
    Generator with diff rasterizer inside.
    """
    def __init__(self, num_segments=2, n_strokes=20, latent_dim=100, img_size=32):
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
        self.width_limits = (0.5, 3.5)

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


class SimpleGANBezier(SimpleGAN):
    def __init__(self, latent_dim=100, img_size=28):
        super(SimpleGANBezier, self).__init__()
        self.generator = VectorGeneratorBezier(img_size=img_size, latent_dim=latent_dim)


def test():
    gen = VectorGeneratorBezier(img_size=28).to('cuda')
    img = gen.generate_batch(batch_size=2)
    print(img.shape)
    plt.imshow(img[0][0].detach().cpu().numpy(), cmap='gray_r')
    plt.axis('off')
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    test()