import torch
import pydiffvg
import matplotlib.pyplot as plt
import numpy as np

from diff_rendering import scene_to_svg

from vector_gan import AwesomeBezierGAN, BezierSNGAN

# model = AwesomeBezierGAN(img_size=64, n_circles=2).cuda()
model = BezierSNGAN(img_size=64).cuda()
checkpoint = torch.load("logs/GOOD_04.24.02.01_n60/checkpoints/last_step.pth")
model.generator.load_state_dict(checkpoint['generator_state_dict'])
model.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])

print(model.latent_dim)
z1 = torch.rand((1,model.latent_dim)).cuda()
z2 = torch.rand((1,model.latent_dim)).cuda()

plt.figure(figsize=(15, 3))
i=1
for alpha in np.linspace(0,1,5):
    print(alpha)
    z = (1-alpha)*z1 + alpha*z2
    with torch.no_grad():
        images, scenes = model.generator.forward_return_scene(z)

    plt.subplot(1,5,i)
    i+=1
    plt.imshow(images[0][0].detach().cpu(), cmap='gray_r')
    plt.axis('off')
    scene_to_svg(scenes[0], f'result_{alpha}.svg')

plt.show()


print(scenes)

# pydiffvg.save_svg("result.svg", canvas_size, canvas_size, shapes, shape_groups)
# scene_to_svg(scenes[0], 'result.svg')