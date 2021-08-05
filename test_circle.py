import pydiffvg
import torch
import skimage
import numpy as np
import matplotlib.pyplot as plt

# Use GPU if available
pydiffvg.set_use_gpu(torch.cuda.is_available())
print("\n\n\n\nCUDA: ", torch.cuda.is_available())

canvas_width = 256
canvas_height = 256
circle = pydiffvg.Circle(radius = torch.tensor(40.0),
                         center = torch.tensor([128.0, 128.0]),
                         stroke_width= torch.tensor(2))
shapes = [circle]
circle_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([0]),
                                   fill_color = torch.tensor([0, 0, 0, 0.0]),
                                   stroke_color=torch.tensor([0.6, 0.3, 0.3, 1.0]),
                                   )
shape_groups = [circle_group]
scene_args = pydiffvg.RenderFunction.serialize_scene(canvas_width, canvas_height, shapes, shape_groups)

render = pydiffvg.RenderFunction.apply
img = render(256, # width
             256, # height
             2,   # num_samples_x
             2,   # num_samples_y
             0,   # seed
             None,
             *scene_args)
# The output image is in linear RGB space. Do Gamma correction before saving the image.
# pydiffvg.imwrite(img.cpu(), 'logs/single_circle/target.png', gamma=2.2)
target = img.clone()
plt.imshow(target.detach().cpu().numpy())
plt.show()