import pydiffvg
import matplotlib.pyplot as plt

canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene("data/lion.svg")
scene_args = pydiffvg.RenderFunction.serialize_scene(canvas_width, canvas_height, shapes, shape_groups)
render = pydiffvg.RenderFunction.apply
img = render(canvas_width,  # width
             canvas_height,  # height
             2,  # num_samples_x
             2,  # num_samples_y
             0,  # seed
             None,  # bg
             *scene_args)

plt.imshow(img.detach().cpu())
plt.show()