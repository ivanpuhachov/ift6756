# Taken from https://github.com/BachiLi/diffvg/blob/master/apps/generative_models/rendering.py

import torch
import random
import pydiffvg


def render(canvas_width, canvas_height, shapes, shape_groups, samples=2,
           seed=None):
    # pydiffvg.set_use_gpu(torch.cuda.is_available())
    if seed is None:
        seed = random.randint(0, 1000000)
    _render = pydiffvg.RenderFunction.apply
    scene_args = pydiffvg.RenderFunction.serialize_scene(
        canvas_width, canvas_height, shapes, shape_groups)
    img = _render(canvas_width, canvas_height, samples, samples,
                  seed,   # seed
                  None,  # background image
                  *scene_args)
    return img


def bezier_render(all_points, all_widths, all_alphas, force_cpu=False,
                  canvas_size=32, colors=None):
    dev = all_points.device
    if force_cpu:
        all_points = all_points.to("cpu")
        all_widths = all_widths.to("cpu")
        all_alphas = all_alphas.to("cpu")

        if colors is not None:
            colors = colors.to("cpu")

    all_points = 0.5*(all_points + 1.0) * canvas_size

    eps = 1e-4
    all_points = all_points + eps*torch.randn_like(all_points)

    bs, num_strokes, num_pts, _ = all_points.shape
    num_segments = (num_pts - 1) // 3
    n_out = 3 if colors is not None else 1
    output = torch.zeros(bs, n_out, canvas_size, canvas_size,
                      device=all_points.device)

    scenes = []
    for k in range(bs):
        shapes = []
        shape_groups = []
        for p in range(num_strokes):
            points = all_points[k, p].contiguous().cuda()
            # bezier
            num_ctrl_pts = torch.zeros(num_segments, dtype=torch.int32) + 2
            width = all_widths[k, p].cuda()
            alpha = all_alphas[k, p].cuda()
            if colors is not None:
                color = colors[k, p]
            else:
                color = torch.ones(3, device=alpha.device)

            color = torch.cat([color, alpha.view(1,)])

            path = pydiffvg.Path(
                num_control_points=num_ctrl_pts, points=points,
                stroke_width=width, is_closed=False)
            shapes.append(path)
            path_group = pydiffvg.ShapeGroup(
                shape_ids=torch.tensor([len(shapes) - 1]),
                fill_color=None,
                stroke_color=color)
            shape_groups.append(path_group)

        # Rasterize
        scenes.append((canvas_size, canvas_size, shapes, shape_groups))
        raster = render(canvas_size, canvas_size, shapes, shape_groups,
                        samples=2)
        raster = raster.permute(2, 0, 1).view(4, canvas_size, canvas_size)

        alpha = raster[3:4]
        if colors is not None:  # color output
            image = raster[:3]
            alpha = alpha.repeat(3, 1, 1)
        else:
            image = raster[:1]

        # alpha compositing
        image = image*alpha
        output[k] = image

    output = output.to(dev)

    return output, scenes


def my_render(curve_points, curve_widths, curve_alphas,
              circle_centers, circle_radiuses, circle_widths, circle_alphas,
              canvas_size=32, colors=None):
    dev = curve_points.device

    curve_points = 0.5*(curve_points + 1.0) * canvas_size
    circle_centers = 0.5*(circle_centers + 1.0) * canvas_size
    circle_radiuses = circle_radiuses * canvas_size / 2

    eps = 1e-4
    curve_points = curve_points + eps*torch.randn_like(curve_points)

    bs, num_strokes, num_pts, _ = curve_points.shape
    num_segments = (num_pts - 1) // 3
    num_circles = circle_centers.shape[1]
    n_out = 3 if colors is not None else 1
    output = torch.zeros(bs, n_out, canvas_size, canvas_size,
                      device=curve_points.device)

    scenes = []
    for k in range(bs):
        shapes = []
        shape_groups = []
        for p in range(num_strokes):
            points = curve_points[k, p].contiguous().cuda()
            # bezier
            num_ctrl_pts = torch.zeros(num_segments, dtype=torch.int32) + 2
            width = curve_widths[k, p].cuda()
            alpha = curve_alphas[k, p].cuda()
            if colors is not None:
                color = colors[k, p]
            else:
                color = torch.ones(3, device=alpha.device)

            color = torch.cat([color, alpha.view(1,)])

            path = pydiffvg.Path(
                num_control_points=num_ctrl_pts, points=points,
                stroke_width=width, is_closed=False)
            shapes.append(path)
            path_group = pydiffvg.ShapeGroup(
                shape_ids=torch.tensor([len(shapes) - 1]),
                fill_color=None,
                stroke_color=color)
            shape_groups.append(path_group)

        for c in range(num_circles):
            center = circle_centers[k, c]
            radius = circle_radiuses[k, c]
            width = circle_widths[k, c]
            alpha = circle_alphas[k, c]
            circle = pydiffvg.Circle(radius=radius,
                                     center=center,
                                     stroke_width=width)
            shapes.append(circle)
            if colors is not None:
                color = colors[k, p]
            else:
                color = torch.ones(3, device=alpha.device)
            color = torch.cat([color, alpha.view(1,)])

            circle_group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([len(shapes) - 1]),
                                               fill_color=torch.tensor([0, 0, 0, 0.0]),
                                               stroke_color=color,
                                               )

            shape_groups.append(circle_group)

        # Rasterize
        scenes.append((canvas_size, canvas_size, shapes, shape_groups))
        raster = render(canvas_size, canvas_size, shapes, shape_groups,
                        samples=2)
        raster = raster.permute(2, 0, 1).view(4, canvas_size, canvas_size)

        alpha = raster[3:4]
        if colors is not None:  # color output
            image = raster[:3]
            alpha = alpha.repeat(3, 1, 1)
        else:
            image = raster[:1]

        # alpha compositing
        image = image*alpha
        output[k] = image

    output = output.to(dev)

    return output, scenes


def scene_to_svg(scene, path):
    canvas_width, canvas_height, shapes, shape_groups = scene
    pydiffvg.save_svg(path,
                      canvas_width, canvas_height, shapes, shape_groups)