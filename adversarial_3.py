import pydiffvg
import argparse
import ttools.modules
import torch
import skimage.io
import torchvision.models as models
import requests

gamma = 1.0


def get_class(pred_vector):
    label_idx = torch.max(pred_vector.data, 1)[1][0].item()
    labels_link = "https://savan77.github.io/blog/labels.json"
    labels_json = requests.get(labels_link).json()
    labels = {int(idx): label for idx, label in labels_json.items()}
    x_pred = labels[label_idx]
    print(f"Label: {label_idx} - {x_pred}")


def main(args):
    inceptionv3 = models.inception_v3(pretrained=True, transform_input=False).cuda()
    inceptionv3.eval()
    perception_loss = ttools.modules.LPIPS().to(pydiffvg.get_device())

    canvas_width, canvas_height, shapes, shape_groups = \
        pydiffvg.svg_to_scene(args.svg)
    scene_args = pydiffvg.RenderFunction.serialize_scene( \
        canvas_width, canvas_height, shapes, shape_groups)

    render = pydiffvg.RenderFunction.apply
    img = render(canvas_width,  # width
                 canvas_height,  # height
                 2,  # num_samples_x
                 2,  # num_samples_y
                 0,  # seed
                 None,  # bg
                 *scene_args)
    # The output image is in linear RGB space. Do Gamma correction before saving the image.
    pydiffvg.imwrite(img.cpu(), 'logs/refine_svg/init.png', gamma=gamma)
    pydiffvg.imwrite(img.cpu(), 'logs/refine_svg/init_.png')

    points_vars = []
    for path in shapes:
        path.points.requires_grad = True
        points_vars.append(path.points)
    # color_vars = {}
    # for group in shape_groups:
    #     group.fill_color.requires_grad = True
    #     color_vars[group.fill_color.data_ptr()] = group.fill_color
    # color_vars = list(color_vars.values())

    # Optimize
    points_optim = torch.optim.Adam(points_vars, lr=1.0)
    # color_optim = torch.optim.Adam(color_vars, lr=0.01)

    # Adam iterations.
    for t in range(args.num_iter):
        print('iteration:', t)
        points_optim.zero_grad()
        # color_optim.zero_grad()
        # Forward pass: render the image.
        scene_args = pydiffvg.RenderFunction.serialize_scene( \
            canvas_width, canvas_height, shapes, shape_groups)
        img = render(canvas_width,  # width
                     canvas_height,  # height
                     2,  # num_samples_x
                     2,  # num_samples_y
                     0,  # seed
                     None,  # bg
                     *scene_args)
        # Compose img with white background
        img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3,
                                                          device=pydiffvg.get_device()) * (1 - img[:, :, 3:4])
        # Save the intermediate render.
        pydiffvg.imwrite(img.cpu(), 'logs/refine_svg/iter_{}.png'.format(t), gamma=gamma)
        img = img[:, :, :3]
        # Convert img from HWC to NCHW
        img = img.unsqueeze(0)
        img = img.permute(0, 3, 1, 2)  # NHWC -> NCHW
        output = inceptionv3.forward(img.cuda())
        get_class(output)

        target = torch.autograd.Variable(torch.LongTensor([291]), requires_grad=False).cuda()
        loss = torch.nn.CrossEntropyLoss()(output, target)
        print('render loss:', loss.item())

        # Backpropagate the gradients.
        loss.backward()

        # Take a gradient descent step.
        points_optim.step()
        # color_optim.step()
        # for group in shape_groups:
        #     group.fill_color.data.clamp_(0.0, 1.0)

        if t % 10 == 0 or t == args.num_iter - 1:
            pydiffvg.save_svg('logs/refine_svg/iter_{}.svg'.format(t),
                              canvas_width, canvas_height, shapes, shape_groups)

    # Render the final result.
    scene_args = pydiffvg.RenderFunction.serialize_scene( \
        canvas_width, canvas_height, shapes, shape_groups)
    img = render(canvas_width,  # width
                 canvas_height,  # height
                 2,  # num_samples_x
                 2,  # num_samples_y
                 0,  # seed
                 None,  # bg
                 *scene_args)
    # Save the intermediate render.
    pydiffvg.imwrite(img.cpu(), 'logs/refine_svg/final.png'.format(t), gamma=gamma)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--svg", help="source SVG path", default='data/lion.svg')
    parser.add_argument("--target", help="target image path")
    parser.add_argument("--use_lpips_loss", dest='use_lpips_loss', action='store_true')
    parser.add_argument("--num_iter", type=int, default=5)
    args = parser.parse_args()
    main(args)
