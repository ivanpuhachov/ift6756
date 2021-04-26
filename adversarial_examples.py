import pydiffvg
import torch
import skimage
import skimage.io
import skimage.transform
import random
import ttools.modules
import argparse
import math
import torchvision.models as models
import matplotlib.pyplot as plt
import json


class AdversarialCreator:
    def __init__(self, num_paths=100, n_path_segments=3, max_width=4.0):
        self.num_paths = 100
        self.max_width = max_width
        self.n_path_segments = n_path_segments

        self.classification_model = models.inception_v3(pretrained=True, transform_input=False).cuda()
        self.classification_model.eval()

        self.canvas_width = 229  # sync with inception model inputs
        self.canvas_height = 229

        self.shapes = list()
        self.shape_groups = list()

        # fill shapes and shape_groups with random strokes
        for i in range(num_paths):
            num_control_points = torch.zeros(self.n_path_segments, dtype=torch.int32) + 2
            points = []
            p0 = (random.random(), random.random())
            points.append(p0)
            for j in range(self.n_path_segments):
                radius = 0.05
                p1 = (p0[0] + radius * (random.random() - 0.5), p0[1] + radius * (random.random() - 0.5))
                p2 = (p1[0] + radius * (random.random() - 0.5), p1[1] + radius * (random.random() - 0.5))
                p3 = (p2[0] + radius * (random.random() - 0.5), p2[1] + radius * (random.random() - 0.5))
                points.append(p1)
                points.append(p2)
                points.append(p3)
                p0 = p3
            points = torch.tensor(points)
            points[:, 0] *= self.canvas_width
            points[:, 1] *= self.canvas_height
            path = pydiffvg.Path(num_control_points=num_control_points,
                                 points=points,
                                 stroke_width=torch.tensor(1.0),
                                 is_closed=False)
            self.shapes.append(path)
            path_group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([len(self.shapes) - 1]),
                                             fill_color=None,
                                             stroke_color=torch.tensor([random.random(),
                                                                        random.random(),
                                                                        random.random(),
                                                                        random.random()]))
            self.shape_groups.append(path_group)

        self.render = pydiffvg.RenderFunction.apply

        # initialize variables
        self.point_variables = list()
        self.widths_variables = list()
        self.color_variables = list()
        for path in self.shapes:
            path.points.requires_grad = True
            path.stroke_width.requires_grad = True
            self.point_variables.append(path.points)
            self.widths_variables.append(path.stroke_width)
        for group in self.shape_groups:
            group.stroke_color.requires_grad = True
            self.color_variables.append(group.stroke_color)

    def fit_to_image(self, img_path, n_iterations=100, use_perc_loss=False):
        """
        This method updates internal pathes to visually match provided image.

        It is an updated version of "painterly rendering" https://github.com/BachiLi/diffvg/blob/master/apps/painterly_rendering.py

        :param img_path:
        :param n_iterations:
        :return:
        """
        image = skimage.io.imread(img_path)
        image = skimage.transform.resize(image, (self.canvas_width, self.canvas_height))
        target = torch.from_numpy(image).float().cuda()
        target = target.permute(2, 0, 1).unsqueeze(0)

        plt.imshow(image)
        plt.title('Target image')
        plt.show()

        optim_points = torch.optim.Adam(self.point_variables, lr=1.0)
        optim_widths = torch.optim.Adam(self.widths_variables, lr=0.1)
        optim_color = torch.optim.Adam(self.color_variables, lr=0.05)

        perception_loss = ttools.modules.LPIPS().to(pydiffvg.get_device())

        for iteration in range(n_iterations):
            optim_color.zero_grad()
            optim_widths.zero_grad()
            optim_points.zero_grad()
            scene_args = pydiffvg.RenderFunction.serialize_scene(self.canvas_width, self.canvas_height,
                                                                 self.shapes, self.shape_groups)
            img = self.render(self.canvas_width,
                              self.canvas_height,
                              2, 2, 0, None, *scene_args)
            # Compose img with white background
            img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3,
                                                              device=pydiffvg.get_device()) * (1 - img[:, :, 3:4])
            img = img[:, :, :3]
            img = img.unsqueeze(0)
            img = img.permute(0, 3, 1, 2)
            if use_perc_loss:
                loss = perception_loss(img, target) + (img.mean() - target.mean()).pow(2)
            else:
                loss = (img - target).pow(2).mean()
            print(f'{iteration}: visual loss = {loss.item():.3f}')

            loss.backward()

            optim_points.step()
            optim_widths.step()
            optim_color.step()
            for path in self.shapes:
                path.stroke_width.data.clamp_(1.0, self.max_width)
            for group in self.shape_groups:
                group.stroke_color.data.clamp_(0.0, 1.0)

            if iteration % 50 == 49:
                self.plot()

    def adversarialExample(self, num_steps=100, target_class=291):
        """
        Creates adversarial example by updating rendering parameters via diff rasterizer.
        Maximizes the corresponding class score.

        :param num_steps:
        :param target_class:
        :return:
        """
        optim_points = torch.optim.Adam(self.point_variables, lr=0.01)
        optim_widths = torch.optim.Adam(self.widths_variables, lr=0.01)
        optim_color = torch.optim.Adam(self.color_variables, lr=0.00005)

        for iteration in range(num_steps):
            optim_points.zero_grad()
            img = self.get_image_white_background()
            img = img.unsqueeze(0)
            img = img.permute(0, 3, 1, 2)
            output = self.classification_model.forward(img.cuda())

            target = torch.autograd.Variable(torch.LongTensor([target_class]), requires_grad=False).cuda()
            loss = torch.nn.CrossEntropyLoss()(output, target)
            print(f'{iteration} classification loss: {loss.item()}')

            loss.backward()

            optim_points.step()
            optim_widths.step()
            optim_color.step()
            for path in self.shapes:
                path.stroke_width.data.clamp_(1.0, self.max_width)
            for group in self.shape_groups:
                group.stroke_color.data.clamp_(0.0, 1.0)

            if iteration % 50 == 49:
                label_idx, labelname, label_value = self.get_current_label(class_output=output)
                self.plot(title=f"{labelname} ({label_value:.1f})")

    def render_img(self):
        """
        Renders image

        :return: img - tensor (229, 299, 4) RGBA
        """
        scene_args = pydiffvg.RenderFunction.serialize_scene(
            self.canvas_width,
            self.canvas_height,
            self.shapes,
            self.shape_groups)
        img = self.render(
            self.canvas_width,
            self.canvas_height,
            2, 2, 0, None, *scene_args)
        return img

    def get_image_white_background(self):
        img = self.render_img()
        img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3,
                                                          device='cuda') * (1 - img[:, :, 3:4])
        img = img[:, :, :3]
        return img

    def plot(self, title=None):
        """
        Plot current state as matplotlib image

        :return:
        """
        img = self.render_img()
        plt.imshow(img.detach().cpu())
        plt.title(title)
        plt.axis('off')
        plt.show()

    def compute_current_label(self):
        with torch.no_grad():
            img = self.get_image_white_background()
            img = img.permute(2,0,1).unsqueeze(0)
            output = self.classification_model.forward(img.cuda())[0].cpu()
        return self.get_current_label(class_output=output)

    def get_current_label(self, class_output):
        class_output = class_output.squeeze(0)
        label_idx = torch.argmax(class_output.data).item()
        label_value = torch.max(class_output.data).item()
        with open('data/imagenet1000_labels.json') as f:
            labels = json.load(f)
        labelname = labels[label_idx]
        print(f"Predicted label: {label_idx} '{labelname}' with proba {label_value:.3f}")
        return label_idx, labelname, label_value


if __name__ == "__main__":
    # pydiffvg.set_print_timing(True)
    creator = AdversarialCreator(num_paths=100, n_path_segments=2)
    creator.plot()
    creator.fit_to_image(img_path='data/lion3.jpg', n_iterations=100)
    creator.adversarialExample(num_steps=1000, target_class=293)
