import pydiffvg
import torch
import skimage
import skimage.io
import skimage.transform
import matplotlib.pyplot as plt
import json

from adversarial_examples import AdversarialCreator


class AdversarialVandal(AdversarialCreator):
    def __init__(self, img_path, num_paths=100, n_path_segments=3, max_width=4.0, min_width=1.0):
        super(AdversarialVandal, self).__init__(num_paths=num_paths,
                                                n_path_segments=n_path_segments,
                                                max_width=4.0,
                                                min_width=1.0)
        # for group in self.shape_groups:
        #     group.stroke_color = torch.tensor([1,1,1,1])
        #     group.stroke_color.requires_grad = False
        self.img_path = img_path
        image = skimage.io.imread(img_path)
        image = skimage.transform.resize(image, (self.canvas_width, self.canvas_height))
        self.target_image = torch.from_numpy(image).float().cuda()[:,:,:3]
        plt.imshow(image)
        plt.title('Target image')
        plt.show()

    def render_vandalized_img(self):
        img = self.render_img()[:, :, :3] + self.target_image
        img = torch.clamp(img, min=0, max=1.0)
        return img

    def plot(self, title=None, name=None):
        with torch.no_grad():
            img = self.render_vandalized_img()
        plt.imshow(img.detach().cpu())
        plt.title(title)
        plt.axis('off')
        if name is not None:
            plt.savefig(name, bbox_inches='tight', dpi=150)
        plt.show()

    def vandalize_image(self, target_class=51, num_steps=100):
        label_idx, labelname, label_value = self.compute_current_label()
        self.plot(title=f"{labelname} ({label_value:.1f})")

        optim_points = torch.optim.Adam(self.point_variables, lr=0.1)
        optim_widths = torch.optim.Adam(self.widths_variables, lr=0.01)
        optim_color = torch.optim.Adam(self.color_variables, lr=0.005)

        for iteration in range(num_steps):
            optim_points.zero_grad()
            img = self.render_vandalized_img()
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
                path.stroke_width.data.clamp_(self.min_width, self.max_width)
            for group in self.shape_groups:
                group.stroke_color.data.clamp_(0.0, 1.0)

            if iteration % self.frequency_plot == self.frequency_plot-1:
                label_idx, labelname, label_value = self.get_current_label(class_output=output)
                self.plot(title=f"{labelname} ({label_value:.1f})")

    def compute_current_label(self):
        with torch.no_grad():
            img = self.render_vandalized_img()
            img = img.permute(2,0,1).unsqueeze(0)
            output = self.classification_model.forward(img.cuda())[0].cpu()
        return self.get_current_label(class_output=output)


if __name__ == "__main__":
    # vandal = AdversarialVandal(img_path='images/picasso3.png', min_width=0.5, max_width=1.1, num_paths=5)
    # vandal = AdversarialVandal(img_path='images/boulder.png', min_width=0.5, max_width=1.5, num_paths=10)
    vandal = AdversarialVandal(img_path='images/lion2.png', min_width=0.5, max_width=1.5, num_paths=10)

    vandal.vandalize_image(num_steps=1000, target_class=254)
    vandal.save_svg(name='images/vandal.svg')
    label_idx, labelname, label_value = vandal.compute_current_label()
    vandal.plot(title=f"{labelname} ({label_value:.1f})", name='images/vandal.png')
