# Taken from my project on SketchRNN

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import PIL


def trim(im):
    # taken from https://stackoverflow.com/questions/10615901/trim-whitespace-using-pil
    bg = PIL.Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = PIL.ImageChops.difference(im, bg)
    diff = PIL.ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)
    else:
        print("meh")
        return im


class Drawing:
    """
    This class is for data visualization.
    """
    def __init__(self, emb_sequence):
        assert emb_sequence.shape[-1] == 5
        self.embedding_sequence = emb_sequence

    @classmethod
    def from_npz_data(cls, npzarray):
        newdata = np.zeros(shape=(npzarray.shape[0], 5), dtype=np.float32)
        newdata[:,:2] = npzarray[:,:2]
        newdata[:,2] = 1 - npzarray[:,2]
        newdata[:,3] = npzarray[:,2]
        newdata[-1,3] = 0
        newdata[-1,4] = 1
        return cls(newdata)

    @classmethod
    def from_tensor_prediction(cls, prediction):
        """
        :param prediction: (seq_len, batch_size=1, 5)
        :return:
        """
        return cls(prediction.squeeze(1).detach().cpu().numpy())

    def get_lines(self):
        current_position = np.array([0, 0], dtype=np.float32)
        lines_list = list()
        lines_stroke_id = list()
        stroke_id = 0
        # lines_list.append([current_position.tolist(), self.embedding_sequence[0, :2].tolist()])
        for i_point in range(len(self.embedding_sequence)-1):
            point = self.embedding_sequence[i_point]
            current_position += point[:2]
            if point[2]==1:
                nextpoint_position = current_position + self.embedding_sequence[i_point+1, :2]
                lines_list.append([current_position.tolist(), nextpoint_position.tolist()])
                lines_stroke_id.append(stroke_id)
            else:
                stroke_id += 1
        return lines_list, lines_stroke_id

    def render_image(self, show=False, color=None, linewidth=2):
        lines, lines_id = self.get_lines()
        colors = ['k']
        if len(lines)>1:
            evenly_spaced_interval = np.linspace(0, 1, lines_id[-1]+1)
            colors = [mpl.cm.tab10(x) for x in evenly_spaced_interval]
        plt.axis('equal')
        plt.axis("off")
        for i in range(len(lines)):
            line = lines[i]
            if color is not None:
                plt.plot([line[0][0], line[1][0]], [-line[0][1], -line[1][1]], color=color, linewidth=linewidth)
            else:
                plt.plot([line[0][0], line[1][0]], [-line[0][1], -line[1][1]], color=colors[lines_id[i]], linewidth=linewidth)
        if show:
            plt.show()

    def plot(self):
        self.render_image(show=True, color='k')

    def tensorboard_plot(self):
        self.render_image(show=False)
        canvas = plt.get_current_fig_manager().canvas
        canvas.draw()
        pil_image = PIL.Image.frombytes('RGB', canvas.get_width_height(),
                                        canvas.tostring_rgb())
        plt.close("all")
        img_array = np.asarray(pil_image)
        return np.transpose(img_array, (2, 0, 1))

    def draw_to_numpy(self, img_size=32, linewidth=6):
        plt.figure(figsize=(3,3))
        self.render_image(show=False, color='k', linewidth=linewidth)
        canvas = plt.get_current_fig_manager().canvas
        canvas.draw()
        pil_image = PIL.Image.frombytes('RGB', canvas.get_width_height(),
                                        canvas.tostring_rgb()).convert('L')
        pil_image = trim(pil_image)
        pil_image = pil_image.resize((img_size, img_size), PIL.Image.LANCZOS)
        plt.close('all')
        img_array = 255 - np.asarray(pil_image)
        img_array *= int(255 / np.max(img_array))
        return img_array
        # return np.transpose(img_array, (2, 0, 1))


if __name__ == "__main__":
    a = np.load("data/owl.npz", encoding='latin1', allow_pickle=True)
    drawing = Drawing.from_npz_data(a['valid'][0])
    aaa = drawing.draw_to_numpy(img_size=32)
    print(aaa)
    print(aaa.shape)
    plt.imshow(aaa, cmap='gray_r')
    plt.colorbar()
    plt.show()


    # image = drawing.tensorboard_plot()
    # plt.imshow(image.transpose(), cmap='gray_r')
    # plt.show()
    # print(drawing.get_lines())