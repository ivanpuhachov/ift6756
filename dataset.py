from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms


class QuickDrawBitmapDataset(Dataset):
    def __init__(self, fpath='data/full_numpy_bitmap_owl.npy', img_size=28, transform=None, category=0):
        self.img_size = img_size
        self.data = np.load(fpath).reshape(-1, img_size, img_size, 1).astype(float) / 255.0
        self.transform = transform
        self.category = category

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        if self.transform:
            return self.transform(self.data[item]), self.category
        return self.data[item], self.category


def test():
    print("standard bitmap QuickDraw")
    qd = QuickDrawBitmapDataset()
    print(len(qd))
    print(qd[0][0].shape)
    plt.imshow(qd[10][0], cmap='gray_r')
    # plt.axis('off')
    plt.colorbar()
    plt.show()

    # trainset = QuickDrawBitmapDataset(fpath='data/full_numpy_bitmap_owl.npy',
    #                                   transform=transforms.Compose([transforms.ToTensor()]))
    print("my bitmap quickdraw")
    trainset = QuickDrawBitmapDataset(fpath='data/bitmap_owl_train_32x10000.npy',
                                      img_size=32,
                                      )
    print(len(trainset))
    print(trainset[0][0].shape)
    plt.imshow(trainset[2][0], cmap='gray_r')
    # plt.axis('off')
    plt.colorbar()
    plt.show()

    print("MNIST")
    trainset = torchvision.datasets.MNIST(
            "data/mnist",
            train=True,
            download=True,
            transform=transforms.Compose([transforms.ToTensor()]),
        )
    print(len(trainset))
    print(trainset[0][0].shape)



if __name__=="__main__":
    test()
# plt.imshow(a[0:5].reshape(-1,28,28))
# plt.show()
