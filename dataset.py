from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms


class QuickDrawBitmapDataset(Dataset):
    def __init__(self, fpath='data/full_numpy_bitmap_owl.npy', transform=None):
        self.data = np.load(fpath).reshape(-1, 28, 28, 1).astype(float) / 255.0
        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        if self.transform:
            return self.transform(self.data[item]), 0
        return self.data[item], 0


def test():
    qd = QuickDrawBitmapDataset()
    print(qd[0][0].shape)
    plt.imshow(qd[10][0], cmap='gray_r')
    plt.axis('off')
    plt.colorbar()
    plt.show()

    trainset = QuickDrawBitmapDataset(fpath='data/full_numpy_bitmap_owl.npy',
                                      transform=transforms.Compose([transforms.ToTensor()]))
    print(trainset[0][0].shape)

    trainset = torchvision.datasets.MNIST(
            "data/mnist",
            train=True,
            download=True,
            transform=transforms.Compose([transforms.ToTensor()]),
        )
    print(trainset[0][0].shape)


if __name__=="__main__":
    test()
# plt.imshow(a[0:5].reshape(-1,28,28))
# plt.show()
