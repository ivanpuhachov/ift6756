import torch
from torch.utils.data import DataLoader, ConcatDataset
import os
from torchvision import transforms
import matplotlib.pyplot as plt

from inception_score import PretrainedInception, frechet_inception_distance
from dataset import QuickDrawBitmapDataset

device='cuda'
datasets = list()
datasets_names = ['apple', 'donut', 'cookie', 'face', 'lollipop']
img_size = 64
for name in datasets_names:
    dataset_path = f"data/bitmap_{name}_train_{img_size}x5000.npy"
    assert os.path.exists(dataset_path)
    datasets.append(
        QuickDrawBitmapDataset(fpath=dataset_path,
                               transform=transforms.Compose([transforms.ToTensor()]),
                               img_size=img_size)
    )
trainset = ConcatDataset(datasets)
dataloader = DataLoader(
        trainset, batch_size=4, shuffle=True, num_workers=8)

pretrainedInception = PretrainedInception().to(device)
print("Calculating FID")
n_iters = 25
real_data = list()
fake_data = list()
ggg = iter(dataloader)
plt.figure(figsize=(5,5))
for i in range(n_iters):
    plt.subplot(5,5,i+1)
    data = next(ggg)
    plt.imshow(data[0][0][0].detach().cpu().numpy(), cmap='gray_r')
    plt.axis("off")
    # real_data.append(data[0].float())
plt.savefig("data.png", dpi=150, bbox_inches='tight')
plt.show()

# for i in range(n_iters):
#     data = next(ggg)
#     fake_data.append(data[0].float())

#
# real_data = torch.cat(real_data, dim=0).repeat(1, 3, 1, 1) * 2 - 1
# fake_data = torch.cat(fake_data, dim=0).repeat(1, 3, 1, 1) * 2 - 1
# print("Computing Frechet stats")
# mu_real, sigma_real = pretrainedInception.compute_frechet_stats(real_data, batch_size=25)
# mu_fake, sigma_fake = pretrainedInception.compute_frechet_stats(fake_data, batch_size=25)
# print("Computing distance")
# fid = frechet_inception_distance(mu_real, sigma_real, mu_fake, sigma_fake)
# print("FID: ", fid)