# Inception score rewritten based on https://github.com/sbarratt/inception-score-pytorch/blob/master/inception_score.py
# Frechet Inception Distance rewritten based on https://github.com/hukkelas/pytorch-frechet-inception-distance/blob/master/fid.py

import torch
from torch import nn

from torchvision.models.inception import inception_v3
import torchvision.datasets as dset
import torchvision.transforms as transforms

import numpy as np
import scipy
from scipy.stats import entropy


class PretrainedInception(nn.Module):
    def __init__(self, device='cuda'):
        super(PretrainedInception, self).__init__()
        self.device = device
        self.model = inception_v3(pretrained=True, transform_input=False).to(device)
        self.model.eval()
        self.last_features = 0
        # forward hook example: https://discuss.pytorch.org/t/how-can-l-load-my-best-model-as-a-feature-extractor-evaluator/17254/5
        # architecture: https://github.com/pytorch/vision/blob/master/torchvision/models/inception.py
        self.model.Mixed_7c.register_forward_hook(self.my_forward_hook)
        self.avg_pooling = self.model.avgpool
        self.upsample = nn.Upsample(size=(299, 299), mode='bilinear', align_corners=True)

    def my_forward_hook(self, module, input, output):
        self.last_features = output.detach()

    def forward_no_grad(self, x):
        with torch.no_grad():
            if x.shape[-1] != 299:
                x = self.upsample(x)
            x = self.model(x)
            return torch.softmax(x, dim=-1)

    def get_last_features(self, x):
        outs = self.forward_no_grad(x)
        return self.avg_pooling(self.last_features).squeeze(2).squeeze(2)

    def compute_frechet_stats(self, imgs, batch_size=32):
        dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)
        features = list()
        for i, data in enumerate(dataloader):
            features.append(self.get_last_features(data))
        features = torch.cat(features, dim=0).cpu().numpy()
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        return mu, sigma

    def inception_score(self, imgs, batch_size=32, splits=1):
        """
        Computes the inception score (mean and std of KL divergence) of the generated images imgs
        CIFAR10 score: (9.672773911271202, 0.14991434585662258)

        imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
        cuda -- whether or not to run on GPU
        batch_size -- batch size for feeding into Inception v3
        splits -- number of splits
        """
        N = len(imgs)

        assert batch_size > 0
        assert N > batch_size

        # Set up dataloader
        dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

        # Get predictions
        preds = list()

        for i, batch in enumerate(dataloader, 0):
            preds.append(self.forward_no_grad(batch.to(self.device)))

        preds = torch.cat(preds, dim=0).cpu().numpy()

        # Now compute the mean kl-div
        split_scores = []

        for k in range(splits):
            part = preds[k * (N // splits): (k+1) * (N // splits), :]
            py = np.mean(part, axis=0)
            scores = []
            for i in range(part.shape[0]):
                pyx = part[i, :]
                scores.append(entropy(pyx, py))
            split_scores.append(np.exp(np.mean(scores)))

        return np.mean(split_scores), np.std(split_scores)


# Modified from: https://github.com/bioinf-jku/TTUR/blob/master/fid.py
def frechet_inception_distance(mu_real, sigma_real, mu_fake, sigma_fake, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

        Stable version by Dougal J. Sutherland.
        Params:
        -- mu1 : Numpy array containing the activations of the pool_3 layer of the
                 inception net ( like returned by the function 'get_predictions')
                 for generated samples.
        -- mu2   : The sample mean over activations of the pool_3 layer, precalcualted
                   on an representive data set.
        -- sigma1: The covariance matrix over activations of the pool_3 layer for
                   generated samples.
        -- sigma2: The covariance matrix over activations of the pool_3 layer,
                   precalcualted on an representive data set.
        Returns:
        --   : The Frechet Distance.
        """

    mu1 = np.atleast_1d(mu_real)
    mu2 = np.atleast_1d(mu_fake)

    sigma1 = np.atleast_2d(sigma_real)
    sigma2 = np.atleast_2d(sigma_fake)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2
    # product might be almost singular
    covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
        # warnings.warn(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


class IgnoreLabelDataset(torch.utils.data.Dataset):
    def __init__(self, orig):
        self.orig = orig

    def __getitem__(self, index):
        return self.orig[index][0]

    def __len__(self):
        return len(self.orig)


if __name__ == '__main__':
    device = 'cuda'
    a = torch.rand(size=(100, 3, 32, 32)).to(device)
    pretrained = PretrainedInception().to(device)
    with torch.no_grad():
        b = pretrained.forward_no_grad(a)
        c = pretrained.get_last_features(a)
        mu, sigma = pretrained.compute_frechet_stats(a)
    print("Forward shape: ", b.shape)
    print("Last features shape: ", c.shape)
    print("Frechet mu shape: ", mu.shape)
    print("Frechet sigma shape: ", sigma.shape)
    print("Identity frechet inception distance: ", frechet_inception_distance(mu, sigma, mu, sigma))

    # cifar = dset.CIFAR10(root='data/', download=True,
    #                          transform=transforms.Compose([
    #                              transforms.Scale(32),
    #                              transforms.ToTensor(),
    #                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    #                          ])
    # )
    #
    # IgnoreLabelDataset(cifar)
    #
    # print("Calculating Inception Score...")
    # with torch.no_grad():
    #     print(pretrained.inception_score(IgnoreLabelDataset(cifar), batch_size=32, splits=10))