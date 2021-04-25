import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, ConcatDataset
import torchvision
from torchvision import transforms

import os
import shutil
import argparse
from datetime import datetime
import numpy as np
import json
from shutil import copyfile

from wgangp_trainer import Trainer, WGAN_Trainer, WGANGP_Trainer, LOGAN_GD_Trainer, LOGAN_NGD_Trainer, GTTrainer, AllIncluded_Trainer
from simple_gan import SimpleGAN
from vector_gan import SimpleGANBezier, BezierGAN, BezierSNGAN, AwesomeBezierGAN
from dataset import QuickDrawBitmapDataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GAN!")
    # parser.add_argument("--dataset", "-d", default="data/full_numpy_bitmap_owl.npy")
    parser.add_argument("--dataset", "-d", default="data/bitmap_owl_train_96x5000.npy")
    parser.add_argument("--imgsize", type=int, default=64, help="generated img size, MUST match with dataset img size")
    parser.add_argument("--n_epochs", "-n", type=int, default=40, help="number of training epochs")
    parser.add_argument("--batch", "-b", type=int, default=4, help="batch_size")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--seed", type=int, default=23, help='random seed')
    parser.add_argument("--workers", type=int, default=8, help='num_workers to use')
    parser.add_argument('--beziergan', dest='beziergan', default=False, action='store_true', help='use BezierGAN')
    parser.add_argument('--simplebeziergan', dest='simplebeziergan', default=False, action='store_true', help='use BezierGAN')
    parser.add_argument('--simplegan', dest='simplegan', default=False, action='store_true', help='use SimpleGAN')

    print("\n-- parsing args --")
    args = parser.parse_args()
    print("Received keys:")
    for key in vars(args):
        print(f"{key} \t\t {getattr(args, key)}")
    # datasets_names = ['apple', 'carrot', 'cat', 'fish', 'owl', 'creativebirds']
    # datasets_names = ['apple', 'carrot', 'cat', 'fish', 'owl']
    datasets_names = ['apple', 'donut', 'cookie', 'face', 'lollipop']
    dataset_path = args.dataset
    img_size = args.imgsize
    batch_size = args.batch
    n_epochs = args.n_epochs
    learning_rate = args.lr
    random_seed = args.seed
    num_workers = args.workers
    use_beziergan = args.beziergan
    use_simplebeziergan = args.simplebeziergan
    use_simplegan = args.simplegan

    # assert (os.path.exists(dataset_path))

    print("\n-- logs and backup  --")
    log_dir = "logs/" + datetime.now().strftime("%m.%d.%H.%M") + f"_n{n_epochs}/"
    os.mkdir(log_dir)

    with open(log_dir + "cli_args.txt", 'w') as f:
        json.dump(args.__dict__, f)

    # reproducibility
    # assert torch.cuda.is_available()
    # torch.manual_seed(seed=random_seed)
    # np.random.seed(random_seed)
    # # torch.set_deterministic(True)
    # torch.backends.cudnn.benchmark = False

    print("\n-- datasets --")
    datasets = list()
    for name in datasets_names:
        dataset_path = f"data/bitmap_{name}_train_{img_size}x5000.npy"
        assert os.path.exists(dataset_path)
        datasets.append(
            QuickDrawBitmapDataset(fpath=dataset_path,
                                   transform=transforms.Compose([transforms.ToTensor()]),
                                   img_size=img_size)
        )
    trainset = ConcatDataset(datasets)

    # trainset = QuickDrawBitmapDataset(fpath="data/full_numpy_bitmap_owl.npy",
    #                                   transform=transforms.Compose([transforms.ToTensor()]),
    #                                   img_size=img_size)

    # trainset = QuickDrawBitmapDataset(fpath=dataset_path,
    #                                   transform=transforms.Compose([transforms.ToTensor()]),
    #                                   img_size=img_size)

    # trainset = torchvision.datasets.MNIST(
    #         "data/mnist",
    #         train=True,
    #         download=True,
    #         transform=transforms.Compose([transforms.ToTensor()]),
    #     )

    dataloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    print("\n-- creating model --")

    # model = SimpleGAN()
    # model = SimpleGANBezier(img_size=img_size)
    # model = BezierSNGAN(img_size=img_size)
    model = AwesomeBezierGAN(img_size=img_size, n_circles=5)

    if use_simplebeziergan:
        model = SimpleGANBezier()

    if use_beziergan:
        model = BezierGAN()

    trainer = GTTrainer(model, dataloader=dataloader, log_dir=log_dir, lr=learning_rate)

    # trainer.load_from_checkpoint("logs/04.02--23_b100_n1200/checkpoints/checkpoint_699.pth")

    print("\n-- training --")

    trainer.train(n_epochs=n_epochs)
