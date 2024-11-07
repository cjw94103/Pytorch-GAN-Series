import torch
import argparse
import numpy as np
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torchvision import datasets

from models import Generator, Discriminator
from train_func import train

## argparse
parser = argparse.ArgumentParser()

## Prepare data
parser.add_argument("--dataset_name", type=str, help="write dataset name", default="mnist")
parser.add_argument("--n_classes", type=int, default=10, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, help="size of image", default=32)
parser.add_argument("--channels", type=int, help="channels of image", default=1)

## Prepare dataloader
parser.add_argument("--batch_size", type=int, help="num of batch size", default=128)
parser.add_argument("--num_workers", type=int, help="num workers of dataloader", default=0)

## model architecture
parser.add_argument("--latent_dim", type=int, help="num of latent vector dimension", default=100)

## Learning parameters
parser.add_argument("--epochs", type=int, help="num epochs", default=200)

## model save
parser.add_argument("--model_save_path", type=str, help="your model save path", default="./model_result/mnist/CGAN_mnist.pth")
parser.add_argument("--save_per_epochs", type=int, help="How many epochs to save the model", default=50)

## Optimizer parameter
parser.add_argument("--b1", type=float, help="b1 of Adam Optimizer", default=0.5)
parser.add_argument("--b2", type=float, help="b2 of Adam Optimizer", default=0.999)
parser.add_argument("--lr", type=float, help="learning rate", default=0.0002)

args = parser.parse_args(args=[])

img_shape = (args.channels, args.img_size, args.img_size)

## get dataloader
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "./data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(args.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=args.batch_size,
    shuffle=True,
)

## make model instance
generator = Generator(n_classes=args.n_classes, latent_dim=args.latent_dim, img_shape=img_shape)
discriminator = Discriminator(n_classes=args.n_classes, img_shape=img_shape)

generator.cuda()
discriminator.cuda()

## get optimizer
optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

## Train!!
train(args, generator, discriminator, dataloader, optimizer_G, optimizer_D)