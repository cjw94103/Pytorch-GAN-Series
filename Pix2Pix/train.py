import torch
import argparse
import numpy as np
import torchvision.transforms as transforms

from models import GeneratorUNet, Discriminator, weights_init_normal
from torch.utils.data import DataLoader
from dataset import ImageDataset
from PIL import Image
from train_func import train

## argparse
parser = argparse.ArgumentParser()

## prepare dataset
parser.add_argument("--dataset_name", type=str, help="pix2pix dataset name", default="maps")
parser.add_argument("--img_height", type=int, help="height of image", default=256)
parser.add_argument("--img_width", type=int, help="width of image", default=256)
parser.add_argument("--channels", type=int, help="channels of image", default=3)

## data generator
parser.add_argument("--num_workers", type=int, help="num workers of dataloader", default=0)
parser.add_argument("--batch_size", type=int, help="num of batch size", default=16)

## Learning Parameter
parser.add_argument("--epochs", type=int, help="num epochs", default=200)

## Loss Parameter
parser.add_argument("--lambda_pixel", type=float, help="loss parameter of lambda pixel", default=100.0)

## model save
parser.add_argument("--model_save_path", type=str, help="your model save path", default="./model_result/maps/Pix2Pix_Model.pth")
parser.add_argument("--save_per_epochs", type=int, help="How many epochs to save the model", default=40)

## Optimizer parameter
parser.add_argument("--b1", type=float, help="b1 of Adam Optimizer", default=0.5)
parser.add_argument("--b2", type=float, help="b2 of Adam Optimizer", default=0.999)
parser.add_argument("--lr", type=float, help="learning rate", default=0.0002)

args = parser.parse_args()

## make dataloader
transforms_ = [
    transforms.Resize((args.img_height, args.img_width), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

dataloader = DataLoader(
    ImageDataset("./data/%s" % args.dataset_name, transforms_=transforms_),
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
)

val_dataloader = DataLoader(
    ImageDataset("./data/%s" % args.dataset_name, transforms_=transforms_, mode="val"),
    batch_size=10,
    shuffle=True,
    num_workers=args.num_workers,
)

## make model instance
generator = GeneratorUNet()
discriminator = Discriminator()

generator = generator.cuda()
discriminator = discriminator.cuda()

generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

## get optimizer
optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

## train
train(args, generator, discriminator, dataloader, optimizer_G, optimizer_D)