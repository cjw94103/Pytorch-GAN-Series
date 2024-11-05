import torch
import numpy as np
import itertools
import argparse

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset import ImageDataset
from PIL import Image
from models import ResnetGenerator_our, NLayerDiscriminator
from train_func import train

from utils import *

# argparse
parser = argparse.ArgumentParser()

## prepare dataset
parser.add_argument("--data_path", type=str, help="your custom dataset", default="./data/apple2orange/")
parser.add_argument("--img_height", type=int, help="height of image", default=256)
parser.add_argument("--img_width", type=int, help="width of image", default=256)
parser.add_argument("--channels", type=int, help="channels of image", default=256)

## data generator
parser.add_argument("--num_workers", type=int, help="num workers of generator", default=0)
parser.add_argument("--batch_size", type=int, help="num of batch size", default=8)

## model architecture
parser.add_argument("--n_blocks", type=int, help="num residual blocks of GAN generator", default=9)
parser.add_argument("--ngf", type=int, help="num conv filters of GAN generator, The number of convs increases as a power of 2", default=64)
parser.add_argument("--n_layers", type=int, help="num layers blokc of GAN discriminator", default=4)
parser.add_argument("--ndf", type=int, help="num conv filters of GAN discriminator, The number of convs increases as a power of 2", default=64)

## Learning Parameter
parser.add_argument("--epochs", type=int, help="num epochs", default=200)
parser.add_argument("--lr_decay_epoch", type=int, help="epochs of learning rate decay", default=100)

## Loss Parameter
parser.add_argument("--lambda_cyc", type=float, help="loss parameter of lambda cycle", default=10.0)
parser.add_argument("--lambda_id", type=float, help="loss parameter of identity loss", default=5.0)

## model save
parser.add_argument("--model_save_path", type=str, help="your model save path", default="./model_result/Apple2Orange/Translation_Model.pth")
parser.add_argument("--save_per_epochs", type=int, help="How many epochs to save the model", default=20)
parser.add_argument("--multi_gpu_flag", type="store_true", help="Whether to use multiple GPUs, False in train.py")
parser.add_argument("--port_num", type=int, help="Which port to use when learning multi-GPU", default=14000)

## Optimizer parameter
parser.add_argument("--b1", type=float, help="b1 of Adam Optimizer", default=0.5)
parser.add_argument("--b2", type=float, help="b2 of Adam Optimizer", default=0.999)
parser.add_argument("--lr", type=float, help="learning rate", default=0.0002)

args = parser.parse_args()

# make dataloader
train_transforms_ = [
    transforms.Resize(int(args.img_height * 1.12), Image.BICUBIC),
    transforms.RandomCrop((args.img_height, args.img_width)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

train_dataloader =  DataLoader(
    ImageDataset(args.data_path, transforms_=train_transforms_, unaligned=True, mode='train'),
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers)

# load model
G_AB = ResnetGenerator_our(input_nc=args.channels, output_nc=None, ngf=args.ngf, n_blocks=args.n_blocks).to('cuda')
G_BA = ResnetGenerator_our(input_nc=args.channels, output_nc=None, ngf=args.ngf, n_blocks=args.n_blocks).to('cuda')
D_A = NLayerDiscriminator(input_nc=args.channels, ndf=args.ndf, n_layers=args.n_layers, norm_layer=torch.nn.InstanceNorm2d).to('cuda')
D_B = NLayerDiscriminator(input_nc=args.channels, ndf=args.ndf, n_layers=args.n_layers, norm_layer=torch.nn.InstanceNorm2d).to('cuda')

# optimizer
optimizer_G = torch.optim.Adam(itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=args.lr, betas=(args.b1, args.b2))
optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=args.lr, betas=(args.b1, args.b2))
optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=args.lr, betas=(args.b1, args.b2))

# train
train(args, G_AB, G_BA, D_A, D_B, train_dataloader, optimizer_G, optimizer_D_A, optimizer_D_B)