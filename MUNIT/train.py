import argparse
import numpy as np
import itertools
import torch
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from dataset import ImageDataset
from models import Encoder, Decoder, MultiDiscriminator, weights_init_normal
from PIL import Image
from train_func import train

parser = argparse.ArgumentParser()

## prepare dataset
parser.add_argument("--data_path", type=str, help="your custom dataset", default="./data/edges2shoes/")
parser.add_argument("--img_height", type=int, help="height of image", default=128)
parser.add_argument("--img_width", type=int, help="width of image", default=128)
parser.add_argument("--channels", type=int, help="channels of image", default=3)

## data generator
parser.add_argument("--num_workers", type=int, help="num workers of dataloader", default=0)
parser.add_argument("--batch_size", type=int, help="num of batch size", default=4)

## model architecture
parser.add_argument("--n_downsample", type=int, help="number downsampling layers in encoder", default=2)
parser.add_argument("--n_residual", type=int, help="number residual blocks in encoder and decoder", default=3)
parser.add_argument("--dim", type=int, help="number of filters in first encoder layer", default=64)
parser.add_argument("--style_dim", type=int, help="dim. of style code", default=8)

## Learning Parameter
parser.add_argument("--epochs", type=int, help="num epochs", default=100)
parser.add_argument("--lr_decay_epoch", type=int, help="epochs of learning rate decay", default=50)

## Loss Parameter
parser.add_argument("--lambda_gan", type=float, help="weight of GAN loss", default=1.0)
parser.add_argument("--lambda_id", type=float, help="weight of Identity loss", default=10.0)
parser.add_argument("--lambda_style", type=float, help="weight of Style loss", default=1.0)
parser.add_argument("--lambda_cont", type=float, help="weight of Contents loss", default=1.0)
parser.add_argument("--lambda_cyc", type=float, help="weigth of cycle-consistency loss", default=0.0)

## model save
parser.add_argument("--model_save_path", type=str, help="your model save path", default="./model_result/edges2shoes/MUNIT_model.pth")
parser.add_argument("--save_per_epochs", type=int, help="How many epochs to save the model", default=50)

## Optimizer parameter
parser.add_argument("--b1", type=float, help="b1 of Adam Optimizer", default=0.5)
parser.add_argument("--b2", type=float, help="b2 of Adam Optimizer", default=0.999)
parser.add_argument("--lr", type=float, help="learning rate", default=0.0002)

args = parser.parse_args()

# make dataloader
transforms_ = [
    transforms.Resize((args.img_height, args.img_width), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

dataloader = DataLoader(
    ImageDataset(args.data_path, transforms_=transforms_, unaligned=True),
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers)

# make model instance
Enc1 = Encoder(dim=args.dim, n_downsample=args.n_downsample, n_residual=args.n_residual, style_dim=args.style_dim)
Dec1 = Decoder(dim=args.dim, n_upsample=args.n_downsample, n_residual=args.n_residual, style_dim=args.style_dim)
Enc2 = Encoder(dim=args.dim, n_downsample=args.n_downsample, n_residual=args.n_residual, style_dim=args.style_dim)
Dec2 = Decoder(dim=args.dim, n_upsample=args.n_downsample, n_residual=args.n_residual, style_dim=args.style_dim)
D1 = MultiDiscriminator()
D2 = MultiDiscriminator()

Enc1 = Enc1.cuda()
Dec1 = Dec1.cuda()
Enc2 = Enc2.cuda()
Dec2 = Dec2.cuda()
D1 = D1.cuda()
D2 = D2.cuda()

Enc1.apply(weights_init_normal)
Dec1.apply(weights_init_normal)
Enc2.apply(weights_init_normal)
Dec2.apply(weights_init_normal)
D1.apply(weights_init_normal)
D2.apply(weights_init_normal)

# get optimizer
optimizer_G = torch.optim.Adam(itertools.chain(Enc1.parameters(), Dec1.parameters(), Enc2.parameters(), Dec2.parameters()),lr=args.lr,betas=(args.b1, args.b2))
optimizer_D1 = torch.optim.Adam(D1.parameters(), lr=args.lr, betas=(args.b1, args.b2))
optimizer_D2 = torch.optim.Adam(D2.parameters(), lr=args.lr, betas=(args.b1, args.b2))

# train!!
train(args, Enc1, Enc2, Dec1, Dec2, D1, D2, dataloader, optimizer_G, optimizer_D1, optimizer_D2)