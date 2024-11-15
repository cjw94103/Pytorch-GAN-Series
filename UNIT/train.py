import argparse
import numpy as np
import itertools
import torch
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from dataset import ImageDataset
from models import Generator, Discriminator, weights_init_normal, ResidualBlock, Encoder
from PIL import Image
from train_func import train

## argparse
# argparse
parser = argparse.ArgumentParser()

## prepare dataset
parser.add_argument("--data_path", type=str, help="your custom dataset", default="./data/summer2winter/")
parser.add_argument("--img_height", type=int, help="height of image", default=256)
parser.add_argument("--img_width", type=int, help="width of image", default=256)
parser.add_argument("--channels", type=int, help="channels of image", default=3)

## data generator
parser.add_argument("--num_workers", type=int, help="num workers of dataloader", default=0)
parser.add_argument("--batch_size", type=int, help="num of batch size", default=4)

## model architecture
parser.add_argument("--n_downsample", type=int, help="number downsampling layers in encoder", default=2)
parser.add_argument("--dim", type=int, help="number of filters in first encoder layer", default=64)

## Learning Parameter
parser.add_argument("--epochs", type=int, help="num epochs", default=100)
parser.add_argument("--lr_decay_epoch", type=int, help="epochs of learning rate decay", default=50)

## Loss Parameter
parser.add_argument("--lambda_gan", type=float, help="weight of GAN loss", default=10.0)
parser.add_argument("--lambda_kl_enc", type=float, help="encoded image loss weight of KL loss", default=0.1)
parser.add_argument("--lambda_id", type=float, help="weight of Identity loss", default=100.0)
parser.add_argument("--lambda_kl_trans_enc", type=float, help="encoded translated image loss weight of KL loss", default=0.1)
parser.add_argument("--lambda_cyc", type=float, help="weigth of cycle-consistency loss", default=100.0)

## model save
parser.add_argument("--model_save_path", type=str, help="your model save path", default="./model_result/summer2winter/UNIT_model.pth")
parser.add_argument("--save_per_epochs", type=int, help="How many epochs to save the model", default=50)

## Optimizer parameter
parser.add_argument("--b1", type=float, help="b1 of Adam Optimizer", default=0.5)
parser.add_argument("--b2", type=float, help="b2 of Adam Optimizer", default=0.999)
parser.add_argument("--lr", type=float, help="learning rate", default=0.0002)

args = parser.parse_args()

input_shape = (args.channels, args.img_height, args.img_width)
# channel-wise dimensionalityof image embedding
shared_dim = args.dim * 2 ** args.n_downsample

# make dataloader
transforms_ = [
    transforms.Resize(int(args.img_height * 1.12), Image.BICUBIC),
    transforms.RandomCrop((args.img_height, args.img_width)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

dataloader = DataLoader(
    ImageDataset(args.data_path, transforms_=transforms_, unaligned=True),
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
)

# make model instance
shared_E = ResidualBlock(features=shared_dim)
E1 = Encoder(dim=args.dim, n_downsample=args.n_downsample, shared_block=shared_E)
E2 = Encoder(dim=args.dim, n_downsample=args.n_downsample, shared_block=shared_E)

shared_G = ResidualBlock(features=shared_dim)
G1 = Generator(dim=args.dim, n_upsample=args.n_downsample, shared_block=shared_G)
G2 = Generator(dim=args.dim, n_upsample=args.n_downsample, shared_block=shared_G)

D1 = Discriminator(input_shape)
D2 = Discriminator(input_shape)

E1 = E1.cuda()
E2 = E2.cuda()
G1 = G1.cuda()
G2 = G2.cuda()
D1 = D1.cuda()
D2 = D2.cuda()

# Initialize weights of normal dist.
E1.apply(weights_init_normal)
E2.apply(weights_init_normal)
G1.apply(weights_init_normal)
G2.apply(weights_init_normal)
D1.apply(weights_init_normal)
D2.apply(weights_init_normal)

# get optimizers
optimizer_G = torch.optim.Adam(
    itertools.chain(E1.parameters(), E2.parameters(), G1.parameters(), G2.parameters()),
    lr=args.lr,
    betas=(args.b1, args.b2),
)
optimizer_D1 = torch.optim.Adam(D1.parameters(), lr=args.lr, betas=(args.b1, args.b2))
optimizer_D2 = torch.optim.Adam(D2.parameters(), lr=args.lr, betas=(args.b1, args.b2))

# Training!!
train(args, E1, E2, G1, G2, D1, D2, dataloader, optimizer_G, optimizer_D1, optimizer_D2)