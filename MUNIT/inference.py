import argparse
import numpy as np
import torch
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from dataset import ImageDataset
from PIL import Image
from models import Encoder, Decoder, MultiDiscriminator
from torch.autograd import Variable

# argparse
parser = argparse.ArgumentParser()

## prepare dataset
parser.add_argument("--data_path", type=str, help="your custom dataset", default="./data/edges2shoes/")
parser.add_argument("--img_height", type=int, help="height of image", default=128)
parser.add_argument("--img_width", type=int, help="width of image", default=128)
parser.add_argument("--channels", type=int, help="channels of image", default=3)

## data generator
parser.add_argument("--num_workers", type=int, help="num workers of dataloader", default=0)
parser.add_argument("--batch_size", type=int, help="num of batch size", default=16)

## model architecture
parser.add_argument("--n_downsample", type=int, help="number downsampling layers in encoder", default=2)
parser.add_argument("--n_residual", type=int, help="number residual blocks in encoder and decoder", default=3)
parser.add_argument("--dim", type=int, help="number of filters in first encoder layer", default=64)
parser.add_argument("--style_dim", type=int, help="dim. of style code", default=8)

## model save
parser.add_argument("--model_save_path", type=str, help="your model save path", default="./model_result/edges2shoes/MUNIT_model_100epochs.pth")

args = parser.parse_args(args=[])

# make dataloader
transforms_ = [
    transforms.Resize((args.img_height, args.img_width), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

dataloader = DataLoader(
    ImageDataset(args.data_path, transforms_=transforms_, unaligned=False, mode='test'),
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers)

# load trained model
weights = torch.load(args.model_save_path)

Enc1 = Encoder(dim=args.dim, n_downsample=args.n_downsample, n_residual=args.n_residual, style_dim=args.style_dim)
Dec1 = Decoder(dim=args.dim, n_upsample=args.n_downsample, n_residual=args.n_residual, style_dim=args.style_dim)
Enc2 = Encoder(dim=args.dim, n_downsample=args.n_downsample, n_residual=args.n_residual, style_dim=args.style_dim)
Dec2 = Decoder(dim=args.dim, n_upsample=args.n_downsample, n_residual=args.n_residual, style_dim=args.style_dim)

Enc1 = Enc1.cuda()
Dec1 = Dec1.cuda()
Enc2 = Enc2.cuda()
Dec2 = Dec2.cuda()

Enc1.load_state_dict(weights['Enc1'])
Dec1.load_state_dict(weights['Dec1'])
Enc2.load_state_dict(weights['Enc2'])
Dec2.load_state_dict(weights['Dec2'])

# inference
for batch in dataloader:
    X1 = Variable(batch["A"].type(torch.cuda.FloatTensor))
    X2 = Variable(batch["B"].type(torch.cuda.FloatTensor))

    style_1 = Variable(torch.randn(X1.size(0), args.style_dim, 1, 1).type(torch.cuda.FloatTensor))
    style_2 = Variable(torch.randn(X1.size(0), args.style_dim, 1, 1).type(torch.cuda.FloatTensor))

    c_code_1, s_code_1 = Enc1(X1)
    c_code_2, s_code_2 = Enc2(X2)

    X21 = Dec1(c_code_2, style_1)
    X12 = Dec2(c_code_1, style_2)

    break
    
origin_a_image = np.transpose(X1.data.cpu().numpy(), (0, 2, 3, 1))
origin_a_image = np.uint8((0.5 + (origin_a_image*0.5)) * 255)

origin_b_image = np.transpose(X2.data.cpu().numpy(), (0, 2, 3, 1))
origin_b_image = np.uint8((0.5 + (origin_b_image*0.5)) * 255)

a2b_image = np.transpose(X12.data.cpu().numpy(), (0, 2, 3, 1))
a2b_image = np.uint8((0.5 + (a2b_image*0.5)) * 255)

# view
## origin a image
se_idx = 4
Image.fromarray(origin_a_image[se_idx])

## translated image
Image.fromarray(a2b_image[se_idx])