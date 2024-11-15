import argparse
import numpy as np
import itertools
import torch
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from dataset import ImageDataset
from models import Generator, Discriminator, weights_init_normal, ResidualBlock, Encoder
from PIL import Image

from torch.autograd import Variable
from tqdm import tqdm

# argparse
parser = argparse.ArgumentParser()

## prepare dataset
parser.add_argument("--data_path", type=str, help="your custom dataset", default="./data/apple2orange/")
parser.add_argument("--img_height", type=int, help="height of image", default=256)
parser.add_argument("--img_width", type=int, help="width of image", default=256)
parser.add_argument("--channels", type=int, help="channels of image", default=3)

## data generator
parser.add_argument("--num_workers", type=int, help="num workers of dataloader", default=0)
parser.add_argument("--batch_size", type=int, help="num of batch size", default=4)

## model architecture
parser.add_argument("--n_downsample", type=int, help="number downsampling layers in encoder", default=2)
parser.add_argument("--dim", type=int, help="number of filters in first encoder layer", default=64)

## model save
parser.add_argument("--model_save_path", type=str, help="your model save path", default="./model_result/apple2orange/UNIT_model_100epochs.pth")

args = parser.parse_args(args=[])

input_shape = (args.channels, args.img_height, args.img_width)

# channel-wise dimensionalityof image embedding
shared_dim = args.dim * 2 ** args.n_downsample

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
    num_workers=args.num_workers,
)

# load trained model
weights = torch.load(args.model_save_path)

shared_E = ResidualBlock(features=shared_dim)
E1 = Encoder(dim=args.dim, n_downsample=args.n_downsample, shared_block=shared_E)
E2 = Encoder(dim=args.dim, n_downsample=args.n_downsample, shared_block=shared_E)

shared_G = ResidualBlock(features=shared_dim)
G1 = Generator(dim=args.dim, n_upsample=args.n_downsample, shared_block=shared_G)
G2 = Generator(dim=args.dim, n_upsample=args.n_downsample, shared_block=shared_G)

E1 = E1.cuda()
E2 = E2.cuda()
G1 = G1.cuda()
G2 = G2.cuda()

# apply trained weigths
E1.load_state_dict(weights['E1'])
E2.load_state_dict(weights['E2'])
G1.load_state_dict(weights['G1'])
G2.load_state_dict(weights['G2'])

E1.eval()
E2.eval()
G1.eval()
G2.eval()

# inference
# for type casting
Tensor = torch.cuda.FloatTensor

a_image_list = []
b_image_list = []
a2b_image_list = []
b2a_image_list = []

for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
    X1 = Variable(batch["A"].type(Tensor))
    X2 = Variable(batch["B"].type(Tensor))

    mu1, Z1 = E1(X1)
    mu2, Z2 = E2(X2)

    # Translate images
    a2b = G2(Z1)
    b2a = G1(Z1)
    
    a_image = np.transpose(X1.data.cpu().numpy(), (0, 2, 3, 1))
    a2b_image = np.transpose(a2b.data.cpu().numpy(), (0, 2, 3, 1))
    b2a_image = np.transpose(b2a.data.cpu().numpy(), (0, 2, 3, 1))
    
    a_image_list.extend(a_image)
    a2b_image_list.extend(a2b_image)
    b2a_image_list.extend(b2a_image)

## Image Decoding
a_image_list = np.array(a_image_list)
a2b_image_list = np.array(a2b_image_list)
b2a_image_list = np.array(b2a_image_list)

a_image_list = np.uint8((0.5 + (a_image_list*0.5)) * 255)
a2b_image_list = np.uint8((0.5 + (a2b_image_list*0.5)) * 255)
b2a_image_list = np.uint8((0.5 + (b2a_image_list*0.5)) * 255)

## Visualization
se_idx = 3
Image.fromarray(a_image_list[se_idx])
Image.fromarray(a2b_image_list[se_idx])
Image.fromarray(b2a_image_list[se_idx])