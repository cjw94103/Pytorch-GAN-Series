import numpy as np
import torch
import torchvision.transforms as transforms
import argparse
import cv2

from dataset import ImageDataset
from PIL import Image
from models import GeneratorResNet, Discriminator, weights_init_normal
from torch.utils.data import DataLoader
from tqdm import tqdm

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
parser.add_argument("--n_residual_blocks", type=int, help="num residual blocks of GAN generator", default=9)

## model save
parser.add_argument("--model_save_path", type=str, help="your model save path", default="./model_result/Apple2Orange/Translation_Model.pth")

args = parser.parse_args()

# make dataloader
test_transforms_ = [
    transforms.Resize(int(args.img_height * 1.12), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]
testdataset = ImageDataset(args.data_path, transforms_=test_transforms_, unaligned=False, mode='test')
test_dataloader = DataLoader(
    testdataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers)

# load trained model
weights = torch.load(args.model_save_path)

input_shape = (args.channels, args.img_height, args.img_width)
G_AB = GeneratorResNet(input_shape, args.n_residual_blocks)
G_BA = GeneratorResNet(input_shape, args.n_residual_blocks)
G_AB = G_AB.load_state_dict(weights['G_AB'])
G_BA = G_BA.load_state_dict(weights['G_BA'])

G_AB.to("cuda").eval()
G_BA.to("cuda").eval()

# inference (for one image, example code)
se_idx = 3
sample = testdataset[se_idx]
img_A, img_B = sample['A'].unsqueeze(0).to('cuda'), sample['B'].unsqueeze(0).to('cuda')

A2B = G_AB(img_A)
B2A = G_BA(A2B)

# pytorch tensor to numpy array
A2B = np.transpose(A2B[0].data.cpu().numpy(), (1, 2, 0))
B2A = np.transpose(B2A[0].data.cpu().numpy(), (1, 2, 0))

A2B_origin = np.uint8(((A2B * 0.5) + 0.5) * 255)
B2A_origin = np.uint8(((B2A * 0.5) + 0.5) * 255)

img_A_origin = np.transpose(img_A[0].data.cpu().numpy(), (1, 2, 0))
img_A_origin = np.uint8(((img_A_origin * 0.5) + 0.5) * 255)

# save generated image
cv2.imwrite("./A2B_image.jpg", A2B_origin)
cv2.imwrite("./B2A_image.jpg", B2A_origin)