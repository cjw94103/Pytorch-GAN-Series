import torch
import argparse
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from torch.autograd import Variable
from models import GeneratorUNet, Discriminator
from torch.utils.data import DataLoader
from dataset import ImageDataset
from PIL import Image
from tqdm import tqdm

## argparse
parser = argparse.ArgumentParser()

## prepare dataset
parser.add_argument("--dataset_name", type=str, help="pix2pix dataset name", default="maps")
parser.add_argument("--img_height", type=int, help="height of image", default=256)
parser.add_argument("--img_width", type=int, help="width of image", default=256)
parser.add_argument("--channels", type=int, help="channels of image", default=3)

## data generator
parser.add_argument("--num_workers", type=int, help="num workers of dataloader", default=0)
parser.add_argument("--batch_size", type=int, help="num of batch size", default=10)

## model save
parser.add_argument("--model_save_path", type=str, help="your model save path", default="./model_result/maps/Pix2Pix_Model_200epochs.pth")

args = parser.parse_args(args=[])

## make dataloader
transforms_ = [
    transforms.Resize((args.img_height, args.img_width), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

val_dataloader = DataLoader(
    ImageDataset("./data/%s" % args.dataset_name, transforms_=transforms_, mode="val"),
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
)

## load trained model
weights = torch.load(args.model_save_path)

generator = GeneratorUNet()
generator.load_state_dict(weights['G'])
generator = generator.cuda()
generator.eval()

## inference
Tensor = torch.cuda.FloatTensor

input_image_list = []
generated_image_list = []
original_image_list = []

for batch in tqdm(val_dataloader, total=len(val_dataloader)):
    real_img_A, real_img_B = batch['A'], batch['B']
    real_img_B = Variable(batch["B"].type(Tensor), requires_grad=False)
    generated_image = generator(real_img_B)

    input_image = np.transpose(real_img_B.data.cpu().numpy(), (0, 2, 3, 1))
    generated_image = np.transpose(generated_image.data.cpu().numpy(), (0, 2, 3, 1))
    original_image = np.transpose(real_img_A.data.cpu().numpy(), (0, 2, 3, 1))

    input_image_list.extend(input_image)
    generated_image_list.extend(generated_image)
    original_image_list.extend(original_image)

input_image_list = np.array(input_image_list)
generated_image_list = np.array(generated_image_list)
original_image_list = np.array(original_image_list)

## Image Decoding
input_image_list = np.uint8((0.5 + (input_image_list*0.5)) * 255)
generated_image_list = np.uint8((0.5 + (generated_image_list*0.5)) * 255)
original_image_list = np.uint8((0.5 + (original_image_list*0.5)) * 255)

## Image view
# input image
se_idx = 63
Image.fromarray(input_image_list[se_idx])

# generated image
Image.fromarray(generated_image_list[se_idx])

# original image
Image.fromarray(original_image_list[se_idx])