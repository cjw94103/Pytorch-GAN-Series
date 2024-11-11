import torch
import argparse
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

from models import Generator, Discriminator, weights_init_normal
from torch.utils.data import DataLoader

from torch.autograd import Variable

## argparse
parser = argparse.ArgumentParser()

## Prepare data
parser.add_argument("--num_classes", type=int, default=10, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, help="size of image", default=32)
parser.add_argument("--channels", type=int, help="channels of image", default=1)

## Prepare dataloader
parser.add_argument("--batch_size", type=int, help="num of batch size", default=100)
parser.add_argument("--num_workers", type=int, help="num workers of dataloader", default=0)

## model architecture
parser.add_argument("--latent_dim", type=int, help="num of latent vector dimension", default=100)

## model save
parser.add_argument("--model_save_path", type=str, help="your model save path", default="./model_result/mnist/SGAN_mnist_100epochs.pth")

args = parser.parse_args()

## load trained model
weights = torch.load(args.model_save_path)

generator = Generator(num_classes=args.num_classes, latent_dim=args.latent_dim, img_size=args.img_size, channels=args.channels)
generator.load_state_dict(weights['G'])
generator.cuda()
generator.eval()

## inference
# for Type Casting
FloatTensor = torch.cuda.FloatTensor

z = Variable(FloatTensor(np.random.normal(0, 1, (args.batch_size, args.latent_dim))), requires_grad=False)
gen_imgs = generator(z)
gen_imgs = np.transpose(gen_imgs.data.cpu().numpy(), (0, 2, 3, 1))

## Visualization
plt.figure(figsize=(10, 10))
for i in range(args.batch_size):
    plt.subplot(10, 10, i+1)
    plt.imshow(gen_imgs[i], interpolation='nearest')
    plt.axis('off')
plt.tight_layout()