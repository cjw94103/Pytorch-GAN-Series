import torch
import argparse
import numpy as np

from models import Generator, Discriminator
from torch.autograd import Variable
import matplotlib.pyplot as plt

## argparse
## argparse
parser = argparse.ArgumentParser()

## Prepare data
parser.add_argument("--n_classes", type=int, default=10, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, help="size of image", default=32)
parser.add_argument("--channels", type=int, help="channels of image", default=1)

## Prepare dataloader
parser.add_argument("--batch_size", type=int, help="num of batch size", default=9)
parser.add_argument("--num_workers", type=int, help="num workers of dataloader", default=0)

## model architecture
parser.add_argument("--latent_dim", type=int, help="num of latent vector dimension", default=100)

## Learning parameters

## model save
parser.add_argument("--model_save_path", type=str, help="your model save path", default="./model_result/mnist/CGAN_mnist_200epochs.pth")

args = parser.parse_args(args=[])

img_shape = (args.channels, args.img_size, args.img_size)

## load trained model
weights = torch.load(args.model_save_path)

generator = Generator(n_classes=args.n_classes, latent_dim=args.latent_dim, img_shape=img_shape)
generator.load_state_dict(weights['G'])
generator.cuda()
generator.eval()

## inference
# for Type Casting
FloatTensor = torch.cuda.FloatTensor
LongTensor = torch.cuda.LongTensor

target_label = 9
target_label = [target_label for i in range(args.batch_size)]

z = Variable(FloatTensor(np.random.normal(0, 1, (args.batch_size, args.latent_dim))), requires_grad=False)
gen_labels = Variable(LongTensor(target_label), requires_grad=False)

gen_imgs = generator(z, gen_labels)
gen_imgs = np.transpose(gen_imgs.data.cpu().numpy(), (0, 2, 3, 1))

## Visualization
plt.figure(figsize=(3, 3))
for i in range(args.batch_size):
    plt.subplot(3, 3, i+1)
    plt.imshow(gen_imgs[i], interpolation='nearest')
    plt.axis('off')
plt.tight_layout()