import torch
from models import Generator, Discriminator

import numpy as np
import argparse
from tqdm import tqdm
from train_func import train
from torch.autograd import Variable
import matplotlib.pyplot as plt

## draw function
def plot_generated_images(generator):
    Tensor = torch.cuda.FloatTensor
    noise = Variable(Tensor(np.random.normal(0, 1, (args.batch_size, args.latent_dim))))
    generated_images = generator(noise)
    generated_images = np.transpose(generated_images.data.cpu().numpy(), (0, 2, 3, 1))
    plt.figure(figsize=(10, 10))
    for i in range(generated_images.shape[0]):
        plt.subplot(10, 10, i+1)
        plt.imshow(generated_images[i], interpolation='nearest')
        plt.axis('off')

    plt.tight_layout()

## argparse
parser = argparse.ArgumentParser()

## Prepare data
parser.add_argument("--img_size", type=int, help="size of  image", default=28)
parser.add_argument("--channels", type=int, help="channels of image", default=1)

## Prepare dataloader
parser.add_argument("--batch_size", type=int, help="num of batch size", default=100)

## model architecture
parser.add_argument("--latent_dim", type=int, help="num of latent vector dimension", default=100)

## model save
parser.add_argument("--model_save_path", type=str, help="your model save path", default="./model_result/mnist/NaiveGAN_mnist_50epochs.pth")

args = parser.parse_args([])

img_shape = (args.channels, args.img_size, args.img_size)

## load trained model
weights = torch.load(args.model_save_path)
generator = Generator(latent_dim=args.latent_dim, img_shape=img_shape)
generator.load_state_dict(weights['G'])

# model instance to CUDA GPU
generator.cuda()
generator.eval()

## inference
plot_generated_images(generator)