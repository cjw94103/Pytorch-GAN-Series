import argparse
import numpy as np
import itertools

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset import ImageDataset
from PIL import Image
from models import GeneratorResNet, Discriminator, weights_init_normal
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
parser.add_argument("--n_residual_blocks", type=int, help="num residual blocks of GAN generator", default=9)

## Learning Parameter
parser.add_argument("--epochs", type=int, help="num epochs", default=200)
parser.add_argument("--lr_decay_epoch", type=int, help="epochs of learning rate decay", default=100)

## Loss Parameter
parser.add_argument("--lambda_cyc", type=float, help="loss parameter of lambda cycle", default=10.0)
parser.add_argument("--lambda_id", type=float, help="loss parameter of identity loss", default=5.0)

## model save
parser.add_argument("--model_save_path", type=str, help="your model save path", default="./model_result/Apple2Orange/Translation_Model.pth")
parser.add_argument("--save_per_epochs", type=int, help="How many epochs to save the model", default=20)
parser.add_argument("--multi_gpu_flag", type="store_false", help="Whether to use multiple GPUs, True in train_dist.py")
parser.add_argument("--port_num", type=int, help="Which port to use when learning multi-GPU", default=14000)

## Optimizer parameter
parser.add_argument("--b1", type=float, help="b1 of Adam Optimizer", default=0.5)
parser.add_argument("--b2", type=float, help="b2 of Adam Optimizer", default=0.999)
parser.add_argument("--lr", type=float, help="learning rate", default=0.0002)

args = parser.parse_args()

# load config.json
args = Args(opt.config_path)

def setup(rank, world_size):
    adress = 'tcp://127.0.0.1:' + str(args.port_num)
    torch.distributed.init_process_group('nccl', rank=rank, world_size=world_size, init_method=adress)

def cleanup():
    torch.distributed.destroy_process_group()

def main_worker(rank, world_size):
    print(f"Running basic DDP example on rank {rank}.")

    # init process group
    batch_size = int(args.batch_size / world_size)
    num_workers = int(args.num_workers / world_size)
    setup(rank, world_size)

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
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers)

    # load model
    input_shape = (args.channels, args.img_height, args.img_width)
    
    G_AB = GeneratorResNet(input_shape, args.n_residual_blocks).to(rank)
    G_AB = torch.nn.parallel.DistributedDataParallel(G_AB, device_ids=[rank])
    
    G_BA = GeneratorResNet(input_shape, args.n_residual_blocks).to(rank)
    G_BA = torch.nn.parallel.DistributedDataParallel(G_BA, device_ids=[rank])
    
    D_A = Discriminator(input_shape).to(rank)
    D_A = torch.nn.parallel.DistributedDataParallel(D_A, device_ids=[rank])
    
    D_B = Discriminator(input_shape).to(rank)
    D_B = torch.nn.parallel.DistributedDataParallel(D_B, device_ids=[rank])
    
    # normal dist. init
    G_AB.apply(weights_init_normal)
    G_BA.apply(weights_init_normal)
    D_A.apply(weights_init_normal)
    D_B.apply(weights_init_normal)

    # get optimizer
    optimizer_G = torch.optim.Adam(itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=args.lr, betas=(args.b1, args.b2))
    optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=args.lr, betas=(args.b1, args.b2))

    # train
    train(args, G_AB, G_BA, D_A, D_B, train_dataloader, optimizer_G, optimizer_D_A, optimizer_D_B)

    # clean process
    cleanup()

def main():
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(main_worker, nprocs=world_size, args=(world_size, ), join=True)
    
if __name__ == '__main__':
    main()