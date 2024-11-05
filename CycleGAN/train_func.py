from utils import LambdaLR, ReplayBuffer
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable
import torch

def save_history(D_loss_list, G_loss_list, save_path):
    history = {}

    history['D_loss'] = D_loss_list
    history['G_loss'] = G_loss_list

    np.save(save_path, history)
    
def train(args, G_AB, G_BA, D_A, D_B, train_dataloader, optimizer_G, optimizer_D_A, optimizer_D_B, ):
    start_epoch = 0
    total_iter = len(train_dataloader) * args.epochs

    D_loss_list = []
    G_loss_list = []

    # get optimizer scheduler
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(args.epochs, 0, args.lr_decay_epoch).step)
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(args.epochs, 0, args.lr_decay_epoch).step)
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(args.epochs, 0, args.lr_decay_epoch).step)

    # for Type Casting
    Tensor = torch.cuda.FloatTensor

    # Buffers for Fake Example
    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    # Define Loss instance
    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()

    # Define Discriminator output shape
    output_shape = (1, args.img_height // 2 ** 4, args.img_width // 2 ** 4)
    
    # Learning!!
    for epoch in range(start_epoch, args.epochs):
        train_t = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        
        for i, batch in train_t:       
            # train input
            real_A = Variable(batch["A"].type(Tensor))
            real_B = Variable(batch["B"].type(Tensor))
    
            # real, fake labeling
            valid = Variable(Tensor(np.ones((real_A.size(0), *output_shape))), requires_grad=False)
            fake = Variable(Tensor(np.zeros((real_A.size(0), *output_shape))), requires_grad=False)
    
            ## Train Generator ##
            G_AB.train()
            G_BA.train()
            optimizer_G.zero_grad()
    
            # Identity loss
            loss_id_A = criterion_identity(G_BA(real_A), real_A)
            loss_id_B = criterion_identity(G_AB(real_B), real_B)
    
            loss_identity = (loss_id_A + loss_id_B) / 2
    
            # Adversarial Loss
            fake_B = G_AB(real_A)
            loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)
            fake_A = G_BA(real_B)
            loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)
    
            loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2
    
            # Cycle Consistency Loss
            recov_A = G_BA(fake_B)
            loss_cycle_A = criterion_cycle(recov_A, real_A)
            recov_B = G_AB(fake_A)
            loss_cycle_B = criterion_cycle(recov_B, real_B)
    
            loss_cycle = (loss_cycle_A + loss_cycle_B) / 2
    
            # Total Generator Loss
            loss_G = loss_GAN + args.lambda_cyc * loss_cycle + args.lambda_id * loss_identity

            loss_G.backward()
            optimizer_G.step()

            ## Train Discriminator A##
            optimizer_D_A.zero_grad()

            # Real and Fake Loss
            loss_real = criterion_GAN(D_A(real_A), valid)
            fake_A_ = fake_A_buffer.push_and_pop(fake_A)
            loss_fake = criterion_GAN(D_A(fake_A_.detach()), fake)
            
            loss_D_A = (loss_real + loss_fake) / 2
            
            loss_D_A.backward()
            optimizer_D_A.step()

            ## Train Discriminator B##
            optimizer_D_B.zero_grad()
            
            # Real and Fake Loss
            loss_real = criterion_GAN(D_B(real_B), valid)
            fake_B_ = fake_B_buffer.push_and_pop(fake_B)
            loss_fake = criterion_GAN(D_B(fake_B_.detach()), fake)
            
            loss_D_B = (loss_real + loss_fake) / 2

            loss_D_B.backward()
            optimizer_D_B.step()

            # Total Discriminator Loss
            loss_D = (loss_D_A + loss_D_B) / 2

            # loss recording
            D_loss_list.append(loss_D.item())
            G_loss_list.append(loss_G.item())

            # print tqdm
            print_D_loss = round(loss_D.item(), 4)
            print_G_loss = round(loss_G.item(), 4)
            train_t.set_postfix_str("Discriminator loss : {}, Generator loss : {}".format(print_D_loss, print_G_loss))

        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()

        # save loss dict
        save_history(D_loss_list, G_loss_list, args.model_save_path.replace('.pth', '.npy'))

        # save model per epochs
        if args.save_per_epochs is not None:
            if (epoch+1) % args.save_per_epochs == 0:
                print("save per epochs {}".format(str(epoch+1)))
                per_epoch_save_path = args.model_save_path.replace(".pth", '_' + str(epoch+1) + 'epochs.pth')
                print(per_epoch_save_path)

                if args.multi_gpu_flag == True:
                    model_dict = {}
                    model_dict['D_A'] = D_A.module.state_dict()
                    model_dict['D_B'] = D_B.module.state_dict()
                    model_dict['G_AB'] = G_AB.module.state_dict()
                    model_dict['G_BA'] = G_BA.module.state_dict()
                    torch.save(model_dict, per_epoch_save_path)
                else:
                    model_dict = {}
                    model_dict['D_A'] = D_A.state_dict()
                    model_dict['D_B'] = D_B.state_dict()
                    model_dict['G_AB'] = G_AB.state_dict()
                    model_dict['G_BA'] = G_BA.state_dict()
                    torch.save(model_dict, per_epoch_save_path)