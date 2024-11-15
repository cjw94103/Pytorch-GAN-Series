import torch
import numpy as np
from torch.autograd import Variable
from tqdm import tqdm
from models import LambdaLR

def save_history(D_loss_list, G_loss_list, save_path):
    history = {}

    history['D_loss'] = D_loss_list
    history['G_loss'] = G_loss_list

    np.save(save_path, history)

def compute_kl(mu):
    mu_2 = torch.pow(mu, 2)
    loss = torch.mean(mu_2)
    return loss

def train(args, E1, E2, G1, G2, D1, D2, train_dataloader, optimizer_G, optimizer_D1, optimizer_D2):
    start_epoch = 0
    total_iter = len(train_dataloader) * args.epochs

    D_loss_list = []
    G_loss_list = []

    # get optimizer scheduler
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(args.epochs, 0, args.lr_decay_epoch).step)
    lr_scheduler_D1 = torch.optim.lr_scheduler.LambdaLR(optimizer_D1, lr_lambda=LambdaLR(args.epochs, 0, args.lr_decay_epoch).step)
    lr_scheduler_D2 = torch.optim.lr_scheduler.LambdaLR(optimizer_D2, lr_lambda=LambdaLR(args.epochs, 0, args.lr_decay_epoch).step)

    # for type casting
    Tensor = torch.cuda.FloatTensor

    # Define loss instance
    criterion_GAN = torch.nn.MSELoss()
    criterion_pixel = torch.nn.L1Loss()
    criterion_GAN.cuda()
    criterion_pixel.cuda()

    # Learning
    for epoch in range(start_epoch, args.epochs):
        train_t = tqdm(enumerate(train_dataloader), total=len(train_dataloader))

        for i, batch in train_t:
            X1 = Variable(batch["A"].type(Tensor))
            X2 = Variable(batch["B"].type(Tensor))

            # real, fake labeling
            valid = Variable(Tensor(np.ones((X1.size(0), *D1.output_shape))), requires_grad=False)
            fake = Variable(Tensor(np.zeros((X1.size(0), *D1.output_shape))), requires_grad=False)

            ### Train Generator ###
            optimizer_G.zero_grad()

            # shared latent representation
            mu1, Z1 = E1(X1)
            mu2, Z2 = E2(X2)

            # Reconstruction
            recon_X1 = G1(Z1)
            recon_X2 = G2(Z2)

            # Translate images
            fake_X1 = G1(Z2)
            fake_X2 = G2(Z1)

            # Cycle Consistency
            mu1_, Z1_ = E1(fake_X1)
            mu2_, Z2_ = E2(fake_X2)
            cycle_X1 = G1(Z2_)
            cycle_X2 = G2(Z1_)

            # Calculate Generator Loss
            loss_GAN_1 = args.lambda_gan * criterion_GAN(D1(fake_X1), valid)
            loss_GAN_2 = args.lambda_gan * criterion_GAN(D2(fake_X2), valid)
            loss_KL_1 = args.lambda_kl_enc * compute_kl(mu1)
            loss_KL_2 = args.lambda_kl_enc * compute_kl(mu2)
            loss_ID_1 = args.lambda_id * criterion_pixel(recon_X1, X1)
            loss_ID_2 = args.lambda_id * criterion_pixel(recon_X2, X2)
            loss_KL_1_ = args.lambda_kl_trans_enc * compute_kl(mu1_)
            loss_KL_2_ = args.lambda_kl_trans_enc * compute_kl(mu2_)
            loss_cyc_1 = args.lambda_cyc * criterion_pixel(cycle_X1, X1)
            loss_cyc_2 = args.lambda_cyc * criterion_pixel(cycle_X2, X2)

            loss_G = (
                loss_KL_1
                + loss_KL_2
                + loss_ID_1
                + loss_ID_2
                + loss_GAN_1
                + loss_GAN_2
                + loss_KL_1_
                + loss_KL_2_
                + loss_cyc_1
                + loss_cyc_2
            )
            
            loss_G.backward()
            optimizer_G.step()

            ## Train Discriminator ##
            optimizer_D1.zero_grad()

            loss_D1 = criterion_GAN(D1(X1), valid) + criterion_GAN(D1(fake_X1.detach()), fake)
            loss_D1.backward()
            optimizer_D1.step()

            optimizer_D2.zero_grad()

            loss_D2 = criterion_GAN(D2(X2), valid) + criterion_GAN(D2(fake_X2.detach()), fake)

            loss_D2.backward()
            optimizer_D2.step()

            # for loss recording
            loss_D = (loss_D1 + loss_D2) / 2

            # loss recording
            D_loss_list.append(loss_D.item())
            G_loss_list.append(loss_G.item())

            # print tqdm
            print_D_loss = round(loss_D.item(), 4)
            print_G_loss = round(loss_G.item(), 4)
            train_t.set_postfix_str("Discriminator loss : {}, Generator loss : {}".format(print_D_loss, print_G_loss))

        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D1.step()
        lr_scheduler_D2.step()

        # save loss dict
        save_history(D_loss_list, G_loss_list, args.model_save_path.replace('.pth', '.npy'))

        # save model per epochs
        if args.save_per_epochs is not None:
            if (epoch+1) % args.save_per_epochs == 0:
                print("save per epochs {}".format(str(epoch+1)))
                per_epoch_save_path = args.model_save_path.replace(".pth", '_' + str(epoch+1) + 'epochs.pth')
                print(per_epoch_save_path)
                
                model_dict = {}
                model_dict['E1'] = E1.state_dict()
                model_dict['E2'] = E2.state_dict()
                model_dict['D1'] = D1.state_dict()
                model_dict['D2'] = D2.state_dict()
                model_dict['G1'] = G1.state_dict()
                model_dict['G2'] = G2.state_dict()
                torch.save(model_dict, per_epoch_save_path)