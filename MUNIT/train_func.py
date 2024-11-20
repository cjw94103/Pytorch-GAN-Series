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

def train(args, Enc1, Enc2, Dec1, Dec2, D1, D2, train_dataloader, optimizer_G, optimizer_D1, optimizer_D2):
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
    criterion_recon = torch.nn.L1Loss()
    criterion_recon.cuda()

    # real, fake labeling
    valid = 1
    fake = 0

    # Learning
    for epoch in range(start_epoch, args.epochs):
        train_t = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        
        for i, batch in train_t:
            X1 = Variable(batch["A"].type(Tensor))
            X2 = Variable(batch["B"].type(Tensor))

            # get style codes
            style_1 = Variable(torch.randn(X1.size(0), args.style_dim, 1, 1).type(Tensor))
            style_2 = Variable(torch.randn(X1.size(0), args.style_dim, 1, 1).type(Tensor))

            ## Train Encoder, Generator ##
            optimizer_G.zero_grad()
    
            # Get shared latent representation
            c_code_1, s_code_1 = Enc1(X1)
            c_code_2, s_code_2 = Enc2(X2)
    
            # Reconstruct images
            X11 = Dec1(c_code_1, s_code_1)
            X22 = Dec2(c_code_2, s_code_2)
    
            # Translate images
            X21 = Dec1(c_code_2, style_1)
            X12 = Dec2(c_code_1, style_2)
    
            # Cycle translation
            c_code_21, s_code_21 = Enc1(X21)
            c_code_12, s_code_12 = Enc2(X12)
            X121 = Dec1(c_code_12, s_code_1) if args.lambda_cyc > 0 else 0
            X212 = Dec2(c_code_21, s_code_2) if args.lambda_cyc > 0 else 0
    
            # Losses
            loss_GAN_1 = args.lambda_gan * D1.compute_loss(X21, valid)
            loss_GAN_2 = args.lambda_gan * D2.compute_loss(X12, valid)
            loss_ID_1 = args.lambda_id * criterion_recon(X11, X1)
            loss_ID_2 = args.lambda_id * criterion_recon(X22, X2)
            loss_s_1 = args.lambda_style * criterion_recon(s_code_21, style_1)
            loss_s_2 = args.lambda_style * criterion_recon(s_code_12, style_2)
            loss_c_1 = args.lambda_cont * criterion_recon(c_code_12, c_code_1.detach())
            loss_c_2 = args.lambda_cont * criterion_recon(c_code_21, c_code_2.detach())
            loss_cyc_1 = args.lambda_cyc * criterion_recon(X121, X1) if args.lambda_cyc > 0 else 0
            loss_cyc_2 = args.lambda_cyc * criterion_recon(X212, X2) if args.lambda_cyc > 0 else 0
    
            # Total loss
            loss_G = (
                loss_GAN_1
                + loss_GAN_2
                + loss_ID_1
                + loss_ID_2
                + loss_s_1
                + loss_s_2
                + loss_c_1
                + loss_c_2
                + loss_cyc_1
                + loss_cyc_2
            )
    
            loss_G.backward()
            optimizer_G.step()

            ## Train D1 ###
            optimizer_D1.zero_grad()

            loss_D1 = D1.compute_loss(X1, valid) + D1.compute_loss(X21.detach(), fake)
    
            loss_D1.backward()
            optimizer_D1.step()

            ## Train D2 ##
            optimizer_D2.zero_grad()

            loss_D2 = D2.compute_loss(X2, valid) + D2.compute_loss(X12.detach(), fake)
    
            loss_D2.backward()
            optimizer_D2.step()

            # Total discriminator loss
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
                model_dict['Enc1'] = Enc1.state_dict()
                model_dict['Enc2'] = Enc2.state_dict()
                model_dict['Dec1'] = Dec1.state_dict()
                model_dict['Dec2'] = Dec2.state_dict()
                model_dict['D1'] = D1.state_dict()
                model_dict['D2'] = D2.state_dict()
                
                torch.save(model_dict, per_epoch_save_path)