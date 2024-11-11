import torch
import numpy as np
from torch.autograd import Variable
from tqdm import tqdm

def save_history(D_loss_list, G_loss_list, save_path):
    history = {}

    history['D_loss'] = D_loss_list
    history['G_loss'] = G_loss_list

    np.save(save_path, history)

def train(args, generator, discriminator, train_dataloader, optimizer_G, optimizer_D):
    start_epoch = 0
    total_iter = len(train_dataloader) * args.epochs

    D_loss_list = []
    G_loss_list = []

    # for Type Casting
    Tensor = torch.cuda.FloatTensor

    # Define loss instance
    criterion_GAN = torch.nn.MSELoss()
    criterion_pixelwise = torch.nn.L1Loss()
    criterion_GAN.cuda()
    criterion_pixelwise.cuda()

    # Learning
    for epoch in range(start_epoch, args.epochs):
        train_t = tqdm(enumerate(train_dataloader), total=len(train_dataloader))

        for i, batch in train_t:
            # train inputs
            real_A = Variable(batch["B"].type(Tensor))
            real_B = Variable(batch["A"].type(Tensor))

            # real, fake labeling
            patch = (1, args.img_height // 2 ** 4, args.img_width // 2 ** 4)
            valid = Variable(Tensor(np.ones((real_A.size(0), *patch))), requires_grad=False)
            fake = Variable(Tensor(np.zeros((real_A.size(0), *patch))), requires_grad=False)

            ## Train Generator ##
            optimizer_G.zero_grad()

            # GAN loss
            fake_B = generator(real_A)
            pred_fake = discriminator(fake_B, real_A)
            loss_GAN = criterion_GAN(pred_fake, valid)
            # Pixel-wise loss
            loss_pixel = criterion_pixelwise(fake_B, real_B)

            # Total G loss
            loss_G = loss_GAN + args.lambda_pixel * loss_pixel
            loss_G.backward()
            optimizer_G.step()

            ## Train Discriminator ##
            optimizer_D.zero_grad()

            # real loss
            pred_real = discriminator(real_B, real_A)
            loss_real = criterion_GAN(pred_real, valid)

            # fake loss
            pred_fake = discriminator(fake_B.detach(), real_A)
            loss_fake = criterion_GAN(pred_fake, fake)

            # total loss
            loss_D = 0.5 * (loss_real + loss_fake)

            loss_D.backward()
            optimizer_D.step()

            # loss recording
            D_loss_list.append(loss_D.item())
            G_loss_list.append(loss_G.item())

            # print tqdm
            print_D_loss = round(loss_D.item(), 4)
            print_G_loss = round(loss_G.item(), 4)
            train_t.set_postfix_str("Discriminator loss : {}, Generator loss : {}".format(print_D_loss, print_G_loss))

        # save loss dict
        save_history(D_loss_list, G_loss_list, args.model_save_path.replace('.pth', '.npy'))

        # save model per epochs
        if args.save_per_epochs is not None:
            if (epoch+1) % args.save_per_epochs == 0:
                print("save per epochs {}".format(str(epoch+1)))
                per_epoch_save_path = args.model_save_path.replace(".pth", '_' + str(epoch+1) + 'epochs.pth')
                print(per_epoch_save_path)
                
                model_dict = {}
                model_dict['G'] = generator.state_dict()
                model_dict['D'] = discriminator.state_dict()
                torch.save(model_dict, per_epoch_save_path)