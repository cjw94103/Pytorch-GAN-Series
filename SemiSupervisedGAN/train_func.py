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
    FloatTensor = torch.cuda.FloatTensor
    LongTensor = torch.cuda.LongTensor

    # Loss functions
    adversarial_loss = torch.nn.BCELoss()
    auxiliary_loss = torch.nn.CrossEntropyLoss()
    adversarial_loss.cuda()
    auxiliary_loss.cuda()

    # Learning!!
    for epoch in range(start_epoch, args.epochs):
        train_t = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        
        for i, batch in train_t:
            imgs, labels = batch
            batch_size = imgs.shape[0]

            # real, fake labeling
            valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
            fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)
            fake_aux_gt = Variable(LongTensor(batch_size).fill_(args.num_classes), requires_grad=False)

            # real images and label information
            real_imgs = Variable(imgs.type(FloatTensor))
            labels = Variable(labels.type(LongTensor))

            ## Train Generator ##
            optimizer_G.zero_grad()

            # Sample noise and labels as generator input
            z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, args.latent_dim))))
    
            # generate fake images
            gen_imgs = generator(z)
    
            # calculate loss for generator
            validity, _ = discriminator(gen_imgs)
            g_loss = adversarial_loss(validity, valid)
    
            g_loss.backward()
            optimizer_G.step()

            ## Train Discriminator ##
            optimizer_D.zero_grad()

            # Loss for real images
            real_pred, real_aux = discriminator(real_imgs)
            d_real_loss = (adversarial_loss(real_pred, valid) + auxiliary_loss(real_aux, labels)) / 2
    
            # Loss for fake images
            fake_pred, fake_aux = discriminator(gen_imgs.detach())
            d_fake_loss = (adversarial_loss(fake_pred, fake) + auxiliary_loss(fake_aux, fake_aux_gt)) / 2
    
            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2
    
            # Calculate discriminator accuracy
            pred = np.concatenate([real_aux.data.cpu().numpy(), fake_aux.data.cpu().numpy()], axis=0)
            gt = np.concatenate([labels.data.cpu().numpy(), fake_aux_gt.data.cpu().numpy()], axis=0)
            d_acc = np.mean(np.argmax(pred, axis=1) == gt)
    
            d_loss.backward()
            optimizer_D.step()

            # loss recording
            D_loss_list.append(d_loss.item())
            G_loss_list.append(g_loss.item())

            # print tqdm
            print_D_loss = round(d_loss.item(), 4)
            print_G_loss = round(g_loss.item(), 4)
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
                model_dict['D'] = discriminator.state_dict()
                model_dict['G'] = generator.state_dict()
                torch.save(model_dict, per_epoch_save_path)