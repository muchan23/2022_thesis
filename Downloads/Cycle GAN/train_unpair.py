import torch
from dataset_unpair import OpticalSARDataset
import sys
from utils import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator_spectrumnorm import Discriminator #check
from generator_model import Generator #check
from generator_ushaped import Generator #check
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms as T

def train_fn(disc_O, disc_S, gen_S, gen_O, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler):
    O_reals = 0
    O_fakes = 0
    #loop = tqdm(loader, leave=True)
    epoch = config.NUM_EPOCHS
    G_loss_iter = np.empty(0)
    D_loss_iter = np.empty(0)
    G_loss_epoch = np.empty(0)
    D_loss_epoch = np.empty(0)

    for epoch in range(config.NUM_EPOCHS):
        loop = tqdm(loader, leave=True)
        for idx, (sar, optical) in enumerate(loop):
            sar = sar.to(config.DEVICE)
            optical = optical.to(config.DEVICE)

        # Train Discriminators O and S
            with torch.cuda.amp.autocast():
                fake_optical = gen_O(sar)
                D_O_real = disc_O(optical)
                D_O_fake = disc_O(fake_optical.detach())
                O_reals += D_O_real.mean().item()
                O_fakes += D_O_fake.mean().item()
                D_O_real_loss = mse(D_O_real, torch.ones_like(D_O_real))  #check
                D_O_fake_loss = mse(D_O_fake, torch.zeros_like(D_O_fake))  #check
                #D_O_real_loss = mse(D_O_real, torch.full_like(D_O_real,0.99)) #label smooth
                #D_O_fake_loss = mse(D_O_fake, torch.full_like(D_O_fake,0.01)) #label smooth
                D_O_loss = D_O_real_loss + D_O_fake_loss

                fake_sar = gen_S(optical)
                D_S_real = disc_S(sar)
                D_S_fake = disc_S(fake_sar.detach())
                D_S_real_loss = mse(D_S_real, torch.ones_like(D_S_real)) #check
                D_S_fake_loss = mse(D_S_fake, torch.zeros_like(D_S_fake)) #check
                #D_S_real_loss = mse(D_S_real, torch.full_like(D_S_real,0.99)) #label smooth
                #D_S_fake_loss = mse(D_S_fake, torch.full_like(D_S_fake,0.01)) #label smooth
                D_S_loss = D_S_real_loss + D_S_fake_loss

                # put it togethor
                D_loss = (D_O_loss + D_S_loss)/2

            opt_disc.zero_grad()
            d_scaler.scale(D_loss).backward()
            d_scaler.step(opt_disc)
            d_scaler.update()

            # Train Generators O and S
            with torch.cuda.amp.autocast():
                # adversarial loss for both generators
                D_O_fake = disc_O(fake_optical)
                D_S_fake = disc_S(fake_sar)
                loss_G_O = mse(D_O_fake, torch.ones_like(D_O_fake)) #check
                loss_G_S = mse(D_S_fake, torch.ones_like(D_S_fake))  #check
                #loss_G_O = mse(D_O_fake, torch.full_like(D_O_fake,0.99)) #label smooth
                #loss_G_S = mse(D_S_fake, torch.full_like(D_S_fake,0.01)) #label smooth
                """
                Adv_loss_sar = l1(sar, fake_sar)
                Adv_loss_optical = l1(optical, fake_optical)
                """

                
                if epoch==0 and idx >120 and idx % 4 ==0:
                    Adv_loss_sar = l1(sar, fake_sar)
                    Adv_loss_optical = l1(optical, fake_optical)
                elif epoch==0 and idx >120 and idx % 4 ==1:
                    Adv_loss_sar = l1(sar, fake_sar)
                    Adv_loss_optical = l1(optical, fake_optical)
                elif epoch==0 and idx >120 and idx % 4 ==2:
                    Adv_loss_sar = l1(sar, fake_sar)
                    Adv_loss_optical = l1(optical, fake_optical)
                elif epoch==0 and idx >120 and idx % 4 ==3:
                    Adv_loss_sar = 0
                    Adv_loss_optical = 0
                elif epoch==0 and idx <= 120:
                    Adv_loss_sar = 0
                    Adv_loss_optical = 0
                elif epoch > 0 and idx % 2 ==0:
                    Adv_loss_sar = l1(sar, fake_sar)
                    Adv_loss_optical = l1(optical, fake_optical)
                elif epoch > 0 and idx % 2 ==1:
                    Adv_loss_sar = 0
                    Adv_loss_optical = 0

                # cycle loss
                cycle_sar = gen_S(fake_optical)
                cycle_optical = gen_O(fake_sar)
                cycle_sar_loss = l1(sar, cycle_sar)
                cycle_optical_loss = l1(optical, cycle_optical)

                # cycle aberration loss
                transform = T.GaussianBlur(kernel_size=(7, 13), sigma=(0.1, 0.2))
                sar_blur = transform(sar)
                optical_blur = transform(optical)
                cycle_sar_blur = transform(cycle_sar)
                cycle_optical_blur = transform(cycle_optical)
                cycle_sar_blur_loss = mse(sar_blur, cycle_sar_blur)
                cycle_optical_blur_loss = mse(optical_blur, cycle_optical_blur)

                # identity loss (remove these for efficiency if you set lambda_identity=0)
                identity_sar = gen_S(sar)
                identity_optical = gen_O(optical)
                identity_sar_loss = l1(sar, identity_sar)
                identity_optical_loss = l1(optical, identity_optical)

                # cycle aberration loss
                transform = T.GaussianBlur(kernel_size=(7, 13), sigma=(0.1, 0.2))
                identity_sar_blur = transform(identity_sar)
                identity_optical_blur = transform(identity_optical)
                identity_sar_blur_loss = mse(sar_blur, identity_sar_blur)
                identity_optical_blur_loss = mse(optical_blur, identity_optical_blur)

                # add all togethor
                G_loss = (
                    loss_G_O * config.LAMBDA_LOSS
                    + loss_G_S * config.LAMBDA_LOSS
                    + Adv_loss_sar * config.LAMBDA_ADV
                    + Adv_loss_optical * config.LAMBDA_ADV
                    + cycle_sar_loss * config.LAMBDA_CYCLE
                    + cycle_optical_loss * config.LAMBDA_CYCLE
                    + cycle_sar_blur_loss * config.LAMBDA_BLUR
                    + cycle_optical_blur_loss * config.LAMBDA_BLUR
                    + identity_optical_loss * config.LAMBDA_IDENTITY
                    + identity_sar_loss * config.LAMBDA_IDENTITY
                    + identity_sar_blur_loss * config.LAMBDA_BLUR
                    + identity_optical_blur_loss * config.LAMBDA_BLUR
                )

            opt_gen.zero_grad()
            g_scaler.scale(G_loss).backward()
            g_scaler.step(opt_gen)
            g_scaler.update()

            if idx % 100 == 0:
                save_image(fake_optical*0.5+0.5, f"/home/yuki_murakami/ドキュメント/Cycle GAN/data_research/pair,lr=5e-5,L=1,A=0.5,C=2,I=0.5,SN_D,SL/fakeoptical/optical_{idx}.jpeg") #check
                save_image(fake_sar*0.5+0.5, f"/home/yuki_murakami/ドキュメント/Cycle GAN/data_research/pair,lr=5e-5,L=1,A=0.5,C=2,I=0.5,SN_D,SL/fakesar/sar_{idx}.jpeg") #chec

            G_loss1 = G_loss.detach().clone().to('cpu').numpy()
            D_loss1 = D_loss.detach().clone().to('cpu').numpy()
            #Gen_loss =  Adv_loss_sar.detach().clone().to('cpu').numpy() + Adv_loss_optical.detach().clone().to('cpu').numpy()
            #Gen_loss_sar = Adv_loss_sar.detach().clone().to('cpu').numpy()
            #Gen_loss_optical = Adv_loss_optical.detach().clone().to('cpu').numpy()
            Gen_Cycle = (cycle_sar_loss * config.LAMBDA_CYCLE).detach().clone().to('cpu').numpy() + (cycle_optical_loss * config.LAMBDA_CYCLE).detach().clone().to('cpu').numpy() 
            Gen_Iden = (identity_optical_loss * config.LAMBDA_IDENTITY).detach().clone().to('cpu').numpy() + (identity_sar_loss * config.LAMBDA_IDENTITY).detach().clone().to('cpu').numpy() 

            #loop.set_postfix(O_real=O_reals/(idx+1), O_fake=O_fakes/(idx+1), G_loss = G_loss.detach().clone().to('cpu').numpy(), D_loss = D_loss.detach().clone().to('cpu').numpy())
            loop.set_postfix(G_loss = G_loss.detach().clone().to('cpu').numpy(), D_loss = D_loss.detach().clone().to('cpu').numpy())
            #loop.set_postfix(idx = idx+1)

            #25は手動で変える必要がある    
            if idx % 400 == 0:
                writer = SummaryWriter(log_dir="./ドキュメント/Graph/pair,second,lr=5e-5,L=1,A=0.5,C=2,I=0.5,SN_D,SL(0.95)") #check
                writer.add_scalar("G_loss", G_loss1.item(),epoch*10000 + idx)
                writer.add_scalar("D_loss", D_loss1.item(),epoch*10000 + idx)
                #writer.add_scalar("Gen_loss", Gen_loss.item(),epoch*10000 + idx)
                #writer.add_scalar("Gen_loss_sar", Gen_loss_sar.item(),epoch*10000 + idx)
                #writer.add_scalar("Gen_loss_optical", Gen_loss_optical.item(),epoch*10000 + idx)
                writer.add_scalar("Gen_Cycle", Gen_Cycle.item(),epoch*10000 + idx)
                writer.add_scalar("Gen_Iden", Gen_Iden.item(),epoch*10000 + idx)

                grid_optical = torchvision.utils.make_grid(fake_optical*0.5+0.5)
                grid_sar = torchvision.utils.make_grid(fake_sar*0.5+0.5)
                writer1 = SummaryWriter("./ドキュメント/Image/pair,second,lr=5e-5,L=1,A=0.5,C=2,I=0.5,SN_D,SL(0.95)") #check
                writer1.add_image("fake_optical",grid_optical,epoch*10000 + idx)
                writer1.add_image("fake_sar",grid_sar,epoch*10000 + idx)

            #writer = SummaryWriter(log_dir="./1")
            #writer.add_scalar("G_loss", G_loss1.item(),epoch)
            #writer.add_scalar("D_loss", D_loss1.item(),epoch)
            G_loss_iter = np.append(G_loss_iter, G_loss1)
            D_loss_iter = np.append(D_loss_iter, D_loss1)

        G_loss_epoch = np.append(G_loss_epoch, G_loss1)
        D_loss_epoch = np.append(D_loss_epoch, D_loss1)
   
        if config.SAVE_MODEL:
            save_checkpoint(gen_O, opt_gen, filename=f"/home/yuki_murakami/ドキュメント/Cycle GAN/data_research/pair,lr=5e-5,L=1,A=0.5,C=2,I=0.5,SN_D,SL/checkpoint/epoch{epoch+1}/geno.pth.tar")
            save_checkpoint(gen_S, opt_gen, filename=f"/home/yuki_murakami/ドキュメント/Cycle GAN/data_research/pair,lr=5e-5,L=1,A=0.5,C=2,I=0.5,SN_D,SL/checkpoint/epoch{epoch+1}/gens.pth.tar")
            save_checkpoint(disc_O, opt_disc, filename=f"/home/yuki_murakami/ドキュメント/Cycle GAN/data_research/pair,lr=5e-5,L=1,A=0.5,C=2,I=0.5,SN_D,SL/checkpoint/epoch{epoch+1}/critico.pth.tar")
            save_checkpoint(disc_S, opt_disc, filename=f"/home/yuki_murakami/ドキュメント/Cycle GAN/data_research/pair,lr=5e-5,L=1,A=0.5,C=2,I=0.5,SN_D,SL/checkpoint/epoch{epoch+1}/critics.pth.tar")

        print(epoch)

    np.save("/home/yuki_murakami/ドキュメント/Cycle GAN/data_research/pair,lr=5e-5,L=1,A=0.5,C=2,I=0.5,SN_D,SL/numpy/G_loss_iter.npy",G_loss_iter)
    np.save("/home/yuki_murakami/ドキュメント/Cycle GAN/data_research/pair,lr=5e-5,L=1,A=0.5,C=2,I=0.5,SN_D,SL/numpy/D_loss_iter.npy",D_loss_iter)
    np.save("/home/yuki_murakami/ドキュメント/Cycle GAN/data_research/pair,lr=5e-5,L=1,A=0.5,C=2,I=0.5,SN_D,SL/numpy/G_loss_epoch.npy",G_loss_epoch)
    np.save("/home/yuki_murakami/ドキュメント/Cycle GAN/data_research/pair,lr=5e-5,L=1,A=0.5,C=2,I=0.5,SN_D,SL/numpy/D_loss_epoch.npy",D_loss_epoch)

def main():
    disc_O = Discriminator(in_channels=3).to(config.DEVICE)
    disc_S = Discriminator(in_channels=3).to(config.DEVICE)
    #gen_S = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    #gen_O = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    gen_S = Generator(in_channels=3, out_channels=3).to(config.DEVICE)  #U-shaped
    gen_O = Generator(in_channels=3, out_channels=3).to(config.DEVICE) #U-shaped
    opt_disc = optim.Adam(
        list(disc_O.parameters()) + list(disc_S.parameters()),
        lr=config.LEARNING_RATE_DIS,
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(gen_S.parameters()) + list(gen_O.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_gen_O, gen_O, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_gen_S, gen_S, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_O, disc_O, opt_disc, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_S, disc_S, opt_disc, config.LEARNING_RATE,
        )

    dataset = OpticalSARDataset(
        root_optical=config.TRAIN_DIR+"/optical_10000", root_sar=config.TRAIN_DIR+"/sar_10000", transform=config.transforms #check
    )
    val_dataset = OpticalSARDataset(
       root_optical=config.VAL_DIR+"/optical_3000", root_sar=config.VAL_DIR+"/sar_3000", transform=config.transforms
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()


    train_fn(disc_O, disc_S, gen_S, gen_O, loader, opt_disc, opt_gen, L1, mse, d_scaler, g_scaler)

if __name__ == "__main__":
    main()
