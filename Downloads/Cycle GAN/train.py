import torch
from dataset import OpticalSARDataset
import sys
from utils import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator_model import Discriminator #check
from generator_model import Generator #check
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torchvision
import matplotlib.pyplot as plt

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
                #D_O_real_loss = mse(D_O_real, torch.full_like(D_O_real,0.95)) #label smooth
                #D_O_fake_loss = mse(D_O_fake, torch.full_like(D_O_fake,0.05)) #label smooth
                D_O_loss = D_O_real_loss + D_O_fake_loss

                fake_sar = gen_S(optical)
                D_S_real = disc_S(sar)
                D_S_fake = disc_S(fake_sar.detach())
                D_S_real_loss = mse(D_S_real, torch.ones_like(D_S_real)) #check
                D_S_fake_loss = mse(D_S_fake, torch.zeros_like(D_S_fake)) #check
                #D_S_real_loss = mse(D_S_real, torch.full_like(D_S_real,0.95)) #label smooth
                #D_S_fake_loss = mse(D_S_fake, torch.full_like(D_S_fake,0.05)) #label smooth
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
                #loss_G_O = mse(D_O_fake, torch.full_like(D_O_fake,0.95)) #label smooth
                #loss_G_S = mse(D_S_fake, torch.full_like(D_S_fake,0.95)) #label smooth

                # cycle loss
                cycle_sar = gen_S(fake_optical)
                cycle_optical = gen_O(fake_sar)
                cycle_sar_loss = l1(sar, cycle_sar)
                cycle_optical_loss = l1(optical, cycle_optical)

                # identity loss (remove these for efficiency if you set lambda_identity=0)
                identity_sar = gen_S(sar)
                identity_optical = gen_O(optical)
                identity_sar_loss = l1(sar, identity_sar)
                identity_optical_loss = l1(optical, identity_optical)

                # add all togethor
                G_loss = (
                    loss_G_S
                    + loss_G_O
                    + cycle_sar_loss * config.LAMBDA_CYCLE
                    + cycle_optical_loss * config.LAMBDA_CYCLE
                    + identity_optical_loss * config.LAMBDA_IDENTITY
                    + identity_sar_loss * config.LAMBDA_IDENTITY
                )

            opt_gen.zero_grad()
            g_scaler.scale(G_loss).backward()
            g_scaler.step(opt_gen)
            g_scaler.update()

            if idx % 200 == 0:
                save_image(fake_optical*0.5+0.5, f"/home/yuki_murakami/ドキュメント/Cycle GAN/data/image_train_firstset/fake_optical/optical_{idx}.jpeg") #check
                save_image(fake_sar*0.5+0.5, f"/home/yuki_murakami/ドキュメント/Cycle GAN/data/image_train_firstset/fake_sar/sar_{idx}.jpeg") #chec

            G_loss1 = G_loss.detach().clone().to('cpu').numpy()
            D_loss1 = D_loss.detach().clone().to('cpu').numpy()
            Gen_loss =  loss_G_S.detach().clone().to('cpu').numpy() + loss_G_O.detach().clone().to('cpu').numpy()
            Gen_loss_sar = loss_G_S.detach().clone().to('cpu').numpy()
            Gen_loss_optical = loss_G_O.detach().clone().to('cpu').numpy()
            Gen_Cycle = (cycle_sar_loss * config.LAMBDA_CYCLE).detach().clone().to('cpu').numpy() + (cycle_optical_loss * config.LAMBDA_CYCLE).detach().clone().to('cpu').numpy() 
            Gen_Iden = (identity_optical_loss * config.LAMBDA_IDENTITY).detach().clone().to('cpu').numpy() + (identity_sar_loss * config.LAMBDA_IDENTITY).detach().clone().to('cpu').numpy() 

            #loop.set_postfix(O_real=O_reals/(idx+1), O_fake=O_fakes/(idx+1), G_loss = G_loss.detach().clone().to('cpu').numpy(), D_loss = D_loss.detach().clone().to('cpu').numpy())
            loop.set_postfix(G_loss = G_loss.detach().clone().to('cpu').numpy(), D_loss = D_loss.detach().clone().to('cpu').numpy())
            #loop.set_postfix(idx = idx+1)

            #25は手動で変える必要がある    
            if idx % 200 == 0:
                writer = SummaryWriter(log_dir="./TensorBoard/Graph/lr=0.00005,d=10000,b=1,C=1,I=1") #check
                writer.add_scalar("G_loss", G_loss1.item(),epoch*10000 + idx)
                writer.add_scalar("D_loss", D_loss1.item(),epoch*10000 + idx)
                writer.add_scalar("Gen_loss", Gen_loss.item(),epoch*10000 + idx)
                writer.add_scalar("Gen_loss_sar", Gen_loss_sar.item(),epoch*10000 + idx)
                writer.add_scalar("Gen_loss_optical", Gen_loss_optical.item(),epoch*10000 + idx)
                writer.add_scalar("Gen_Cycle", Gen_Cycle.item(),epoch*10000 + idx)
                writer.add_scalar("Gen_Iden", Gen_Iden.item(),epoch*10000 + idx)

                grid_optical = torchvision.utils.make_grid(fake_optical*0.5+0.5)
                grid_sar = torchvision.utils.make_grid(fake_sar*0.5+0.5)
                writer1 = SummaryWriter("./TensorBoard/Image/lr=0.00005,d=10000,b=1,C=1,I=1")
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
            save_checkpoint(gen_O, opt_gen, filename=config.CHECKPOINT_GEN_O)
            save_checkpoint(gen_S, opt_gen, filename=config.CHECKPOINT_GEN_S)
            save_checkpoint(disc_O, opt_disc, filename=config.CHECKPOINT_CRITIC_O)
            save_checkpoint(disc_S, opt_disc, filename=config.CHECKPOINT_CRITIC_S)

        print(epoch)

    np.save("/home/yuki_murakami/ドキュメント/Cycle GAN/data/numpy_firstset/G_loss_iter",G_loss_iter)
    np.save("/home/yuki_murakami/ドキュメント/Cycle GAN/data/numpy_firstset/D_loss_iter",D_loss_iter)
    np.save("/home/yuki_murakami/ドキュメント/Cycle GAN/data/numpy_firstset/G_loss_epoch",G_loss_epoch)
    np.save("/home/yuki_murakami/ドキュメント/Cycle GAN/data/numpy_firstset/D_loss_epoch",D_loss_epoch)

def main():
    disc_O = Discriminator(in_channels=3).to(config.DEVICE)
    disc_S = Discriminator(in_channels=3).to(config.DEVICE)
    gen_S = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    gen_O = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
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
        root_optical=config.TRAIN_DIR+"/optical", root_sar=config.TRAIN_DIR+"/sar", transform=config.transforms #check
    )
    val_dataset = OpticalSARDataset(
       root_optical=config.VAL_DIR+"/optical", root_sar=config.VAL_DIR+"/sar", transform=config.transforms
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
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()


    train_fn(disc_O, disc_S, gen_S, gen_O, loader, opt_disc, opt_gen, L1, mse, d_scaler, g_scaler)

if __name__ == "__main__":
    main()
