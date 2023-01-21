import torch
from dataset_unify import OpticalSARDataset
import sys
from utils import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config_unify
from tqdm import tqdm
from torchvision.utils import save_image
#from discriminator_model import Discriminator #check
from discriminator_spectrumnorm import Discriminator
from generator_model import Generator #check
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torchvision
import matplotlib.pyplot as plt

import albumentations as A
from albumentations.pytorch import ToTensorV2
from model_unet import UNET #later check
from utils_unet import (
    load_checkpoint_unet,
    save_checkpoint_unet,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)

def train_fn(disc_O, disc_S, gen_S, gen_O, model, loader,val_loader, opt_disc, opt_gen, opt_unet,l1, mse, BCEWithLogitsLoss, d_scaler, g_scaler,u_scaler):
    O_reals = 0
    O_fakes = 0
    #loop = tqdm(loader, leave=True)
    epoch = config_unify.NUM_EPOCHS
    G_loss_iter = np.empty(0)
    D_loss_iter = np.empty(0)
    G_loss_epoch = np.empty(0)
    D_loss_epoch = np.empty(0)

    for epoch in range(config_unify.NUM_EPOCHS):
        loop = tqdm(loader, leave=True)
        for idx, (sar, optical, mask) in enumerate(loop):
            sar = sar.to(config_unify.DEVICE)
            optical = optical.to(config_unify.DEVICE)
            mask = mask.to(config_unify.DEVICE)
            mask = mask.unsqueeze(dim=0)

        # Train Discriminators O and S
            with torch.cuda.amp.autocast():
                fake_optical = gen_O(sar)
                D_O_real = disc_O(optical)
                D_O_fake = disc_O(fake_optical.detach())
                O_reals += D_O_real.mean().item()
                O_fakes += D_O_fake.mean().item()
                #D_O_real_loss = mse(D_O_real, torch.ones_like(D_O_real))  #check
                #D_O_fake_loss = mse(D_O_fake, torch.zeros_like(D_O_fake))  #check
                D_O_real_loss = mse(D_O_real, torch.full_like(D_O_real,0.95)) #label smooth
                D_O_fake_loss = mse(D_O_fake, torch.full_like(D_O_fake,0.05)) #label smooth
                D_O_loss = D_O_real_loss + D_O_fake_loss

                fake_sar = gen_S(optical)
                D_S_real = disc_S(sar)
                D_S_fake = disc_S(fake_sar.detach())
                #D_S_real_loss = mse(D_S_real, torch.ones_like(D_S_real)) #check
                #D_S_fake_loss = mse(D_S_fake, torch.zeros_like(D_S_fake)) #check
                D_S_real_loss = mse(D_S_real, torch.full_like(D_S_real,0.95)) #label smooth
                D_S_fake_loss = mse(D_S_fake, torch.full_like(D_S_fake,0.05)) #label smooth
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
                #loss_G_O = mse(D_O_fake, torch.ones_like(D_O_fake)) #check
                #loss_G_S = mse(D_S_fake, torch.ones_like(D_S_fake))  #check
                loss_G_O = mse(D_O_fake, torch.full_like(D_O_fake,0.95)) #label smooth
                loss_G_S = mse(D_S_fake, torch.full_like(D_S_fake,0.95)) #label smooth

                #pair loss
                fake_sar = gen_S(optical)
                Adv_loss_sar = l1(sar, fake_sar)
                Adv_loss_optical = l1(optical, fake_optical)

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
                #print(identity_sar_loss.detach().clone().to('cpu').numpy())

                # Unet loss
                fake_sar = gen_S(optical)
                concat_sar = torch.cat((sar,fake_sar),1)
                #print(concat_sar.shape)
                #concat = concat_sar.to('cpu').detach().numpy().copy()
                #concat_sar = concat_sar.to(.DEVICE)
                #concat_numpy = np.array(concat)
                #print(concat_numpy.shape)
                #concat_numpy = np.squeeze(concat_numpy, axis=None)
                #concat_numpy = np.transpose(concat_numpy, (1,2,0))
                #print(concat_numpy.shape)
                predictions = model(concat_sar) #modelの定義を行う
                unet_loss = BCEWithLogitsLoss(predictions, mask) #targetsの定義を行う

                # add all togethor
                G_loss = (
                    loss_G_S
                    + loss_G_O
                    + Adv_loss_sar * config_unify.LAMBDA_ADV
                    + Adv_loss_optical * config_unify.LAMBDA_ADV
                    + cycle_sar_loss * config_unify.LAMBDA_CYCLE
                    + cycle_optical_loss * config_unify.LAMBDA_CYCLE
                    + identity_optical_loss * config_unify.LAMBDA_IDENTITY
                    + identity_sar_loss * config_unify.LAMBDA_IDENTITY
                    + unet_loss *  config_unify.LAMBDA_UNET
                )

            opt_gen.zero_grad()
            g_scaler.scale(G_loss).backward()
            g_scaler.step(opt_gen)
            g_scaler.update()

            with torch.cuda.amp.autocast():
                fake_sar = gen_S(optical)
                concat_sar = torch.cat((sar,fake_sar),1)
                predictions = model(concat_sar) #modelの定義を行う
                unet_loss = BCEWithLogitsLoss(predictions, mask) #targetsの定義を行う
            
            opt_unet.zero_grad()
            u_scaler.scale(unet_loss).backward()
            u_scaler.step(opt_unet)
            u_scaler.update()


            if idx % 100 == 0:
                save_image(fake_optical*0.5+0.5, f"/home/yuki_murakami/ドキュメント/CycleGAN_UNet/lr=5e-5,A=1,C=1.5,I=1,U=1,SN_D,SL/fake_optical/{idx}.jpeg") #check
                save_image(fake_sar*0.5+0.5, f"/home/yuki_murakami/ドキュメント/CycleGAN_UNet/lr=5e-5,A=1,C=1.5,I=1,U=1,SN_D,SL/fake_sar/{idx}.jpeg") #check
                save_image(predictions*0.5+0.5, f"/home/yuki_murakami/ドキュメント/CycleGAN_UNet/lr=5e-5,A=1,C=1.5,I=1,U=1,SN_D,SL/fake_groundtruth/{idx}.jpeg") #check

            G_loss1 = G_loss.detach().clone().to('cpu').numpy()
            D_loss1 = D_loss.detach().clone().to('cpu').numpy()
            Gen_loss =  loss_G_S.detach().clone().to('cpu').numpy() + loss_G_O.detach().clone().to('cpu').numpy()
            Gen_loss_sar = loss_G_S.detach().clone().to('cpu').numpy()
            Gen_loss_optical = loss_G_O.detach().clone().to('cpu').numpy()
            Gen_Cycle = (cycle_sar_loss * config_unify.LAMBDA_CYCLE).detach().clone().to('cpu').numpy() + (cycle_optical_loss * config_unify.LAMBDA_CYCLE).detach().clone().to('cpu').numpy() 
            Gen_Iden = (identity_optical_loss * config_unify.LAMBDA_IDENTITY).detach().clone().to('cpu').numpy() + (identity_sar_loss * config_unify.LAMBDA_IDENTITY).detach().clone().to('cpu').numpy() 
            Gen_unet = unet_loss.detach().clone().to('cpu').numpy()

            #loop.set_postfix(O_real=O_reals/(idx+1), O_fake=O_fakes/(idx+1), G_loss = G_loss.detach().clone().to('cpu').numpy(), D_loss = D_loss.detach().clone().to('cpu').numpy())
            loop.set_postfix(G_loss = G_loss.detach().clone().to('cpu').numpy(), D_loss = D_loss.detach().clone().to('cpu').numpy())
            #loop.set_postfix(idx = idx+1)

            #25は手動で変える必要がある    
            if idx % 200 == 0:
                writer = SummaryWriter(log_dir="./ドキュメント/Graph/pair,second,lr=5e-5,A=1,C=1.5,I=1,U=1,SN_D,SL") #check
                writer.add_scalar("G_loss", G_loss1.item(),epoch*10000 + idx)
                writer.add_scalar("D_loss", D_loss1.item(),epoch*10000 + idx)
                writer.add_scalar("Gen_loss", Gen_loss.item(),epoch*10000 + idx)
                writer.add_scalar("Gen_loss_sar", Gen_loss_sar.item(),epoch*10000 + idx)
                writer.add_scalar("Gen_loss_optical", Gen_loss_optical.item(),epoch*10000 + idx)
                writer.add_scalar("Gen_Cycle", Gen_Cycle.item(),epoch*10000 + idx)
                writer.add_scalar("Gen_Iden", Gen_Iden.item(),epoch*10000 + idx)
                writer.add_scalar("Gen_unet", Gen_Iden.item(),epoch*10000 + idx)

                grid_optical = torchvision.utils.make_grid(fake_optical*0.5+0.5)
                grid_sar = torchvision.utils.make_grid(fake_sar*0.5+0.5)
                grid_predictions = torchvision.utils.make_grid(predictions*0.5+0.5)
                writer1 = SummaryWriter("./ドキュメント/Graph/pair,second,lr=5e-5,A=1,C=1.5,I=1,U=1,SN_D,SL")
                writer1.add_image("fake_optical",grid_optical,epoch*10000 + idx)
                writer1.add_image("fake_sar",grid_sar,epoch*10000 + idx)
                writer1.add_image("predictions",grid_predictions,epoch*10000 + idx)

            #writer = SummaryWriter(log_dir="./1")
            #writer.add_scalar("G_loss", G_loss1.item(),epoch)
            #writer.add_scalar("D_loss", D_loss1.item(),epoch)
            G_loss_iter = np.append(G_loss_iter, G_loss1)
            D_loss_iter = np.append(D_loss_iter, D_loss1)

        G_loss_epoch = np.append(G_loss_epoch, G_loss1)
        D_loss_epoch = np.append(D_loss_epoch, D_loss1)

        # UnetのCheckpoint
        """
        unet_checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":opt_gen.state_dict(),
        }
        """
   
        if config_unify.SAVE_MODEL:
            save_checkpoint(gen_O, opt_gen, filename=config_unify.CHECKPOINT_GEN_O)
            save_checkpoint(gen_S, opt_gen, filename=config_unify.CHECKPOINT_GEN_S)
            save_checkpoint(disc_O, opt_disc, filename=config_unify.CHECKPOINT_CRITIC_O)
            save_checkpoint(disc_S, opt_disc, filename=config_unify.CHECKPOINT_CRITIC_S)
            save_checkpoint(model, opt_gen,  filename=config_unify.CHECKPOINT_UNET)

        save_predictions_as_imgs(
            val_loader, model, folder="/home/yuki_murakami/ドキュメント/CycleGAN_UNet/lr=5e-5,A=1,C=1.5,I=1,U=1,SN_D,SL", device=config_unify.DEVICE
        )

        check_accuracy(val_loader, model, device=config_unify.DEVICE)   

        print(epoch)

    np.save("/home/yuki_murakami/ドキュメント/CycleGAN_UNet/lr=5e-5,A=1,C=1.5,I=1,U=1,SN_D,SL/numpy/G_loss_iter.npy",G_loss_iter)
    np.save("/home/yuki_murakami/ドキュメント/CycleGAN_UNet/lr=5e-5,A=1,C=1.5,I=1,U=1,SN_D,SL/numpy/D_loss_iter.npy",D_loss_iter)
    np.save("/home/yuki_murakami/ドキュメント/CycleGAN_UNet/lr=5e-5,A=1,C=1.5,I=1,U=1,SN_D,SL/numpy/G_loss_epoch.npy",G_loss_epoch)
    np.save("/home/yuki_murakami/ドキュメント/CycleGAN_UNet/lr=5e-5,A=1,C=1.5,I=1,U=1,SN_D,SL/numpy/D_loss_epoch.npy",D_loss_epoch)

def main():
    disc_O = Discriminator(in_channels=3).to(config_unify.DEVICE)
    disc_S = Discriminator(in_channels=3).to(config_unify.DEVICE)
    gen_S = Generator(img_channels=3, num_residuals=9).to(config_unify.DEVICE)
    gen_O = Generator(img_channels=3, num_residuals=9).to(config_unify.DEVICE)

    model = UNET(in_channels=6, out_channels=1).to(config_unify.DEVICE)

    opt_disc = optim.Adam(
        list(disc_O.parameters()) + list(disc_S.parameters()),
        lr=config_unify.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(gen_S.parameters()) + list(gen_O.parameters()),
        lr=config_unify.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    opt_unet = optim.Adam(model.parameters(), lr=config_unify.UNET_LEARNING_RATE)

    L1 = nn.L1Loss()
    mse = nn.MSELoss()
    BCEWithLogitsLoss = nn.BCEWithLogitsLoss()

    if config_unify.LOAD_MODEL:
        load_checkpoint(
            config_unify.CHECKPOINT_GEN_O, gen_O, opt_gen, config_unify.LEARNING_RATE,
        )
        load_checkpoint(
            config_unify.CHECKPOINT_GEN_S, gen_S, opt_gen, config_unify.LEARNING_RATE,
        )
        load_checkpoint(
            config_unify.CHECKPOINT_CRITIC_O, disc_O, opt_disc, config_unify.LEARNING_RATE,
        )
        load_checkpoint(
            config_unify.CHECKPOINT_CRITIC_S, disc_S, opt_disc, config_unify.LEARNING_RATE,
        )
        load_checkpoint(
            config_unify.CHECKPOINT_UNET, model, opt_disc, config_unify.LEARNING_RATE,
        )

    dataset = OpticalSARDataset(
        root_sar=config_unify.TRAIN_DIR+"sar_10000", root_optical=config_unify.TRAIN_DIR+"optical_10000", root_mask = config_unify.TRAIN_DIR+"groundtruth_10000", transform=config_unify.transforms #check
    )
    val_dataset = OpticalSARDataset(
       root_sar=config_unify.VAL_DIR+"sar_10000", root_optical=config_unify.VAL_DIR+"optical_10000", root_mask = config_unify.TRAIN_DIR+"groundtruth_10000", transform=config_unify.transforms
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=config_unify.BATCH_SIZE,
        shuffle=False,
        num_workers=config_unify.NUM_WORKERS,
        pin_memory=True
    )

    #check accuracy用のデータセット
    BATCH_SIZE = 1
    NUM_WORKERS = 2
    IMAGE_HEIGHT = 256  # 1280 originally
    IMAGE_WIDTH = 256  # 1918 originally
    PIN_MEMORY = True
    LOAD_MODEL = True
    TRAIN_IMG_DIR = "/home/yuki_murakami/ドキュメント/GAN_image_make/pair,lr=5e-5,A=1,C=1.5,I=0.5,SN_D,SL(colab)/sarconcat_10000/"
    TRAIN_MASK_DIR = "/home/yuki_murakami/ドキュメント/UNET/data_second/groundtruth_10000/"
    VAL_IMG_DIR = "/home/yuki_murakami/ドキュメント/GAN_image_make/pair,lr=5e-5,A=1,C=1.5,I=0.5,SN_D,SL(colab)/sarconcat_3000/"
    VAL_MASK_DIR = "/home/yuki_murakami/ドキュメント/UNET/data_second/groundtruth_3000/"

    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    train_loader_unet, val_loader_unet = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )


    if LOAD_MODEL:
        load_checkpoint_unet(torch.load("/home/yuki_murakami/ドキュメント/UNET/data_output/pair,lr=5e-5,A=1,C=1.5,I=0.5,SN_D,SL(colab)/checkpoint/unet_checkpoint.pth.tar"), model)

    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()
    u_scaler = torch.cuda.amp.GradScaler()

    train_fn(disc_O, disc_S, gen_S, gen_O, model, loader, val_loader_unet, opt_disc, opt_gen, opt_unet,L1, mse, BCEWithLogitsLoss, d_scaler, g_scaler,u_scaler)

if __name__ == "__main__":
    main()
