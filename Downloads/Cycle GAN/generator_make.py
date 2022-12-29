import torch
from dataset import OpticalSARDataset
import sys
from utils import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config_make
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator_model import Discriminator
from generator_model import Generator

Optical = True
SAR = True

def testshow_fakesar(gen_S, loader,):
    loop = tqdm(loader, leave=True)

    for idx,  optical in enumerate(loop):
        optical = optical.to(config_make.DEVICE)

        #with torch.cuda.amp.autocast():
        fake_sar = gen_S(optical)

        if idx % 1 == 0:
            save_image(fake_sar*0.5+0.5, f"drive/My Drive/Colab Notebooks/UiT_HCD_California_2017/UNET/val_3000/lr=0.00005,C=1,I=1/optical_to_sar/{idx}.jpeg") #元 optical 後 sar

def testshow_fakeoptical(gen_O, loader,):
    loop = tqdm(loader, leave=True)

    for idx, sar in enumerate(loop):
        sar = sar.to(config_make.DEVICE)

        #with torch.cuda.amp.autocast():
        fake_optical = gen_O(sar)

        if idx % 1 == 0:
            save_image(fake_optical*0.5+0.5, f"drive/My Drive/Colab Notebooks/UiT_HCD_California_2017/UNET/val_3000/lr=0.00005,C=1,I=1/sar_to_optical/{idx}.jpeg") # 元 sar 後 optical

def main():
    test_optical = "drive/My Drive/Colab Notebooks/UiT_HCD_California_2017/UNET/val_3000/optical_10"
    test_sar = "drive/My Drive/Colab Notebooks/UiT_HCD_California_2017/UNET/val_3000/sar_mono_10"
    gen_S = Generator(img_channels=3, num_residuals=9).to(config_make.DEVICE)
    gen_O = Generator(img_channels=3, num_residuals=9).to(config_make.DEVICE)

    opt_gen = optim.Adam(
        list(gen_S.parameters()) + list(gen_O.parameters()),
        lr=config_make.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    if Optical:
        load_checkpoint(
            config_make.CHECKPOINT_GEN_S, gen_S, opt_gen, config_make.LEARNING_RATE,
        )

    optical_dataset = Dataset(
        root_change=test_optical, transform=config_make.transforms
    )

    loader = DataLoader(
        optical_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
    )

    testshow_fakesar(gen_S,loader)

    if SAR:
        load_checkpoint(
            config_make.CHECKPOINT_GEN_O, gen_O, opt_gen, config_make.LEARNING_RATE,
        )

    sar_dataset = Dataset(
        root_change=test_sar, transform=config_make.transforms
    )

    loader = DataLoader(
        sar_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
    )

    testshow_fakeoptical(gen_O,loader)


if __name__ == "__main__":
    main()
