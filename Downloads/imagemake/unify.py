import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "drive/My Drive/Colab Notebooks/UiT_HCD_California_2017/UNET/val_3000/"
VAL_DIR = "drive/My Drive/Colab Notebooks/UiT_HCD_California_2017/UNET/val_3000/"
BATCH_SIZE = 1
LEARNING_RATE = 5e-5
LAMBDA_IDENTITY = 1
LAMBDA_CYCLE = 1
NUM_WORKERS = 4
NUM_EPOCHS = 3
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_GEN_O = "drive/My Drive/Colab Notebooks/UiT_HCD_California_2017/CycleGAN_UNET/lr=5e-5,C=1,I=1,U=1/checkpoint/geno.pth.tar"
CHECKPOINT_GEN_S = "drive/My Drive/Colab Notebooks/UiT_HCD_California_2017/CycleGAN_UNET/lr=5e-5,C=1,I=1,U=1/checkpoint/gens.pth.tar"
CHECKPOINT_CRITIC_O = "drive/My Drive/Colab Notebooks/UiT_HCD_California_2017/CycleGAN_UNET/lr=5e-5,C=1,I=1,U=1/checkpoint/critico.pth.tar"
CHECKPOINT_CRITIC_S = "drive/My Drive/Colab Notebooks/UiT_HCD_California_2017/CycleGAN_UNET/lr=5e-5,C=1,I=1,U=1/checkpoint/critics.pth.tar"
CHECKPOINT_UNET = "drive/My Drive/Colab Notebooks/UiT_HCD_California_2017/CycleGAN_UNET/lr=5e-5,C=1,I=1,U=1/checkpoint/unet.pth.tar"

transforms = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
     ],
    additional_targets={"image0": "image"},
)

from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np
from natsort import natsorted

class Dataset(Dataset):
    def __init__(self, root_change, transform=None):
        self.root_change = root_change
        self.transform = transform

        self.change_images = natsorted(os.listdir(root_change))
        self.change_len = len(self.change_images)

    def __len__(self):
        return self.change_len

    def __getitem__(self, index):
        change_img = self.change_images[index % self.change_len]

        change_path = os.path.join(self.root_change, change_img)  #パスの結合

        change_img = np.array(Image.open(change_path).convert("RGB"))

        if self.transform:
           augmentations = self.transform(image=change_img)
           change_img = augmentations["image"]

        return change_img



from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np

# sar = zebra optical = horse

class OpticalSARDataset(Dataset):
    def __init__(self, root_sar, root_optical, transform=None):
        self.root_sar = root_sar
        self.root_optical = root_optical
        self.transform = transform

        self.sar_images = os.listdir(root_sar)
        self.optical_images = os.listdir(root_optical)
        self.length_dataset = max(len(self.sar_images), len(self.optical_images)) # 1000, 1500
        self.sar_len = len(self.sar_images)
        self.optical_len = len(self.optical_images)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        sar_img = self.sar_images[index % self.sar_len]
        optical_img = self.optical_images[index % self.optical_len]

        sar_path = os.path.join(self.root_sar, sar_img)  #パスの結合
        optical_path = os.path.join(self.root_optical, optical_img)

        sar_img = np.array(Image.open(sar_path).convert("RGB"))
        optical_img = np.array(Image.open(optical_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=sar_img, image0=optical_img)
            sar_img = augmentations["image"]
            optical_img = augmentations["image0"]

        return sar_img, optical_img      
      
#both
import torch
#from dataset import OpticalSARDataset
import sys
from utils import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
#import config_make
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator_model import Discriminator
from generator_model import Generator
from model_unet import UNET

Optical = True
SAR = True

def testshow_fakesar(gen_S, model, loader):
    loop = tqdm(loader, leave=True)

    for idx,  (optical,sar) in enumerate(loop):
        optical = optical.to(DEVICE)
        sar = sar.to(DEVICE)

        #with torch.cuda.amp.autocast():
        fake_sar = gen_S(optical)
        concat_sar = torch.cat((sar,fake_sar),1)
        predictions = model(concat_sar) #modelの定義を行う

        if idx % 1 == 0:
            save_image(fake_sar*0.5+0.5, f"drive/My Drive/Colab Notebooks/UiT_HCD_California_2017/UNET/trash/fakesar_{idx}.jpeg") #元 optical 後 sar
            save_image(predictions*0.5+0.5, f"drive/My Drive/Colab Notebooks/UiT_HCD_California_2017/UNET/trash/predictions_sar{idx}.jpeg")

def testshow_fakeoptical(gen_O, model, loader):
    loop = tqdm(loader, leave=True)

    for idx, (sar, optical) in enumerate(loop):
        sar = sar.to(DEVICE)
        optical = optical.to(DEVICE)

        #with torch.cuda.amp.autocast():
        fake_optical = gen_O(sar)
        concat_optical = torch.cat((sar,fake_optical),1)
        predictions = model(concat_optical) #modelの定義を行う

        if idx % 1 == 0:
            save_image(fake_optical*0.5+0.5, f"drive/My Drive/Colab Notebooks/UiT_HCD_California_2017/UNET/trash/fakeoptical_{idx}.jpeg") # 元 sar 後 optical
            save_image(predictions*0.5+0.5, f"drive/My Drive/Colab Notebooks/UiT_HCD_California_2017/UNET/trash/predictions_optical{idx}.jpeg")


def main():
    test_optical = "drive/My Drive/Colab Notebooks/UiT_HCD_California_2017/UNET/val_3000/optical_10"
    test_sar = "drive/My Drive/Colab Notebooks/UiT_HCD_California_2017/UNET/val_3000/sar_10"
    gen_S = Generator(img_channels=3, num_residuals=9).to(DEVICE)
    gen_O = Generator(img_channels=3, num_residuals=9).to(DEVICE)
    model = UNET(in_channels=6, out_channels=1).to(DEVICE)

    opt_gen = optim.Adam(
        list(gen_S.parameters()) + list(gen_O.parameters()) + list(model.parameters()),
        lr=LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    if Optical:
        load_checkpoint(
            CHECKPOINT_GEN_S, gen_S, opt_gen, LEARNING_RATE,
        )
        load_checkpoint(
            CHECKPOINT_UNET, model, opt_gen, LEARNING_RATE,
        )

    optical_dataset = OpticalSARDataset(
        root_sar = test_sar, root_optical=test_optical, transform=transforms
    )

    loader = DataLoader(
        optical_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
    )

    testshow_fakesar(gen_S, model, loader)

    if SAR:
        load_checkpoint(
            CHECKPOINT_GEN_O, gen_O, opt_gen, LEARNING_RATE,
        )
        load_checkpoint(
            CHECKPOINT_UNET, model, opt_gen, LEARNING_RATE,
        )

    sar_dataset = OpticalSARDataset(
        root_sar=test_sar, root_optical = test_optical, transform=transforms
    )

    loader = DataLoader(
        sar_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
    )

    testshow_fakeoptical(gen_O, model, loader)


if __name__ == "__main__":
    main()
    
 #SAR
import torch
#from dataset import OpticalSARDataset
import sys
from utils import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
#import config_make
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator_model import Discriminator
from generator_model import Generator
from model_unet import UNET

Optical = True

def testshow_fakesar(gen_S, model, loader):
    loop = tqdm(loader, leave=True)

    for idx,  (optical,sar) in enumerate(loop):
        optical = optical.to(DEVICE)
        sar = sar.to(DEVICE)

        #with torch.cuda.amp.autocast():
        fake_sar = gen_S(optical)
        concat_sar = torch.cat((sar,fake_sar),1)
        predictions = model(concat_sar) #modelの定義を行う

        if idx % 1 == 0:
            save_image(fake_sar*0.5+0.5, f"drive/My Drive/Colab Notebooks/UiT_HCD_California_2017/CycleGAN_UNET/lr=5e-5,C=1,I=1,U=1/val_fakesar/{idx}.jpeg") #元 optical 後 sar
            save_image(predictions*0.5+0.5, f"drive/My Drive/Colab Notebooks/UiT_HCD_California_2017/CycleGAN_UNET/lr=5e-5,C=1,I=1,U=1/sarbase_prediction/{idx}.jpeg")


def main():
    test_optical = "drive/My Drive/Colab Notebooks/UiT_HCD_California_2017/UNET/val_3000/optical_10"
    test_sar = "drive/My Drive/Colab Notebooks/UiT_HCD_California_2017/UNET/val_3000/sar_10"
    gen_S = Generator(img_channels=3, num_residuals=9).to(DEVICE)
    gen_O = Generator(img_channels=3, num_residuals=9).to(DEVICE)
    model = UNET(in_channels=6, out_channels=1).to(DEVICE)

    opt_gen = optim.Adam(
        list(gen_S.parameters()) + list(gen_O.parameters()) + list(model.parameters()),
        lr=LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    if Optical:
        load_checkpoint(
            CHECKPOINT_GEN_S, gen_S, opt_gen, LEARNING_RATE,
        )
        load_checkpoint(
            CHECKPOINT_UNET, model, opt_gen, LEARNING_RATE,
        )

    optical_dataset = OpticalSARDataset(
        root_sar = test_sar, root_optical=test_optical, transform=transforms
    )

    loader = DataLoader(
        optical_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
    )

    testshow_fakesar(gen_S, model, loader)

if __name__ == "__main__":
    main()
    
#optical    
    
import torch
#from dataset import OpticalSARDataset
import sys
from utils import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
#import config_make
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator_model import Discriminator
from generator_model import Generator
from model_unet import UNET

SAR = True

def testshow_fakeoptical(gen_O, model, loader):
    loop = tqdm(loader, leave=True)

    for idx, (sar, optical) in enumerate(loop):
        sar = sar.to(DEVICE)
        optical = optical.to(DEVICE)

        #with torch.cuda.amp.autocast():
        fake_optical = gen_O(sar)
        concat_optical = torch.cat((sar,fake_optical),1)
        predictions = model(concat_optical) #modelの定義を行う

        if idx % 1 == 0:
            save_image(fake_optical*0.5+0.5, f"drive/My Drive/Colab Notebooks/UiT_HCD_California_2017/UNET/trash/fakeoptical_{idx}.jpeg") # 元 sar 後 optical
            save_image(predictions*0.5+0.5, f"drive/My Drive/Colab Notebooks/UiT_HCD_California_2017/UNET/trash/predictions_optical{idx}.jpeg")


def main():
    test_optical = "drive/My Drive/Colab Notebooks/UiT_HCD_California_2017/UNET/val_3000/optical_10"
    test_sar = "drive/My Drive/Colab Notebooks/UiT_HCD_California_2017/UNET/val_3000/sar_10"
    gen_S = Generator(img_channels=3, num_residuals=9).to(DEVICE)
    gen_O = Generator(img_channels=3, num_residuals=9).to(DEVICE)
    model = UNET(in_channels=6, out_channels=1).to(DEVICE)

    opt_gen = optim.Adam(
        list(gen_S.parameters()) + list(gen_O.parameters()) + list(model.parameters()),
        lr=LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    if SAR:
        load_checkpoint(
            CHECKPOINT_GEN_O, gen_O, opt_gen, LEARNING_RATE,
        )
        load_checkpoint(
            CHECKPOINT_UNET, model, opt_gen, LEARNING_RATE,
        )

    sar_dataset = OpticalSARDataset(
        root_sar=test_sar, root_optical = test_optical, transform=transforms
    )

    loader = DataLoader(
        sar_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
    )

    testshow_fakeoptical(gen_O, model, loader)


if __name__ == "__main__":
    main()
