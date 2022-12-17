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
