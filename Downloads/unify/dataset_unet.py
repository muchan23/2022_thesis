import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from natsort import natsorted

class  SegmantationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir #6channels, directory
        self.mask_dir = mask_dir #mask, directory
        self.transform = transform
        self.images = natsorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.images) #6channel_image_length

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index]) #ex) image_dir/0.npy
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".npy", ".jpeg")) #ex) mask_dir/
        image = np.array(np.load(img_path, allow_pickle=True))
        image = np.transpose(image, (1,2,0))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask == 255.0] = 1.0

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask
