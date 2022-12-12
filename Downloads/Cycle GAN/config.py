import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "/Users/murakamiyuki/Downloads/Cycle GAN/data/unpair_train"
VAL_DIR = "/Users/murakamiyuki/Downloads/Cycle GAN/data/unpair_val"
BATCH_SIZE = 1
LEARNING_RATE = 1e-5
LAMBDA_IDENTITY = 0.0
LAMBDA_CYCLE = 10
NUM_WORKERS = 4
NUM_EPOCHS = 10
LOAD_MODEL = True
SAVE_MODEL = True
CHECKPOINT_GEN_O = "/Users/murakamiyuki/Downloads/Cycle GAN/data/geno.pth.tar"
CHECKPOINT_GEN_S = "/Users/murakamiyuki/Downloads/Cycle GAN/data/gens.pth.tar"
CHECKPOINT_CRITIC_O = "/Users/murakamiyuki/Downloads/Cycle GAN/data/critico.pth.tar"
CHECKPOINT_CRITIC_S = "/Users/murakamiyuki/Downloads/Cycle GAN/data/critics.pth.tar"

transforms = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
     ],
    additional_targets={"image0": "image"},
)