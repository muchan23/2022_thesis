import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "/home/yuki_murakami/ドキュメント/UNET/data_second/"
VAL_DIR = "/home/yuki_murakami/ドキュメント/UNET/data_second/"
BATCH_SIZE = 1
LEARNING_RATE = 5e-5
UNET_LEARNING_RATE = 1e-5
LAMBDA_ADV = 0.05
LAMBDA_CYCLE = 10
LAMBDA_IDENTITY = 5
LAMBDA_UNET = 0.1
NUM_WORKERS = 4
NUM_EPOCHS = 3
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_GEN_O = "/home/yuki_murakami/ドキュメント/CycleGAN_UNet/lr=5e-5,A=1,C=1.5,I=1,U=1,SN_D,SL/checkpoint/geno.pth.tar"
CHECKPOINT_GEN_S = "/home/yuki_murakami/ドキュメント/CycleGAN_UNet/lr=5e-5,A=1,C=1.5,I=1,U=1,SN_D,SL/checkpoint/gens.pth.tar"
CHECKPOINT_CRITIC_O = "/home/yuki_murakami/ドキュメント/CycleGAN_UNet/lr=5e-5,A=1,C=1.5,I=1,U=1,SN_D,SL/checkpoint/critico.pth.tar"
CHECKPOINT_CRITIC_S = "/home/yuki_murakami/ドキュメント/CycleGAN_UNet/lr=5e-5,A=1,C=1.5,I=1,U=1,SN_D,SL/checkpoint/critics.pth.tar"
CHECKPOINT_UNET = "/home/yuki_murakami/ドキュメント/CycleGAN_UNet/lr=5e-5,A=1,C=1.5,I=1,U=1,SN_D,SL/checkpoint/unet.pth.tar"

transforms = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.HorizontalFlip(p=0),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
     ],
    additional_targets={"image0": "image"},
)
