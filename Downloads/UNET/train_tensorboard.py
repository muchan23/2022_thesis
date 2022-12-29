import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model_unet import UNET
from torch.utils.tensorboard import SummaryWriter
from utils_unet import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)

# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
NUM_EPOCHS = 18
NUM_WORKERS = 2
IMAGE_HEIGHT = 256  # 1280 originally
IMAGE_WIDTH = 256  # 1918 originally
PIN_MEMORY = True
LOAD_MODEL = True
TRAIN_IMG_DIR = "drive/My Drive/Colab Notebooks/UiT_HCD_California_2017/UNET/train_10000/lr=0.00005,C=1,I=1,mono/sar_concat/"
TRAIN_MASK_DIR = "drive/My Drive/Colab Notebooks/UiT_HCD_California_2017/UNET/train_10000/groundtruth/"
VAL_IMG_DIR = "drive/My Drive/Colab Notebooks/UiT_HCD_California_2017/UNET/val_3000/lr=0.00005,C=1,I=1,mono/sar_concat/"
VAL_MASK_DIR = "drive/My Drive/Colab Notebooks/UiT_HCD_California_2017/UNET/val_3000/groundtruth/"

def train_fn(loader, val_loader, model, optimizer, loss_fn, scaler):
    
    for epoch in range(NUM_EPOCHS):
      loop = tqdm(loader)
      for batch_idx, (data, targets) in enumerate(loop):
          data = data.to(device=DEVICE)
          targets = targets.float().unsqueeze(1).to(device=DEVICE)

          # forward
          with torch.cuda.amp.autocast():
              predictions = model(data)
              loss = loss_fn(predictions, targets)

          # backward
          optimizer.zero_grad()
          scaler.scale(loss).backward()
          scaler.step(optimizer)
          scaler.update()

          loss_unet = loss.detach().clone().to('cpu').numpy()

                      #25は手動で変える必要がある    
          if batch_idx % 200 == 0:
            writer = SummaryWriter(log_dir="logs/lr=0.00005,C=1,I=1,mono") #check
            writer.add_scalar("loss", loss_unet.item(),epoch*10000 + batch_idx)

          # update tqdm loop
          loop.set_postfix(loss=loss.item())

      # save model
      checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
      }
      save_checkpoint(checkpoint, filename="drive/My Drive/Colab Notebooks/UiT_HCD_California_2017/UNET/checkpoint/lr=0.00005,C=1,I=1,mono/unet_checkpoint.pth.tar")

      # check accuracy
      check_accuracy(val_loader, model, device=DEVICE)

        # print some examples to a folder
      save_predictions_as_imgs(
            val_loader, model, folder="drive/My Drive/Colab Notebooks/UiT_HCD_California_2017/UNET/pred", device=DEVICE
      )


def main():
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

    model = UNET(in_channels=6, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
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
        load_checkpoint(torch.load("drive/My Drive/Colab Notebooks/UiT_HCD_California_2017/UNET/checkpoint/lr=0.00005,C=1,I=1,mono/unet_checkpoint.pth.tar"), model)

    #print("123")
    #print(list(val_loader))
    check_accuracy(val_loader, model, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()

    train_fn(train_loader, val_loader, model, optimizer, loss_fn, scaler)

if __name__ == "__main__":
    main()
