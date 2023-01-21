import torch
import torchvision
from torch.utils.data import DataLoader
from dataset_unet import SegmantationDataset
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
import numpy as np

def save_checkpoint_unet(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint_unet(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
):
    #print("get_loaders")

    train_ds =  SegmantationDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
    )
    #print("train_ds")
    #print("train_ds.__getitem__(0)")
    #print(train_ds.__getitem__(0))
    
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    
    val_ds = SegmantationDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
    )
    #print("val_ds.__getitem__(0)")
    #print(val_ds.__getitem__(0))

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader

def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    acc = 0
    f1 = 0
    recall = 0
    pre = 0
    acc_t = 0
    f1_t = 0
    recall_t = 0
    pre_t = 0
    TP_t = 0
    TN_t = 0
    FN_t = 0
    FP_t = 0
    i = 0
    model.eval()

    with torch.no_grad():
        #iii= 0
        #print(iii)
        for x, y in loader:
            #iii+=1
            #print(iii)
            TP = 0
            TN = 0
            FN = 0
            FP = 0
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            y = (y > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )
            """
            preds  = preds.cpu().detach().numpy().copy()
            preds = np.squeeze(preds)
            preds = np.ravel(preds)
            y = y.cpu().detach().numpy().copy()
            y = np.squeeze(y)
            y = np.ravel(y)
            C = confusion_matrix(preds,y)
            TN,FP,FN,TP = C.flat
            TN_t += TN
            FP_t += FP
            FN_t += FN
            TP_t += TP
            i += 1
            #if i % 200 == 0:
                #print(i)
            """
            """
            acc = accuracy_score(y, preds)
            acc_t += acc
            pre = precision_score(y, preds)
            pre_t += pre
            recall = recall_score(y, preds)
            recall_t += recall
            f1 = f1_score(y, preds)
            f1_t += f1
            """

            """
            for i in range(256):
                for j in range(256):
                    if y[:,:,i,j] == 1 and preds[:,:,i,j] == 1:
                        TP += 1
                    if y[:,:,i,j] == 0 and preds[:,:,i,j] == 0:
                        TN += 1
                    if y[:,:,i,j] == 1 and preds[:,:,i,j] == 0:
                        FN += 1
                    if y[:,:,i,j] == 0 and preds[:,:,i,j] == 1:
                        FP += 1

            print(TP)
            """
            

            
    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    print(f"Dice score: {dice_score/len(loader)}")
    """
    print(f"Accuracy: {(TP_t + TN_t)/(TP_t + FN_t + TN_t + FP_t)*100:.2f}")
    print(f"Recall: {TP_t/(TP_t + FN_t)*100:.2f}")
    print(f"Precision: {TP_t/(TP_t + FP_t)*100:.2f}")
    print(f"F_score: {(2 * TP_t)/(2 * TP_t + FN_t + FP_t)*100:.2f}")
    """
    """
    print(f"Recall: {(recall_t/len(loader))*100:.2f}")
    print(f"Precision: {(pre_t/len(loader))*100:.2f}")
    print(f"F_score: {(f1_t/len(loader))*100:.2f}")
    """
    

    model.train()

def save_predictions_as_imgs(
    loader, model, folder="saved_images/", device="cuda"
):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/predictions/{idx}.jpeg"
        )
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}/unsqueeze/{idx}.jpeg")

    model.train()
