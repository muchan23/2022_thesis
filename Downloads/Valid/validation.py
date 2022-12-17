from PIL import Image
import numpy as np

#img_t = Image.open("drive/My Drive/Colab Notebooks/UiT_HCD_California_2017/UNET/train_10000/groundtruth_1000/1.jpeg")
#img_p = Image.open("/content/pred_0.jpeg")
#img_p = img_p.convert(mode="L")

img_t = Image.open("/home/yuki_murakami/ドキュメント/UNET/data/groundtruth_3000/0.jpeg")
img_p = Image.open("/home/yuki_murakami/ドキュメント/UNET/data/prediction_image/pred_0.jpeg")
img_p = img_p.convert(mode="L")

#輝度値の取得と変更
def change_image(img):
  a = np.zeros((256,256))
  for i in range(256):
    for j in range(256):
      b = img.getpixel((i, j))
      if (b > 127):
        a[i, j] = 255
      else:
        a[i, j] = 0
  return a

img_truth = change_image(img_t)
img_predict = change_image(img_p)
print(img_truth)
print(img_predict)

#TP, TN, FP, FN
def valid(truth,predict):
  TP = 0
  TN = 0
  FN = 0
  FP = 0 
  Recall = 0
  Precision = 0
  F1 = 0
  for i in range(256):
    for j in range(256):
      if img_truth[i,j] == 255 and img_predict[i,j] == 255:
        TP += 1
      if img_truth[i,j] == 0 and img_predict[i,j] == 0:
        TN += 1
      if img_truth[i,j] == 255 and img_predict[i,j] == 0:
        FN += 1
      if img_truth[i,j] == 0 and img_predict[i,j] == 255:
        FP += 1
  
  print(TP,TN,FP,FN)
  Recall = TP / (TP + FN)
  Precision = TP / (TP + FP)
  F1 = (2 * Precision * Recall) / (Precision + Recall)

  print(f"Recall:{Recall}, Precision:{Precision}, F1:{F1}")

valid(img_truth,img_predict)
