import torch
from torch import nn,optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import os
import cv2
import mahotas
import numpy as np
from net import CNN_net

#圖像處理
def image(image):
    image=cv2.imread(input_image)
    re_image = cv2.resize(image,(28,28) )
    binary = cv2.cvtColor(re_image,cv2.COLOR_RGB2GRAY)
    blur=cv2.GaussianBlur(binary,(3,3),3)
    _,thresh = cv2.threshold(blur,140,255,cv2.THRESH_BINARY)
    #定义结构矩形元素
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
    erode = cv2.erode(thresh,kernel)       #腐蚀图像
    dilate = cv2.dilate(erode,kernel)      #膨胀图像
    dst = cv2.bitwise_not(dilate)          #取反
    return dst

input_image = '/home/yevette/文档/opencv_program/number identify/num2.jpg'    #要識別的圖片
dst = image(input_image)

cv2.namedWindow("result")
cv2.imshow("result",dst)
cv2.waitKey(1000)

#加载参数
cnn=CNN_net(1,10)
ckpt = torch.load('/home/yevette/文档/CNN_model_weight2.pth.tar')
cnn.load_state_dict(ckpt['state_dict'])      #加載到訓練好的模型中     
im_data = np.array(dst)
im_data = torch.from_numpy(im_data).float()
im_data = im_data.view(1, 1,28,28)
out = cnn(im_data)
_, predict = torch.max(out, 1)

print('预测結果为:{}'.format(predict))
