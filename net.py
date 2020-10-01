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
#定義參數
batch_size = 200        # 每批的训练的样本数目
learning_rate = 0.01    # 学习率
num_epoches = 10        # 训练次数
DOWNLOAD_MNIST = False  # 是否從网上下载数据
#下載訓練集
train_dataset = datasets.MNIST(
    root = './mnist',
    train= True,       
    transform = transforms.ToTensor(),
    download=DOWNLOAD_MNIST
)
#下載測試機
test_dataset = datasets.MNIST(
    root='./mnist',
    train=False,      
    transform=transforms.ToTensor(),
    download=DOWNLOAD_MNIST
)
#裝載訓練集
train_loader = DataLoader(train_dataset, 
                    batch_size=batch_size, 
                    shuffle=True)       #打亂數據
#裝載測試集
test_loader = DataLoader(test_dataset, 
                    batch_size=batch_size, 
                    shuffle=False)

class CNN_net(nn.Module):
    def __init__(self,in_dim,n_class):
        super(CNN_net,self).__init__()
        #兩層卷積和兩層池化
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim,6,kernel_size=3,stride=1,padding=1),
            nn.ReLU(True),       
            nn.MaxPool2d(2,2),    
            nn.Conv2d(6,16,5,stride=1,padding=0),  
            nn.ReLU(True),
            nn.MaxPool2d(2,2)    
        )
        #全連接層
        self.fc = nn.Sequential(  
            nn.Linear(400,120),
            nn.Linear(120,84),
            nn.Linear(84,n_class)
        )
    #向前傳播函數
    def forward(self, x):
        out = self.conv(x)                  #先卷積
        out = out.view(out.size(0), -1)     #將參數扁平化
        out = self.fc(out)                  #最後通過全連接層進行分類
        return out


cnn = CNN_net(1, 10)
#定義交叉熵损失函数
criterion = nn.CrossEntropyLoss() 
#定義模型優化器        
optimizer = optim.Adam(cnn.parameters(), lr=learning_rate) 

if __name__ == 'main':          #只有在直接作爲腳本時執行
    for epoch in range(num_epoches):
            running_loss = 0.0
            running_acc = 0.0
            #训练
            for i,data in enumerate(train_loader,1):
                img,label = data
                img = Variable(img)
                label = Variable(label)
                out = cnn(img)
                loss = criterion(out,label)     #損失函數
                running_loss += loss.item() * label.size(0)
                _, pred = torch.max(out,1)
                optimizer.zero_grad()           #梯度歸零
                loss.backward()                 #反向傳播
                optimizer.step()                #更新參數

            print('Finish {} epoch,Loss:{:.6f}'.format(epoch+1,running_loss/(len(train_dataset))))

            #测试
            cnn.eval()     
            eval_loss = 0
            eval_acc = 0
            for i, data in enumerate(test_loader, 1):
                img, label = data
                img = Variable(img)
                label = Variable(label)
                out = cnn(img)
                loss = criterion(out,label)
                eval_loss += loss.item() * label.size(0)
                _, pred = torch.max(out, 1)
    
            print('Test Loss: {:.6f}'.format(eval_loss / (len(test_dataset))))

    # 將訓練好的模型保存下來
    ckpt_dir = '/home/yevette/文档'
    save_path = os.path.join(ckpt_dir, 'CNN_model_weight2.pth.tar')
    torch.save({'state_dict': cnn.state_dict()}, save_path)
