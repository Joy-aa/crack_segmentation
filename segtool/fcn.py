import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
 
 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision import models
 
from tqdm import tqdm
import warnings

class FCN8(nn.Module): 
    def __init__(self, num_classes):
        super(FCN8, self).__init__()  
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=96,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=96),
            nn.MaxPool2d(kernel_size=2,padding=0)
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(in_channels=96,out_channels=256,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=256),
            nn.MaxPool2d(kernel_size=2,padding=0) 
        )
        
        self.stage3 = nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=384,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=384),
            
            nn.Conv2d(in_channels=384,out_channels=384,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=384),   
            
            nn.Conv2d(in_channels=384,out_channels=256,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=256),    
            
            nn.MaxPool2d(kernel_size=2,padding=0) 
        )
        
        self.stage4 = nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=512),
            
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=512),   
  
            nn.MaxPool2d(kernel_size=2,padding=0) 
        )
        
        self.stage5 = nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=num_classes,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=num_classes),
            
            nn.MaxPool2d(kernel_size=2,padding=0) 
        )
        
        #k倍上采样
        self.upsample_2 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=4, padding= 1,stride=2)
        self.upsample_4 = nn.ConvTranspose2d(in_channels=num_classes, out_channels=num_classes, kernel_size=4, padding= 0,stride=4)
        self.upsample_81 = nn.ConvTranspose2d(in_channels=512+num_classes+256, out_channels=512+num_classes+256, kernel_size=4, padding= 0,stride=4)
        self.upsample_82 = nn.ConvTranspose2d(in_channels=512+num_classes+256, out_channels=512+num_classes+256, kernel_size=4, padding= 1,stride=2)
        #最后的预测模块
        self.final = nn.Sequential(
            nn.Conv2d(512+num_classes+256, num_classes, kernel_size=7, padding=3),
        )
        
    def forward(self, x):
        x = x.float()
        #conv1->pool1->输出
        x = self.stage1(x)
        #conv2->pool2->输出
        x = self.stage2(x)
        #conv3->pool3->输出输出, 经过上采样后, 需要用pool3暂存
        x = self.stage3(x)
        pool3 = x
        #conv4->pool4->输出输出, 经过上采样后, 需要用pool4暂存
        x = self.stage4(x)
        pool4 = self.upsample_2(x)
 
        x = self.stage5(x)
        conv7 = self.upsample_4(x)
 
        #对所有上采样过的特征图进行concat, 在channel维度上进行叠加
        x = torch.cat([pool3, pool4, conv7], dim = 1)
        #经过一个分类网络,输出结果(这里采样到原图大小,分别一次2倍一次4倍上采样来实现8倍上采样)
        output = self.upsample_81(x)
        output = self.upsample_82(output)
        output = self.final(output)
 
        return output

if __name__ == '__main__':
    import cv2 as cv
    import os
    img = cv.imread('/home/wj/local/crack_segmentation/images/creak_s28.jpg')
    # path = 'ima'
    img_height, img_width, img_channels = img.shape
    input_size=(448,448)
    cof = 1
    w, h = int(cof * input_size[0]), int(cof * input_size[1])
    for i in range(0, img_height+h, h):
        for j in range(0, img_width+w, w):
            i1 = i
            j1 = j
            i2 = i + h
            j2 = j + w
            if i2>img_height:
                        i1 = max(img_height - h, 0)
                        i2 = img_height
            if j2>img_width:
                        j1 = max(img_width - w, 0)
                        j2 = img_width
            img_pat = img[i1:i2, j1:j2]
            # print(i1,i2,j1,j2)
            cv.imwrite(os.path.join('../result_images', 'creak_s28'+str(i)+str(j)+'.jpg'), img_pat)
