import torch.nn as nn
import torchvision
import torch
from torch.utils.data import DataLoader,Dataset,TensorDataset
import torch.nn.functional as F
import pandas as pd
import numpy as np
class DataProcess:

    def PreProcess(self):
        #数据预处理，读取数据并且将数据拼成CNN的图片模式
        self.raw = pd.read_csv('abaqus_12_8_plastic.csv')
        self.raw = self.raw.values
        self.rawdata = self.raw[:,0:17]
        self.rawlabels = self.raw[:,17:]
    def Normlize(self):
        #线性归一化
        self.norm_x = np.zeros(self.rawdata.shape)
        for i in range(self.rawdata.shape[1]):
            max = np.max(self.rawdata[:,i])
            min = np.min(self.rawdata[:,i])
            self.norm_x[:,i] = (self.rawdata[:,i]-min)/(max-min)
        self.norm_y = np.zeros((self.rawlabels.shape[0],1))
        self.max = np.max(self.rawlabels[:,0])
        self.min = np.min(self.rawlabels[:,0])
        self.norm_y =(self.rawlabels[:,0]-self.min)/(self.max-self.min)
    def Mat(self,Batch_size):
        #特征矩阵化
        temp = np.hstack((self.norm_x,self.norm_x))
        for i in range(239):
            temp = np.hstack((temp,self.norm_x))
        temp = temp[:,0:4096]
        print(temp.shape)
        temp = temp.reshape((temp.shape[0],64,64))
        temp = torch.from_numpy(temp).float()
        self.norm_y = torch.from_numpy(self.norm_y).float()
        self.trainset = TensorDataset(temp[0:1300],self.norm_y[0:1300])#数据转换成dataset
        self.testset = TensorDataset(temp[1300:],self.norm_y[1300:])#数据转换成dataset

        self.trainLoader = DataLoader(self.trainset,batch_size=Batch_size,shuffle=False,drop_last=True,num_workers=0)
        self.testLoader = DataLoader(self.testset,batch_size=1,shuffle=False,drop_last=True,num_workers=0)#测试集Loade Batch Size = 1
    def DeNormlize(self,x):

        temp = x*(self.max-self.min)+self.min
        return temp

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1,32,3)
        self.pool1=nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(32,32,3)
        self.pool2=nn.MaxPool2d(2,2)
        self.conv3 = nn.Conv2d(32,32,3)
        self.pool3=nn.MaxPool2d(2,2)
        self.L1 =  nn.Linear(6*6*32,64)
        self.L2 = nn.Linear(64,1)



    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(-1,6*6*32)
        x = F.relu(self.L1(x))
        x = self.L2(x)



        return x









