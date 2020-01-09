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
        self.norm_y =(self.rawlabels[:,0]-self.max)/(self.max-self.min)
    def Mat(self,Batch_size):
        #特征矩阵化
        temp = np.hstack((self.norm_x,self.norm_x))
        for i in range(239):
            temp = np.hstack((temp,self.norm_x))
        temp = temp[:,0:4096]
        print(temp.shape)
        temp = temp.reshape((temp.shape[0],64,64))
        temp = torch.from_numpy(temp)
        self.norm_y = torch.from_numpy(self.norm_y)
        self.dataset = TensorDataset(temp,self.norm_y)#数据转换成dataset
        self.trainLoader = DataLoader(self.dataset[0:1000],batch_size=Batch_size,shuffle=False)
        self.testLoader = DataLoader(self.dataset[1000:],batch_size=1,shuffle=False)#测试集Loade Batch Size = 1

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        x = self.pool(F.tanh(self.conv1(x)))
        x = self.pool(F.tanh(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = self.fc3(x)
        return x









