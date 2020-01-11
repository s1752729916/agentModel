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
        self.norm_x = torch.from_numpy(self.norm_x).float()
        self.norm_y = torch.from_numpy(self.norm_y).float()
        self.trainset = TensorDataset(self.norm_x[0:1300],self.norm_y[0:1300])#数据转换成dataset
        self.testset = TensorDataset(self.norm_x[1300:],self.norm_y[1300:])#数据转换成dataset

        self.trainLoader = DataLoader(self.trainset,batch_size=Batch_size,shuffle=False,drop_last=True,num_workers=0)
        self.testLoader = DataLoader(self.testset,batch_size=1,shuffle=False,drop_last=True,num_workers=0)#测试集Loade Batch Size = 1
    def DeNormlize(self,x):

        temp = x*(self.max-self.min)+self.min
        return temp

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.L1 = nn.Linear(17, 10)
        self.L2 = nn.Linear(10, 1)


    def forward(self, x):
        x = F.tanh(self.L1(x))
        nn.Dropout(0.5)
        x = self.L2(x)


        return x









