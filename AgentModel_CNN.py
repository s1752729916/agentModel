import torch.nn as nn
import torchvision
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
            self.norm_x[:,i] = (self.rawdata[:,i]-max)/(max-min)
        self.norm_y = np.zeros((self.rawlabels.shape[0],1))
        self.max = np.max(self.rawlabels[:,0])
        self.min = np.min(self.rawlabels[:,0])
        self.norm_x =(self.rawlabels[:,0]-self.max)/(self.max-self.min)



a = DataProcess()
a.PreProcess()
print(a.rawdata.shape)
a.Normlize()
print(a.rawdata[0])





