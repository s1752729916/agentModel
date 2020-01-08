import torch.nn as nn
import torchvision
import pandas as pd
import numpy as np
class DataProcess:

    def PreProcess(self):
        #数据预处理，读取数据并且将数据拼成CNN的图片模式
        raw = pd.read_csv('abaqus_12_8.csv')
        self.rawdata = raw[:,0:16]
        self.labels = raw[:,17:]


a = DataProcess()
a.PreProcess()
print(a.rawdata[0])
print(a.labels[0])





