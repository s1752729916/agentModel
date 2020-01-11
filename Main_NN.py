from AgentModel_NN import DataProcess,Net
import torch.nn as nn
import torchvision
import torch
from matplotlib import pyplot as plt

#数据预处理
#-----------------------------------
a = DataProcess()
a.PreProcess()
a.Normlize()
a.Mat(32)

#网络设置
#-----------------------------------
net = Net()
LearnRate=  0.025
epoch = 200
criterion = nn.MSELoss()
#训练网络
#-----------------------------------

print(len(a.trainLoader.dataset))
trainLoader = a.trainLoader
validation_test_loss = []
validation_x = []
for j in range(12):
    net = Net()
    test_loss_min = 1000
    LearnRate = 0.0001*2**j
    validation_x.append(LearnRate)
    optimizer = torch.optim.Adam(net.parameters(), lr=LearnRate)
    for k in range(epoch):
        run_loss=0
        for i,data in enumerate(trainLoader,0):
            inputs,labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            labels = labels.view(-1,1)
            loss = criterion(outputs,labels)
            loss.backward()#w误差反向传播
            optimizer.step()
            run_loss+=loss.item()
        print(k,'train loss:',run_loss)
        test_loss=0
        #测试集误差
        #---------------------------------
        for i,data in  enumerate(a.testLoader,0):
            inputs,labels = data
            outputs = net(inputs)
            outputs = a.DeNormlize(outputs)
            labels = a.DeNormlize(labels)
            test_loss += abs(labels-outputs)/(len(a.testLoader.dataset))
        if(test_loss<test_loss_min):
            test_loss_min = test_loss
        print(k, ' test loss:', test_loss)
    validation_test_loss.append(test_loss_min)


plt.plot(validation_x,validation_test_loss)
plt.show()
