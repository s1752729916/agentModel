from AgentModel_CNN import DataProcess,Net
import torch.nn as nn
import torchvision
import torch
#数据预处理
#-----------------------------------
a = DataProcess()
a.PreProcess()
a.Normlize()
a.Mat(32)

#网络设置
#-----------------------------------
net = Net()
LearnRate=  0.001
epoch = 100
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(),lr=LearnRate)
#训练网络
#-----------------------------------
for k in range(epoch):
    run_loss=0
    for i,data in  enumerate(a.trainLoader,0):
        inputs,labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs,labels)
        loss.backward()#w误差反向传播
        optimizer.step()
        run_loss+=loss.item()
    print(i,'train loss:',run_loss)
    test_loss=0
    #测试集误差
    #---------------------------------
    for i,data in  enumerate(a.testLoader,0):
        inputs,labels = data
        outputs = net(inputs)
        optimizer.step()
        loss = criterion(outputs,labels)
        test_loss+=loss.item()
    print(i,' test loss:',test_loss)
