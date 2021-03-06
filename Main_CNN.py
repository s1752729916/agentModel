from AgentModel_CNN import DataProcess,Net
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
LearnRate=  0.001
epoch = 200
lamda = 0.000
criterion = nn.MSELoss()
#训练网络
#-----------------------------------

print(len(a.trainLoader.dataset))
trainLoader = a.trainLoader
test_loss_min = 1000
validation_test_loss = []
validation_x = []
for j in range(1):
    net = Net()
    test_loss_min = 1000
    LearnRate =0.001
    validation_x.append(LearnRate)
    weight_p, bias_p = [], []
    for name, p in net.named_parameters():
        if 'bias' in name:
            bias_p += [p]
        else:
            weight_p += [p]
    optimizer = torch.optim.Adam([{'params': weight_p, 'weight_decay':0},
                      {'params': bias_p, 'weight_decay':0}],
                      lr=LearnRate,
                      )



    for k in range(epoch):
        run_loss=0
        for i,data in enumerate(trainLoader,0):
            inputs,labels = data
            inputs = inputs.view(-1,1,64,64)
            optimizer.zero_grad()
            outputs = net(inputs)
            labels = labels.view(-1,1)
            loss = criterion(outputs,labels)
            regularization_loss=0
            for param in net.parameters():
                regularization_loss += torch.sum(abs(param))

            loss +=   lamda * regularization_loss

            loss.backward()#w误差反向传播
            optimizer.step()
            run_loss+=loss.item()
        print(k,'train loss:',run_loss)
        test_loss=0
        #测试集误差
        #---------------------------------
        for i,data in  enumerate(a.testLoader,0):
            inputs,labels = data
            inputs=inputs.view(-1,1,64,64)
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
