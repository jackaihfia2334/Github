# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 13:43:01 2022

@author: myf
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim



#定义一个LeNet网络
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.features=nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5), 
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2))
        
        self.classifier=nn.Sequential(\
            nn.Linear(16*5*5,120),
            nn.ReLU(),
            nn.Linear(120,84),
            nn.ReLU(),
            nn.Linear(84,10)
            )
            
    def forward(self, x):
        x=self.features(x)
        print(x.size())
        x=x.view(-1,16*5*5)
        x=self.classifier(x)
        return x
net=Net()

optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
print(optimizer.param_groups[0].keys())
optimizer.zero_grad()   #梯度清零，相当于net.zero_grad()
 
input=torch.randn(size =(1,1,28,28),requires_grad =True)
output=net(input)
output.backward(output)     #fake backward
optimizer.step()    #执行优化
 
#为不同子网络设置不同的学习率，在finetune中经常用到
#如果对某个参数不指定学习率，就使用默认学习率

 
#只为两个全连接层设置较大的学习率，其余层的学习率较小
special_layers=nn.ModuleList([net.classifier[0],net.classifier[3]])               #id() 函数返回对象的唯一标识符，标识符是一个整数。
special_layers_params=list(map(id,special_layers.parameters()))
base_params=filter(lambda p:id(p) not in special_layers_params,net.parameters())   
"""
map()是 Python 内置的高阶函数，它接收一个函数 f 和一个 list，并通过把函数 f 依次作用在 list 的每个元素上，得到一个新的 list 并返回。
eg: map(function,list1[])

"""
optimizer=optim.SGD([
                {'params': base_params},
                {'params':special_layers.parameters(), 'lr': 1e-3}
            ], lr=1e-2)


print(net.classifier[0],net.classifier[3])

