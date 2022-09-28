# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 15:45:13 2022

@author: myf
"""
import d2l 
#Import Library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader,TensorDataset
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import torch.optim.lr_scheduler as lr_scheduler


#加载数据
path1 = "C:/Users/ys/Desktop//note_code/ML_course/Assignment1/training.data"
path2 = "C:/Users/ys/Desktop//note_code/ML_course/Assignment1/testing.data"

df1 = pd.read_csv(path1, header=None, sep='\s+')
df2 = pd.read_csv(path2, header=None, sep='\s+')
#print(len(df))

np_tests = np.array(df1)
features = np_tests[...,:6]
labels = np_tests[...,6]
labels = (labels + 1)/2
#print(features[0])

#划分训练集和验证集
from sklearn.model_selection import train_test_split
train_X,test_X, train_y, test_y = train_test_split(features,labels,test_size = 0.1,random_state = 232)
train_X = torch.tensor(train_X).float()
train_y = torch.tensor(train_y).float()
test_X = torch.tensor(test_X).float()
test_y = torch.tensor(test_y).float()

train_dataset = TensorDataset(train_X,train_y)
test_dataset = TensorDataset(test_X,test_y)
train_iter = DataLoader(train_dataset, shuffle = True, batch_size = 2)
test_iter = DataLoader(test_dataset, shuffle = True, batch_size = 2)

#for features,labels in dl_train:
#    print(features,labels)
#    break


#模型定义
net = nn.Sequential(nn.Linear(6, 24),
                    nn.ReLU(),
                    nn.Linear(24,2))


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)
net.apply(init_weights)


#评估函数
def cross_entropy(y_hat, y):
    return - torch.log(y_hat[range(len(y_hat)), y])

def accuracy(y_hat, y):  #@save
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

def evaluate_accuracy(net, data_iter):  #@save
    #计算精度
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
    metric = Accumulator(2)  # 正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

class Accumulator:  #@save
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def train_epoch_ch3(net, train_iter, loss, updater):

    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y.long())
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])
        #scheduler.step()
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]

def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):  #@save

    for epoch in range(num_epochs):
        #adjust_learning_rate(updater, epoch, )
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        train_loss = train_metrics[0]
        train_acc = train_metrics[1]
        test_acc = evaluate_accuracy(net, test_iter)
        #print(f"epoch{epoch}: train_loss = {train_loss} ")
        print(f"epoch{epoch}: train_acc = {train_acc} ")
        #print(f"epoch{epoch}: test_acc = {test_acc} ")

    train_loss, train_acc = train_metrics
    return train_metrics
    #assert train_loss < 0.5, train_loss
    #assert train_acc <= 1 and train_acc > 0.7, train_acc
    #assert test_acc <= 1 and test_acc > 0.7, test_acc

"""
def sgd(params, lr, batch_size):  #@save
    #小批量随机梯度下降
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()
            
def updater(batch_size):
    return d2l.sgd([W, b], lr, batch_size)
"""
def adjust_learning_rate(optimizer, epoch, start_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = start_lr * (0.1 ** (epoch // 3))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


#训练过程
trainer = torch.optim.SGD(net.parameters(), lr=0.01)
#scheduler = lr_scheduler.StepLR(trainer, step_size=5, gamma=0.1)
#scheduler = torch.optim.lr_scheduler.LambdaLR(trainer, lr_lambda = lambda epoch: 1/(10*(epoch//10)+10))
#scheduler = lr_scheduler.ExponentialLR(trainer, gamma=0.9)  # 学习率调度器
loss = nn.CrossEntropyLoss(reduction='none')
num_epochs = 30
train_metric = train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)