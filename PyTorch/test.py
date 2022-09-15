# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 13:43:01 2022

@author: myf
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from scipy.optimize import linear_sum_assignment 
import os,sys
import copy
import numpy as np
import math
import h5py
import time
import pdb
import threading
import torch.optim as optim 

def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # 通过is_grad_enabled来判断当前模式是训练模式还是预测模式
    if not torch.is_grad_enabled():
        # 如果是在预测模式下，直接使用传入的移动平均所得的均值和方差
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # 使用全连接层的情况，计算特征维上的均值和方差
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            # 使用二维卷积层的情况，计算通道维上（axis=1）的均值和方差。
            # 这里我们需要保持X的形状以便后面可以做广播运算
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        # 训练模式下，用当前的均值和方差做标准化
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # 更新移动平均的均值和方差
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma * X_hat + beta  # 缩放和移位
    return Y, moving_mean.data, moving_var.data

class BatchNorm(nn.Module):
    # num_features：完全连接层的输出数量或卷积层的输出通道数。
    # num_dims：2表示完全连接层，4表示卷积层
    def __init__(self, num_features, num_dims):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # 参与求梯度和迭代的拉伸和偏移参数，分别初始化成1和0
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        # 非模型参数的变量初始化为0和1
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

    def forward(self, X):
        # 如果X不在内存上，将moving_mean和moving_var
        # 复制到X所在显存上
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        # 保存更新过的moving_mean和moving_var
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma, self.beta, self.moving_mean,
            self.moving_var, eps=1e-5, momentum=0.9)
        return Y
X = torch.randn(20, 10, 35, 45)
gamma = torch.randn(1,10, 1, 1)
beta = torch.randn(1,10, 1, 1)
Y = gamma * X + beta

x1 = torch.tensor([[1,2],[1,3]])
x2 = torch.tensor([[1,2]])
y =x1*x2
"""
list 与单变量的不同
涉及到python中的赋值、浅拷贝、深拷贝

"""
"""
T = 4
Z_pred = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1]])
Z_gt = np.array([[1,0,0],[1,0,0],[0,0,1]])
Dis = 1.0 - np.matmul(np.transpose(Z_gt),Z_pred)/T    
print(Dis)       #[K,M] <== [T,K]^T matmul [T,M] 
perm_gt, perm_pred = linear_sum_assignment(Dis)                #linear_sum_assignment 匈牙利算法实现分配
print('alignment cost {}'.format(np.sum(Dis[perm_gt,perm_pred]))) #分配算法带来的cost

Z_pred_perm = Z_pred[:,perm_pred] 
Z_gt_perm = Z_gt[:,perm_gt] 

pred_bg = 1-np.sum(Z_pred_perm,1)           #[T]
gt_bg = 1-np.sum(Z_gt_perm,1)               #[T]
Z_pred_perm = np.concatenate([Z_pred_perm,pred_bg[:,np.newaxis]],axis=1)     #[T,K+1] <== [T,K],[T,1]

"""

"""
intersect = np.multiply(Z_pred_perm,Z_gt_perm)                 #实现逻辑与的关系  since these are binary matrix, the elementwise product is only 1 if and only if the prediction and ground-truth values are both 1 
union = np.clip((Z_pred_perm+Z_gt_perm).astype(np.float),0,1)  #实现逻辑或的关系  clip实现约束数值范围
n_intersect = np.sum(intersect)  #预测正确的帧数
n_union = np.sum(union) 
n_predict = np.sum(Z_pred_perm)   #预测的所有帧数
         
n_gt = np.sum(Z_gt_perm) 
         
MoF = n_intersect/n_gt      #recall
IoU = n_intersect/n_union   #交割比
Precision = n_intersect/n_predict  #precision
print(MoF)
"""
"""
x = torch.randn(size=(1, 400, 49, 512))
output = x.new_ones((x.size()[:-1]))
alphas = F.softmax(output, dim = 2)
fbar = torch.einsum('bnr, bnrd -> bnd', alphas, x)  


class Discriminator(nn.Module):
    def __init__(self, im_chan = 3, hidden_dim = 32):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size = 3, stride = 2),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 64, kernel_size = 3, stride = 2),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 32, kernel_size = 3, stride = 2),
            nn.LeakyReLU(0.2),

            nn.Conv2d(32, 16, kernel_size = 3, stride = 2),
            nn.LeakyReLU(0.02),

            nn.Flatten(),
            #nn.Linear(5520, 16),
            nn.Linear(16, 1),
            nn.Sigmoid()

        )
        self.loss_function = nn.BCELoss()

        self.optimiser = torch.optim.Adam(self.parameters(), lr = 0.0001)

        self.counter = 0

        self.progress = []

    def forward(self, inputs):
        return self.model(inputs)

    def train(self, inputs, targets):

        outputs = self.forward(inputs)

        print(outputs.shape)

        targets = targets.unsqueeze(0)

        print(targets.shape)

        loss = self.loss_function(outputs, targets)
        print('Loss calculated!')
        self.counter += 1

        if (self.counter % 10 == 0):
            self.progress.append(loss.item())

        if (self.counter % 1000 == 0):
            print('counter = ', self.counter)

        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()


X = torch.randn(size=(1, 3, 270, 387))
target = torch.FloatTensor([1.0])
net = Discriminator()
net.train(X,target)
"""