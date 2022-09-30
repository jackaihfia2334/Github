import numpy as np
import pandas as pd
import torch
from FM_pytorch.fm import FactorizationMachineModel
from FM_pytorch.movielens import MovieLens1MDataset
from FM_pytorch.train import train,test,EarlyStopper
from torch.utils.data import DataLoader

dataset=MovieLens1MDataset('./data/ml-1m/ratings.dat')
#field_dims = dataset.field_dims
#print(field_dims)
#offsets = np.array((0, *np.cumsum(field_dims)))   
model=FactorizationMachineModel(dataset.field_dims, embed_dim=16)

#按8:1:1比例拆分为训练集、验证集、测试集
train_length = int(len(dataset) * 0.8)
valid_length = int(len(dataset) * 0.1)
test_length = len(dataset) - train_length - valid_length
train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
    dataset, (train_length, valid_length, test_length))

#利用DataLoader加载，每个batch_size=256
train_data_loader = DataLoader(train_dataset, batch_size=256, num_workers=0)
valid_data_loader = DataLoader(valid_dataset, batch_size=256, num_workers=0)
test_data_loader = DataLoader(test_dataset, batch_size=256, num_workers=0)

def try_gpu(i=0):  #@save
    #如果存在,则返回gpu(i),否则返回cpu()
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

device = try_gpu()   #torch.device('cpu') 
print(device)
model = model.to(device)

criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001, weight_decay=0.000001)
#num_trials:表示尝试num_trials次后，如果没有提升就提前终止训练
#save_path：表示每次最优模型的存放路径
early_stopper = EarlyStopper(num_trials=2, save_path='result/model_001.pt')
#开始训练
for epoch_i in range(5):
    #print('第{}个epoch开始：'.format(epoch_i))
    train(model, optimizer, train_data_loader, criterion, device)
    auc_train = test(model, train_data_loader, device)
    auc_valid = test(model, valid_data_loader, device)
    auc_test = test(model, test_data_loader, device)
    print('第{}个epoch结束：'.format(epoch_i))
    """
    #print('训练集AUC:{}'.format(auc_train))
    #print('验证集AUC:{}'.format(auc_valid))
    #print('测试集AUC:{}'.format(auc_test))
    if not early_stopper.is_continuable(model, auc_valid):
        print('验证集上AUC的最高值是:{}'.format(early_stopper.best_accuracy))
        break
    """