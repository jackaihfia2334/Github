import torch
import tqdm
from sklearn.metrics import roc_auc_score

"""
def try_gpu(i=0):  #@save
    #如果存在,则返回gpu(i),否则返回cpu()
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')
"""

def train(model, optimizer, data_loader, criterion, device, log_interval=100):

    #model = model.to(device)
    model.train()
    total_loss = 0
    tk0 = tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)
    for i, (fields, target) in enumerate(tk0):
        fields=fields.long().to(device)
        target=target.long().to(device)
        y = model(fields)
        loss = criterion(y, target.float())
        model.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        """
        1.item()取出张量具体位置的元素元素值
        2.并且返回的是该位置元素值的高精度值
        3.保持原元素类型不变；必须指定位置
        """
        if (i + 1) % log_interval == 0:
            tk0.set_postfix(loss=total_loss / log_interval)
            total_loss = 0


def test(model, data_loader, device):
    #model = model.to(device)
    model.eval()
    targets, predicts = list(), list()
    with torch.no_grad():
        for fields, target in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
            fields=fields.long().to(device)
            target=target.long().to(device)
            y = model(fields)
            targets.extend(target.tolist())
            predicts.extend(y.tolist())
    return roc_auc_score(targets, predicts)

    
class EarlyStopper(object):

    def __init__(self, num_trials, save_path):
        self.num_trials = num_trials  #num_trials:表示尝试num_trials次后，如果没有提升就提前终止训练
        self.trial_counter = 0
        self.best_accuracy = 0
        self.save_path = save_path   #save_path：表示每次最优模型的存放路径

    def is_continuable(self, model, accuracy):
        if accuracy > self.best_accuracy:  
            self.best_accuracy = accuracy
            self.trial_counter = 0
            torch.save(model, self.save_path)
            return True
        elif self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            return True
        else:
            return False