import csv
import time

from torch.utils.data.dataset import random_split
from torch.optim.lr_scheduler import LambdaLR
from net.GTCN_new_ontcn_24 import *
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import trange
from torch_geometric.data import Data
from torch_geometric.data import Dataset
from utils.seed import seed_everything
from torch_geometric.loader import DataLoader
import torch
import torch.nn.functional as F
from utils.ieee118_process import process_ieee118_attr
from utils.Mydataset import MyDataset
from torch.optim.lr_scheduler import LambdaLR


def custom_loss(output, target, alpha=1):
    # 计算L1损失
    l1_loss = F.l1_loss(output, target)

    # 计算每个时刻总发电量之间的差异
    output_sum = output.view(-1, 6).sum(dim=1)
    target_sum = target.view(-1, 6).sum(dim=1)
    sum_loss = F.l1_loss(output_sum, target_sum)
    # print( l1_loss,sum_loss)

    # 返回加权和
    return alpha * l1_loss + (1 - alpha) * sum_loss


smoothf1 = torch.nn.SmoothL1Loss(reduction='mean', beta=0.5)


def train():
    model.train()
    best_val_loss = 99999
    train_losses = []
    train_mse = []
    val_losses = []
    val_mse = []
    test_losses = []
    test_mse = []

    for epoch in trange(EPOCH, desc="Epochs"):
        model.train()
        epoch_train_loss = 0
        epoch_train_mse = 0
        smoothf1_train = 0
        for step, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()

            out = model(data)
            y = data.y.view(-1, 54 * y_length)  # 改变 data.y 的形状
            loss = smoothf1(out, y)
            f1loss = F.l1_loss(out, y)
            mseloss = F.mse_loss(out, y)
            loss.backward()
            optimizer.step()
            epoch_train_loss += f1loss.item() / len(train_loader)
            epoch_train_mse += mseloss.item() / len(train_loader)
            smoothf1_train += loss.item() / len(train_loader)
        train_losses.append(epoch_train_loss)
        train_mse.append(epoch_train_mse)
        print("trainmaeloss:", epoch_train_loss, "train_smoothf1", smoothf1_train)
        epoch_val_loss = 0
        epoch_test_loss = 0
        epoch_test_mse = 0
        epoch_val_mse = 0
        smoothf1_val = 0
        model.eval()
        with torch.no_grad():
            for step, data in enumerate(val_loader):
                data = data.to(device)
                out = model(data)
                y = data.y.view(-1, 54 * y_length)  # 改变 data.y 的形状
                f1loss = F.l1_loss(out, y)
                mseloss = F.mse_loss(out, y)
                loss = smoothf1(out, y)
                epoch_val_loss += f1loss.item() / len(val_loader)
                epoch_val_mse += mseloss.item() / len(val_loader)
                smoothf1_val += loss.item() / len(val_loader)
        with torch.no_grad():
            for step, data in enumerate(test_loader):
                data = data.to(device)
                out = model(data)
                y = data.y.view(-1, 54 * y_length)  # 改变 data.y 的形状
                f1loss = F.l1_loss(out, y)
                mseloss = F.mse_loss(out, y)
                epoch_test_loss += f1loss.item() / len(test_loader)
                epoch_test_mse += mseloss.item() / len(test_loader)
        val_losses.append(epoch_val_loss)
        val_mse.append(epoch_val_mse)
        test_losses.append(epoch_test_loss)
        test_mse.append(epoch_test_mse)
        print("validmaeloss:", epoch_val_loss, "valsmoothf1", smoothf1_val, "testmaeloss:", epoch_test_loss,"mse",epoch_test_mse)
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(),
                       f'./model_save_118_new/model_best{BATCHSIZE}_{lEARINING_RATE}_{EPOCH}_length24.pth')
        scheduler1.step()
        scheduler2.step()
    torch.save(model.state_dict(),
               f'./model_save_118_new/model_last{BATCHSIZE}_{lEARINING_RATE}_{EPOCH}_length24.pth')
    print("best_loss: ", best_val_loss)
    return train_losses, train_mse, val_losses, val_mse, test_losses, test_mse


seed = 5  # 0.0845
x_length = 24
y_length = 24
BATCHSIZE =32 # 不能太大 16
lEARINING_RATE = 0.0008
EPOCH = 400

seed_everything(seed)

expanded_x, newy, edgeindex, edgeweight = process_ieee118_attr()
indices = torch.arange(366)
test_indices = torch.randperm(366)[:30]
# 按照这些数值在序列中的顺序排序
test_indices = test_indices.sort().values
# 生成测试集
expanded_x_test = expanded_x[test_indices]
newy_test = newy[test_indices]
# 生成训练集
train_indices = torch.tensor([i for i in indices if i not in test_indices])
expanded_x_train = expanded_x[train_indices]
newy_train = newy[train_indices]

data_list = []

for i in range(366 - 30):
    for j in range(288-x_length+1):
        x = expanded_x_train[i, j:j + x_length, :].t()
        y = newy_train[i][54 * j:54 * (j + y_length)]
        edge_index = edgeindex  # 邻接矩阵
        edge_weight = edgeweight
        data = Data(x=x, edge_index=edge_index, y=y, edge_weight=edge_weight)
        data_list.append(data)
train_dataset = MyDataset(data_list)
print(p_max_12.shape, p_min_12.shape)
print(train_dataset[0].y)
data_list = []
for i in range(30):
    for j in range(288-x_length+1):
        x = expanded_x_test[i, j:j + x_length, :].t()
        y = newy_test[i][54 * j:54 * (j + y_length)]
        edge_index = edgeindex  # 邻接矩阵
        edge_weight = edgeweight
        data = Data(x=x, edge_index=edge_index, y=y, edge_weight=edge_weight)
        data_list.append(data)
test_dataset = MyDataset(data_list)

# 划分数据集
train_dataset_size = len(train_dataset)
train_size = int(train_dataset_size * 0.9)
train_dataset, val_dataset = random_split(train_dataset, [train_size, train_dataset_size - train_size])
# train_dataset = train_dataset[:train_size]
# val_dataset = train_dataset[train_size:]
print(len(train_dataset), len(val_dataset), len(test_dataset))
# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=BATCHSIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCHSIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCHSIZE, shuffle=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Net().to(device)


optimizer = torch.optim.Adam(model.parameters(), lr=lEARINING_RATE, weight_decay=1e-5)  # 5e-4

scheduler1 = MultiStepLR(optimizer, milestones=[90, 130, 170, 210, 260, 290], gamma=0.5)  # 原来没有265
scheduler2 = MultiStepLR(optimizer,
                         milestones=[30, 40, 50, 60, 70, 80],
                         gamma=0.8)

train_losses, train_mse, val_losses, val_mse, test_losses, test_mse = train()

# 绘制损失函数随训练轮次变化的图
plt.figure()
plt.plot(train_losses[10:], label='Train Loss')
plt.plot(val_losses[10:], label='Validation Loss')
plt.plot(train_mse[10:], label='Train mse')
plt.plot(val_mse[10:], label='Validation mse')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
with open('gtcn_new_12_losses_and_mse_smoothl1_batch16_24.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Train Losses', 'Train MSE', 'Validation Losses', 'Validation MSE', 'Test Losses', 'Test MSE'])
    writer.writerows(zip(train_losses, train_mse, val_losses, val_mse, test_losses, test_mse))
