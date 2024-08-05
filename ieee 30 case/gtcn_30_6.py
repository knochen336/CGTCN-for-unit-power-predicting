import csv

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import trange
from net.gtcn_6 import Net
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
import torch
import torch.nn.functional as F
from torch_geometric.data import Dataset
from torch_geometric.data import Data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
df = pd.read_csv("./output_data.csv", index_col=None, header=None)
data = df.values
reshaped_data = data.ravel(order='F')
reshaped_data = np.array(reshaped_data)
newdata = reshaped_data.reshape((365, 20, 288)).transpose((0, 2, 1))


class MyDataset(Dataset):
    def __init__(self, data_list):
        super(MyDataset, self).__init__(root=None, transform=None, pre_transform=None)
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


x_length = 6
y_length = 6
BATCHSIZE = 32
lEARINING_RATE = 0.0002
insert_positions = [1, 2, 3, 6, 7, 9, 11, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 25, 28, 29]
expanded_x = np.zeros((365, 288, 30))
expanded_x[:, :, insert_positions] = newdata

edgeindex = [[0, 1], [0, 2], [1, 3], [2, 3], [1, 4], [1, 5], [3, 5], [4, 6], [5, 6], [5, 27], [5, 7], [7, 27],
             [27, 26], [26, 29], [29, 28], [26, 28], [26, 24], [24, 25], [5, 8], [8, 10], [8, 9], [5, 9], [9, 20],
             [20, 21], [9, 16], [15, 16], [3, 11], [11, 12], [11, 17], [11, 15], [17, 18], [18, 19], [9, 19], [9, 23],
             [11, 13], [13, 14], [14, 22], [22, 23], [23, 24]]
undirected_edgeindex = edgeindex.copy()
for edge in edgeindex:
    undirected_edgeindex.append([edge[1], edge[0]])
edgeindex = undirected_edgeindex
edgeindex = np.array(edgeindex)
edgeindex = edgeindex.transpose()

df2 = pd.read_csv("./result_y_myself_addptdf.csv", index_col=None, header=None)
y_data = df2.values
y_data = y_data.ravel(order='F')
newy = y_data.reshape((365, 1728)).transpose((0, 1))

expanded_x = torch.from_numpy(expanded_x).float()
newy = torch.from_numpy(newy).float()
edgeindex = torch.from_numpy(edgeindex).long()
indices = torch.arange(365)
test_indices = torch.randperm(365)[:30]
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

for i in range(365 - 30):
    for j in range(288 - x_length + 1):
        x = expanded_x_train[i, j:j + x_length, :].t()
        y = newy_train[i][6 * j:6 * (j + y_length)]
        edge_index = edgeindex  # 邻接矩阵
        data = Data(x=x, edge_index=edge_index, y=y)
        data_list.append(data)
train_dataset = MyDataset(data_list)

data_list = []
for i in range(30):
    for j in range(288 - x_length + 1):
        x = expanded_x_test[i, j:j + x_length, :].t()
        y = newy_test[i][6 * j:6 * (j + y_length)]
        edge_index = edgeindex  # 邻接矩阵
        data = Data(x=x, edge_index=edge_index, y=y)
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

model = Net().to(device)

# pathstr = f'./model_best16_0.001_2.pth'  # 0.0999
# state = torch.load(pathstr, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
# model.load_state_dict(state)

optimizer = torch.optim.Adam(model.parameters(), lr=lEARINING_RATE, weight_decay=1e-5)  # 5e-4
EPOCH = 280
scheduler1 = MultiStepLR(optimizer, milestones=[70, 110, 150, 190, 240], gamma=0.5)  # 原来没有265
scheduler2 = MultiStepLR(optimizer,
                         milestones=[15, 30, 45],
                         gamma=0.8)

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
            y = data.y.view(-1, 6 * y_length)  # 改变 data.y 的形状
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
                y = data.y.view(-1, 6 * y_length)  # 改变 data.y 的形状
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
                y = data.y.view(-1, 6 * y_length)  # 改变 data.y 的形状
                f1loss = F.l1_loss(out, y)
                mseloss = F.mse_loss(out, y)
                epoch_test_loss += f1loss.item() / len(test_loader)
                epoch_test_mse += mseloss.item() / len(test_loader)
        val_losses.append(epoch_val_loss)
        val_mse.append(epoch_val_mse)
        test_losses.append(epoch_test_loss)
        test_mse.append(epoch_test_mse)
        print("validmaeloss:", epoch_val_loss, "valsmoothf1", smoothf1_val, "testmaeloss:", epoch_test_loss, "mse",
              epoch_test_mse)
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(),
                       f'./model_save/gtcn_best{BATCHSIZE}_{lEARINING_RATE}_{EPOCH}_6.pth')
        scheduler1.step()
        scheduler2.step()
    torch.save(model.state_dict(),
               f'./model_save/gtcn_last{BATCHSIZE}_{lEARINING_RATE}_{EPOCH}_6.pth')
    print("best_loss: ", best_val_loss)
    return train_losses, train_mse, val_losses, val_mse, test_losses, test_mse


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
with open('gtcn_6.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Train Losses', 'Train MSE', 'Validation Losses', 'Validation MSE', 'Test Losses', 'Test MSE'])
    writer.writerows(zip(train_losses, train_mse, val_losses, val_mse, test_losses, test_mse))
