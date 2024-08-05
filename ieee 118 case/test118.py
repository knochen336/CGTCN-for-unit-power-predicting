import time
from sklearn.metrics import r2_score
from torch.utils.data.dataset import random_split
from torch.optim.lr_scheduler import LambdaLR
from net.GTCN_new_ontcn_2_cons import *
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
from utils.Mydataset import MyDataset
from torch.optim.lr_scheduler import LambdaLR

test_length = 30
seed = 5
x_length = 12
y_length = 12
BATCHSIZE = 16  

seed_everything(seed)

df = pd.read_csv("./original data/all_busloads.csv", index_col=None)
data = df.values
reshaped_data = data.ravel(order='F')
reshaped_data = np.array(reshaped_data)
newdata = reshaped_data.reshape((366, 99, 288)).transpose((0, 2, 1))
newdatacopy = newdata.copy()
length = 12
zeros = np.zeros((newdata.shape[0], newdata.shape[1], 1))
zero_point = [4, 8, 9, 24, 25, 29, 36, 37, 60, 62, 63, 64, 67, 68, 70, 80, 86, 88, 110]
for i in zero_point:
    newdata = np.concatenate((newdata[:, :, :i], zeros, newdata[:, :, i:]), axis=2)

edgeindex = pd.read_excel('./original data/edge_index.xlsx', header=None)
edgeindex = np.array(edgeindex)
edgeindex = edgeindex.transpose() - 1
edge_weight = np.array(pd.read_excel('./original data/edge_attr.xlsx', header=None)).transpose()
edge_weight = edge_weight[0]

df2 = pd.read_csv("./original data/y.csv", index_col=None)
y_data = df2.values
y_data = y_data.ravel(order='F')
newy = y_data.reshape((366, 15552)).transpose((0, 1))

edgeweight = torch.from_numpy(edge_weight).float()
expanded_x = torch.from_numpy(newdata).float()
newy = torch.from_numpy(newy).float()
edgeindex = torch.from_numpy(edgeindex).long()

indices = torch.arange(366)
test_indices = torch.randperm(366)[:test_length]

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

for i in range(366 - test_length):
    for j in range(277):
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
for i in range(test_length):
    for j in range(277):
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

device = torch.device('cpu')

model = Net().to(device)
pathstr = f'./model_save_118_new/model_best32_1e-05_400_12_cons.pth'
state = torch.load(pathstr, map_location=torch.device('cpu'))
model.load_state_dict(state)
model.eval()

print(test_indices)
# train_loss = 0
# valid_loss = 0
# test_loss = 0
# origin1 = torch.empty(0)
# result1 = torch.empty(0)
# for i in range(len(test_dataset)):
#     data = test_dataset[i]
#     y = data.y.view(-1, 54 * y_length)
#     out = model(data)
#     out = out[0, :]
#     y = y[0, :]
#     result1 = torch.cat((result1, out), dim=0)
#     origin1 = torch.cat((origin1, y), dim=0)
# loss = F.mse_loss(result1, origin1)
# print("mseloss:", loss)
#
# time1 = time.time()
# for i in range(24):
#     data = test_dataset[i]
#     y = data.y.view(-1, 54 * y_length)
#     out = model(data)
# time2 = time.time()
# print(time2 - time1)
#
# for i in range(test_length):
#     j = test_indices[i]
#     loss = 0
#     for k in range(24):
#         data = test_dataset[277 * i + 12 * k].to(device)
#         y = data.y.view(-1, 54 * y_length)
#         out = model(data)
#
#         loss += F.l1_loss(out, y)
#     print("day", j, "loss", loss / 24)
# print(len(test_dataset))
# results = pd.DataFrame(columns=["testsample", "day", "MAE", "MSE", "RMSE", "R2"])
# summse = 0
# for i in range(test_length):
#     j = test_indices[i]
#     origin = torch.empty(0)
#     result = torch.empty(0)
#     for k in range(24):
#         data = test_dataset[277 * i + 12 * k].to(device)
#         y = data.y.view(-1, 54 * y_length)
#         out = model(data)
#         y = y[0, :]
#         out = out[0, :]
#         result = torch.cat((result, out), dim=0)
#         origin = torch.cat((origin, y), dim=0)
#     result_sum = result.reshape(-1, 54).sum(dim=1)
#     origin_sum = origin.reshape(-1, 54).sum(dim=1)
#     data_sum = pd.DataFrame({"result_sum": result_sum.detach().numpy(), "origin_sum": origin_sum.detach().numpy()})
#     data_sum.to_excel(f"./test_result_new/data_sum{i}.xlsx", index=False)
#     #创建一个新的DataFrame来保存result和origin的值
#     datasave = pd.DataFrame({"result": result.detach().numpy(), "origin": origin.detach().numpy()})
#     #将DataFrame保存为Excel文件
#     datasave.to_excel(f"./test_result_new/data{i}.xlsx", index=False)
#     MAE = F.l1_loss(origin, result)
#     MSE = F.mse_loss(origin, result)
#     RMSE = torch.sqrt(MSE)
#     R2 = r2_score(origin.detach().numpy(), result.detach().numpy())
#     print(f"testsample:{i + 1} ", "day:", j, " MAE:", MAE, " MSE:", MSE, " RMSE:", RMSE, " R2:", R2, )
#
# # 将结果添加到DataFrame中
#     results = results.append(
#         {"testsample": i + 1, "day": j, "MAE": MAE.item(), "MSE": MSE.item(), "RMSE": RMSE.item(), "R2": R2},
#         ignore_index=True)
#
# # 将DataFrame保存为Excel文件
# results.to_excel("./test_result_new/118test_results.xlsx", index=False)

import torch


def adjust_unit_operations(unit_predict, power_predict, min_start, min_end, p_min):
    T, N = unit_predict.shape

    for g in range(N):
        change_flag = 0
        last_Ug = unit_predict[0, g]

        for t in range(1, T):
            if unit_predict[t, g] != last_Ug:
                change_flag = 1
                last_Ug = unit_predict[t, g]
                break

        if change_flag == 1:
            tg = []
            stage = []
            current_stage = unit_predict[0, g]
            start_time = 0

            for t in range(1, T):
                if unit_predict[t, g] != current_stage:
                    tg.append(t - start_time)
                    stage.append(current_stage)
                    start_time = t
                    current_stage = unit_predict[t, g]
            tg.append(T - start_time)
            stage.append(current_stage)

            i = 0
            while i < len(tg):
                adjusted = False
                if i == 0 and ((stage[i] == 1 and tg[i] < min_start and tg[i] < tg[i + 1]) or
                               (stage[i] == 0 and tg[i] < min_end and tg[i] < tg[i + 1])):
                    start = 0
                    end = tg[i]
                    unit_predict[start:end, g] = stage[i + 1]
                    power_predict[start:end, g] = 0 if stage[i + 1] == 0 else p_min
                    adjusted = True
                elif 0 < i < len(tg) - 1 and (
                        (stage[i] == 1 and tg[i] < min_start and tg[i] <= tg[i - 1] and tg[i] <= tg[i + 1]) or
                        (stage[i] == 0 and tg[i] < min_end and tg[i] <= tg[i - 1] and tg[i] <= tg[i + 1])):
                    start = sum(tg[:i])
                    end = start + tg[i]
                    unit_predict[start:end, g] = stage[i + 1]
                    power_predict[start:end, g] = 0 if stage[i + 1] == 0 else p_min
                    adjusted = True

                if adjusted:
                    tg[i] += tg.pop(i + 1)
                    stage.pop(i + 1)
                else:
                    i += 1

    return power_predict, unit_predict


for testday in range(30):

    print("day", testday + 1)
    pmin = 20
    min_start = 36
    min_end = 36

    f30 = pd.read_excel('f30_4.xlsx', header=None, index_col=None).values
    genptdf4 = pd.read_excel('gen_ptdf30_ 4.xlsx', index_col=None, header=None).values
    loadptdf4 = pd.read_excel('load_ptdf30_4.xlsx', index_col=None, header=None).values

    p_max = [100, 100, 100, 100, 550, 185, 100, 100, 100, 100, 320, 414, 100, 107, 100, 100, 100, 100, 100, 119, 304,
             148,
             100, 100, 255, 260, 100, 491, 492, 805.2, 100, 100, 100, 100, 100, 100, 577, 100, 104, 707, 100, 100, 100,
             100,
             352, 140, 100, 100, 100, 100, 136, 100, 100, 100]
    p_min = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    f12 = np.tile(f30, 12).T.reshape(-1)
    max_change = 20
    p_max_12 = np.tile(p_max, 12)
    p_min_12 = np.tile(p_min, 12)
    x_length = 12
    y_length = 12
    matrix_list = [[genptdf4 if i == j else np.zeros_like(genptdf4) for j in range(12)] for i in range(12)]
    A = np.block(matrix_list)
    day = test_indices[testday]
    # Number of units and time periods
    n_units = 54
    n_periods = 12

    # Initialize A matrix with zeros
    rampup = np.zeros((n_units * (n_periods - 1), n_units * n_periods))

    # Fill in the A matrix
    for i in range(54 * 11):
        rampup[i][i] = 1
        rampup[i][i + 54] = -1
    # Create b vector
    b1 = np.ones(n_units * (n_periods - 1)) * 20

    starttime = time.time()
    result = torch.empty(0)
    origin = torch.empty(0)
    for k in range(24):
        data = test_dataset[277 * testday + 12 * k]
        out = model(data)
        out = out[0, :]
        y = data.y.view(-1, 54 * y_length)
        y = y[0, :]

        result = torch.cat((result, out), dim=0)
        origin = torch.cat((origin, y), dim=0)
    origin = origin.reshape(-1, 54)
    result_sum = result.reshape(-1, 54)
    result_sum[result_sum < 10] = 0
    result_sum[(result_sum >= 10) & (result_sum < 20)] = 20
    power_predict = result_sum

    unit_predict = torch.where(result_sum > 5, torch.ones_like(result_sum), torch.zeros_like(result_sum))
    adjusted_power, adjusted_unit = adjust_unit_operations(unit_predict, power_predict, min_start, min_end, pmin)

    final = torch.empty(0)
    for k in range(24):
        time1 = 12 * k
        load12 = (loadptdf4 @ newdatacopy[day, time1:time1 + 12, :].T).T.reshape(-1)
        b_1 = - load12 + f12
        b_2 = f12 + load12
        constraints = [
            # 所有元素都大于等于0
            LinearConstraints(
                A=-np.eye(54 * 12),  # 注意这里是负的单位矩阵，因为我们要表示的是 x >= 0，而不是 x <= 0
                b=-p_min_12,  # 同样，这里也是负的下限
            ),
            # 所有元素都小于等于各自的上限
            LinearConstraints(
                A=np.eye(54 * 12),
                b=p_max_12,
            ),
            LinearConstraints(
                A=rampup,
                b=b1,
            ),
            LinearConstraints(
                A=-rampup,
                b=b1,
            ),
            LinearConstraints(
                A=-A,
                b=b_1,
            ),
            LinearConstraints(
                A=A,
                b=b_2,
            ),

        ]
        c_layer = ConstraiNetLayer(
            constraints=constraints,
            mode='center_projection',
        )
        final_power = c_layer(adjusted_power[time1:time1 + 12, :].reshape(-1))

        final = torch.cat((final, final_power[0, :]), dim=0)

    endtime = time.time()

    print("time", endtime - starttime)
    final = final.reshape(-1, 54)
    import torch.nn.functional as F

    print("MAE", F.l1_loss(origin, final), "MSE", F.mse_loss(origin, final))

    print("负载差值比例{:.10f}".format(-(origin.sum(dim=1) - final.sum(dim=1)).sum() / origin.sum()))

    tunit_price = pd.read_csv('./tunit_price.csv', index_col=None, encoding="gb2312")
    tunit_price = tunit_price[['NETCOST', 'INCCOST']]
    data = pd.read_excel(f'./test_result/data{testday}.xlsx', index_col=None).values

    total_cost0 = 0
    for i in range(54):  # 遍历每个机组
        a = tunit_price.iat[i, 0]
        b = tunit_price.iat[i, 1]
        for j in range(288):  # 遍历每个时刻
            x = final[j, i]
            cost =  (a * x ** 2) + b * x #0.5
            total_cost0 += cost
    print("预测成本是：", total_cost0)

    total_cost1 = 0
    for i in range(54):  # 遍历每个机组
        a = tunit_price.iat[i, 0]
        b = tunit_price.iat[i, 1]
        for j in range(288):  # 遍历每个时刻
            x = origin[j, i]
            cost = (a * x ** 2) + b * x
            total_cost1 += cost
    print("总成本是：", total_cost1)
    print("成本差值比例{:.10f}".format(- (total_cost1 - total_cost0) / total_cost1))
