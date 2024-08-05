from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import pandas as pd
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm, trange
import torch
from constrainet import (
    LinearConstraints,
    QuadraticConstraints,
    ConstraiNetLayer,
    DenseConstraiNet,
)

# 你的上限和下限
p_max = [80, 80, 50, 55, 30, 40]
p_min = [0, 0, 0, 0, 0, 0]

# 将上限和下限复制12次
p_max_12 = np.tile(p_max, 12)
p_min_12 = np.tile(p_min, 12)
n_units = 6
n_periods = 12
# Initialize A matrix with zeros
rampup = np.zeros((n_units * (n_periods - 1), n_units * n_periods))
b1 = np.ones(n_units * (n_periods - 1)) * 8.25
for i in range(6 * 11):
    rampup[i][i] = 1
    rampup[i][i + 6] = -1

# 创建约束
constraints = [
    # 所有元素都大于等于0
    LinearConstraints(
        A=-np.eye(6 * 12),  # 注意这里是负的单位矩阵，因为我们要表示的是 x >= 0，而不是 x <= 0
        b=-p_min_12,  # 同样，这里也是负的下限
    ),
    # 所有元素都小于等于各自的上限
    LinearConstraints(
        A=np.eye(6 * 12),
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
]

x_length = 12
y_length = 12

from torch_geometric.nn import GCNConv
from torch.nn.utils import weight_norm


class Chomp1d(torch.nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(torch.nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(torch.nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                                 stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = torch.nn.LeakyReLU()
        self.layernorm = torch.nn.LayerNorm(30)  # 添加LayerNorm层
        self.net = torch.nn.Sequential(self.conv1, self.chomp1, self.relu1, self.layernorm)
        self.downsample = torch.nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = torch.nn.ReLU()
        self.init_weights()

    def init_weights(self):
        # self.conv1.weight.data.normal_(0, 0.01)
        torch.nn.init.xavier_normal_(self.conv1.weight)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(torch.nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size)]
        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(x_length, x_length * 2)
        self.conv2 = GCNConv(x_length * 2, x_length * 4)
        self.tcn = TemporalConvNet(x_length * 4, [256, 128, 64])
        self.fc = torch.nn.Linear(30 * 64, y_length * 6)
        self.layernorm = torch.nn.LayerNorm(x_length * 4)
        self.layernorm2 = torch.nn.LayerNorm(30)
        self.c_layer = ConstraiNetLayer(
            constraints=constraints,
            mode='center_projection'
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = x.view(-1, 30, x_length * 4)  # 将数据的形状变为 [batch_size, num_nodes, num_features]
        # x = self.layernorm(x)
        x = x.permute(0, 2, 1)  # 将数据的形状变为 [batch_size, num_features, num_nodes]
        x = self.tcn(x)  # 通过TCN处理时间序列
        x = F.relu(x)
        # x = self.layernorm2(x)
        x = torch.flatten(x, start_dim=1)  # 展开张量
        x = x.view(-1, 30 * 64)  # 将数据的形状变为 [batch_size, 64*30]
        x = self.fc(x)
        x = self.c_layer(x)
        return x
