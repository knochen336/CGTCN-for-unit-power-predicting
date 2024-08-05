import numpy as np
from torch.nn.utils import weight_norm
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from constrainet import (
    LinearConstraints,
    QuadraticConstraints,
    ConstraiNetLayer,
    DenseConstraiNet,
)

# 你的上限和下限
p_max = [100, 100, 100, 100, 550, 185, 100, 100, 100, 100, 320, 414, 100, 107, 100, 100, 100, 100, 100, 119, 304, 148,
         100, 100, 255, 260, 100, 491, 492, 805.2, 100, 100, 100, 100, 100, 100, 577, 100, 104, 707, 100, 100, 100, 100,
         352, 140, 100, 100, 100, 100, 136, 100, 100, 100]
p_min = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

max_change = 20
# 将上限和下限复制12次
p_max_12 = np.tile(p_max, 12)
p_min_12 = np.tile(p_min, 12)
x_length = 6
y_length = 6
import numpy as np

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
print(rampup[0, :])

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

]


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

        self.layernorm = torch.nn.LayerNorm(118)  # 添加LayerNorm层
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
        self.residual1 = torch.nn.Linear(x_length, x_length * 2)
        self.residual2 = torch.nn.Linear(x_length * 2, x_length * 4)
        self.tcn = TemporalConvNet(x_length * 4, [256, 128, 64])
        self.fc = torch.nn.Linear(118 * 64, y_length * 54)
        # self.fc2 = torch.nn.Linear(y_length * 128, y_length * 54)
        # self.conv4 = GCNConv(x_length * 4, x_length * 4)
        self.layernorm = torch.nn.LayerNorm(x_length * 4)
        self.layernorm2 = torch.nn.LayerNorm(118)
        # self.c_layer = ConstraiNetLayer(
        #     constraints=constraints,
        #     mode='center_projection',
        #     #  onto_edge=True,#新加的
        #     # 'center_projection' or 'ray_shift'
        # )

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        residual = self.residual1(x)
        x = self.conv1(x, edge_index)
        x = F.relu(x + residual)
        residual = self.residual2(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x + residual)
        x = x.view(-1, 118,
                   x_length * 4)  # 将数据的形状变为 [batch_size, num_nodes, num_features], [batch_size, features, sequence]
        x = self.layernorm(x)
        x = x.permute(0, 2, 1)  # 将数据的形状变为 [batch_size, num_features, num_nodes]
        x = self.tcn(x)  # 通过TCN处理时间序列
        # x = x.permute(0, 2, 1)  # 将数据的形状变回 [batch_size, num_nodes, num_features]
        # x = x.reshape(-1, x_length * 4)  # 将数据的形状变为 [num_nodes, num_features]
        # x = self.conv4(x, edge_index, edge_weight)  # 将调整后的输出送入新的GCN层
        x = F.relu(x)
        x = self.layernorm2(x)
        x = torch.flatten(x, start_dim=1)  # 展开张量
        x = x.view(-1, 118 * 64)  # 将数据的形状变为 [batch_size, 64*30]
        x = self.fc(x)
        # x = self.fc2(x)
        # x = self.c_layer(x)
        return x
