
import torch.nn.functional as F
import torch

x_length = 24
y_length = 24

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
        self.tcn = TemporalConvNet(x_length, [256, 128, 64, 32])
        self.fc = torch.nn.Linear(30 * 32, 512)
        self.fc2 = torch.nn.Linear(512, y_length * 6)
        self.layernorm = torch.nn.LayerNorm(x_length )
        self.layernorm2 = torch.nn.LayerNorm(30)
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = x.view(-1, 30, x_length)  # 将数据的形状变为 [batch_size, num_nodes, num_features]
        x = self.layernorm(x)
        x = x.permute(0, 2, 1)  # 将数据的形状变为 [batch_size, num_features, num_nodes]
        x = self.tcn(x)  # 通过TCN处理时间序列
        x = F.relu(x)
        x = self.layernorm2(x)
        x = torch.flatten(x, start_dim=1)  # 展开张量
        x = x.view(-1, 30 * 32)  # 将数据的形状变为 [batch_size, 64*30]
        x = self.fc(x)
        x = self.fc2(x)
        return x
