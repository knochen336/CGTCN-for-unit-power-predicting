import numpy as np
from torch.nn.utils import weight_norm
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

x_length = 6
y_length = 6




class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc = torch.nn.Linear(30 * x_length, y_length * 64)
        self.fc2 = torch.nn.Linear(64 * x_length, y_length * 32)
        self.fc3 = torch.nn.Linear(32 * x_length, y_length *16)
        self.fc4 = torch.nn.Linear(16 * x_length, y_length * 6)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight

        x = torch.flatten(x, start_dim=1)  # 展开张量
        x = x.view(-1, 30 * x_length)  # 将数据的形状变为 [batch_size, 64*30]
        x = self.fc(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        return x

