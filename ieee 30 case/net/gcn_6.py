
import torch.nn.functional as F

import numpy as np

import torch


x_length = 6
y_length = 6


from torch_geometric.nn import GCNConv
from torch.nn.utils import weight_norm



class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(x_length, x_length * 2)
        self.conv2 = GCNConv(x_length * 2, x_length * 4)
        self.conv3 = GCNConv(x_length * 4, x_length * 2)
        self.fc = torch.nn.Linear(30 * x_length * 2, y_length * 20)
        self.fc2 = torch.nn.Linear(y_length * 20, y_length * 6)




    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = x.view(-1, 30, x_length * 2)  # 将数据的形状变为 [batch_size, num_nodes, num_features]
        x = torch.flatten(x, start_dim=1)  # 展开张量
        x = x.view(-1, 30 * x_length * 2)  # 将数据的形状变为 [batch_size, 64*30]
        x = self.fc(x)
        x = self.fc2(x)

        return x
