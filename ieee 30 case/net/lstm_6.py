import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn import LSTM

x_length = 6
y_length = 6


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = torch.nn.Linear(64 * x_length, y_length * 32)
        self.fc2 = torch.nn.Linear(32 * x_length, y_length * 6)
        self.lstm = LSTM(30, 64, 3, batch_first=True)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        x = x.view(-1, x_length, 30)
        x, (h_n, c_n) = self.lstm(x)
        x = F.relu(x)
        x = torch.flatten(x, start_dim=1)  # 展开张量
        x = x.view(-1, 64 * x_length)  # 将数据的形状变为 [batch_size, 64*30]
        x = self.fc(x)
        x = self.fc2(x)
        return x
