from torch_geometric.data import Dataset, DataLoader
class MyDataset(Dataset):
    def __init__(self, data_list):
        super(MyDataset, self).__init__(root=None, transform=None, pre_transform=None)
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]