import pandas as pd
import numpy as np
import torch


def process_ieee118_attr():
    df = pd.read_csv("./original data/all_busloads.csv", index_col=None)
    data = df.values
    reshaped_data = data.ravel(order='F')
    reshaped_data = np.array(reshaped_data)
    newdata = reshaped_data.reshape((366, 99, 288)).transpose((0, 2, 1))
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
    return expanded_x, newy, edgeindex, edgeweight

