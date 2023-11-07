import numpy as np
import pandas as pd
from sklearn import preprocessing
from torch.utils.data import Dataset, DataLoader
from torch_geometric import nn
from torch_geometric.data import Data
import torch.nn.functional as F
from GAT import GAT
from LSTM import LSTM

import torch


edge_path = 'edge_test_4_A.csv'
node_path = 'node_test_4_A.csv'
edge_data = pd.read_csv(edge_path)
node_data = pd.read_csv(node_path)
node_mean = node_data.iloc[:, 2:].mean()
edge_mean = edge_data.iloc[:, 2:4].mean()
node_stds = node_data.iloc[:, 2:].std()
edge_stds = edge_data.iloc[:, 2:4].std()
node_data.iloc[:, 2:] = node_data.iloc[:, 2:].fillna(node_mean)
edge_data.iloc[:, 2:] = edge_data.iloc[:, 2:].fillna(edge_mean)
edge_outliers = (edge_data.iloc[:, 2:4] > edge_mean + 3 * edge_stds) | (
            edge_data.iloc[:, 2:4] < edge_mean - 3 * edge_stds)
node_outliers = (node_data.iloc[:, 2:] > node_mean + 3 * node_stds) | (
            node_data.iloc[:, 2:] < node_mean - 3 * node_stds)
edge_rows_to_drop = edge_outliers.any(axis=1)
node_rows_to_drop = node_outliers.any(axis=1)
node_data = node_data.drop(node_data[node_rows_to_drop].index)
edge_data = edge_data.drop(edge_data[edge_rows_to_drop].index)

# 按照日期分组
grouped_node_data = node_data.groupby('date_id')
# 对id进行编码
le = preprocessing.LabelEncoder()
node = []
for _, node_data in grouped_node_data:
    node_data.iloc[:, 0] = le.fit_transform(node_data.iloc[:, 0])
    node.append(node_data.iloc[:, [0] + list(range(2, 37))])
grouped_edge_data = edge_data.groupby('date_id')
edge = []
for _, edge_data in grouped_edge_data:
    edge_data.iloc[:, 0] = le.fit_transform(edge_data.iloc[:, 0])
    edge_data.iloc[:, 1] = le.fit_transform(edge_data.iloc[:, 1])
    edge.append(edge_data.iloc[:, 0:4])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
in_size=35
out_size=2
h_size=64
num_heads=8
edge_dim=2
model=GAT(in_feats=in_size,h_feats=h_size,out_feats=out_size,edge_dim=edge_dim)
model.to(device)
for i in range(len(edge)):
    sorted_node = node[i].sort_values(by='geohash_id')
    x = sorted_node.iloc[:, 1:].values
    node_num = len(sorted_node)
    edges = edge[i]
    # 删除多余的边
    edges = edges[(edges['geohash6_point1'] < node_num) & (edges['geohash6_point2'] < node_num)]
    # 边的头结点
    edge_index1 = edges[['geohash6_point1']].values.astype(int)
    # 边的尾结点
    edge_index2 = edges[['geohash6_point2']].values.astype(int)
    edge_attr = edges[['F_1', 'F_2']].values
    x = torch.FloatTensor(x).to(device)
    edge_index = np.array([edge_index1, edge_index2])
    edge_index = torch.LongTensor(edge_index).reshape(2,-1).to(device)
    edge_attr = torch.from_numpy(edge_attr).float().to(device)
    # print(x.shape)
    # print(edge_index.shape)
    # print(edge_attr.shape)
    node_emb=model(x,edge_index,edge_attr)
    print(node_emb)



class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


def create_sliding_windows(data,label, window_size):
    X,y=[],[]
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size,:])
        y.append(label[i + window_size])
        X = torch.FloatTensor(X)
        y = torch.FloatTensor(y)
    return X,y


def train_model(model, X_train, y_train, num_epochs):
        loss_fn = torch.nn.MSELoss(reduction='sum')
        optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
        for t in range(num_epochs):
            model.hidden = model.reset_hidden_state()
            y_pred = model(X_train)
            loss = loss_fn(y_pred.float().to(device), y_train)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            if t % 10 == 0:
                print(f'Epoch {t} train loss: {loss.item()}')

        return model.eval()


# #设置随机数种子
# torch.manual_seed(1234)
# #窗口大小
# window_size = 5
# batch_size = 3
# X,y=create_sliding_windows(data,widow_size)
# dataset=MyDataset(X,y)
# dataloader=DataLoader(dataset,batch_size=batch_size,shuffle=False)
#
# model = LSTM(batch_size=batch_size, input_size=35, hidden_size=32, num_layers=1, output_size=35)
# model.to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# criterion = nn.MSELoss()
#
# num_epochs = 100
# model.train()
# for epoch in range(num_epochs):
#     for data in seq:
#         inputs = data[0].to(device)
#         targets = data[1].to(device)
#         # 前向传播
#         outputs = model(inputs)
#
#         # 计算损失
#         loss = criterion(outputs, targets)
#
#         # 反向传播和优化
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#     if (epoch + 1) % 10 == 0:
#         print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
