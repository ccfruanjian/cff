import numpy as np

import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.nn import GATConv


class GAT(torch.nn.Module):
    def __init__(self, in_feats, out_feats, edge_dim, num_heads):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_feats, out_feats, heads=num_heads, edge_dim=edge_dim, concat=False)

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index, edge_attr)

        return x


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size, device):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_directions = 1
        self.batch_size = batch_size
        self.device = device
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        # self.fcs = [nn.Linear(self.hidden_size, self.output_size).to(device) for i in range(self.n_outputs)]
        self.fc1 = nn.Linear(self.hidden_size, self.output_size)
        self.fc2 = nn.Linear(self.hidden_size, self.output_size)
        self.fc3 = nn.Linear(self.hidden_size, self.output_size)
        self.fc4 = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input_seq):
        # print(input_seq.shape)
        batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]
        h_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(self.device)
        c_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(self.device)
        # print(input_seq.size())
        # input(batch_size, seq_len, input_size)
        # output(batch_size, seq_len, num_directions * hidden_size)
        output, _ = self.lstm(input_seq, (h_0, c_0))
        pred1, pred2, pred3, pred4 = self.fc1(output), self.fc2(output), self.fc3(output), self.fc4(output)
        pred1, pred2, pred3, pred4 = pred1[:, -1, :], pred2[:, -1, :], pred3[:, -1, :], pred4[:, -1, :]

        # pred = torch.cat([pred1, pred2], dim=0)
        pred = torch.stack([pred1, pred2, pred3, pred4], dim=0).reshape(batch_size, 4, -1)

        return pred


class GATLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, num_layers, window_size, device):
        super(GATLSTM, self).__init__()
        self.device = device
        self.window_size = window_size
        self.gat = torch_geometric.nn.GAT(input_size, hidden_size, num_layers=2)
        self.lstm = LSTM(input_size=hidden_size*1140, hidden_size=250, num_layers=num_layers, output_size=2280,
                         batch_size=1, device=device)
        # self.fc=nn.Linear(in_features=2,out_features=1)

    def forward(self, nodes, edges_index, edges_attr):
        data = []
        for i in range(len(nodes)):
            nodes_ = torch.tensor(nodes[i], dtype=torch.float).to(self.device)
            edges_index_ = torch.tensor(edges_index[i], dtype=torch.long).to(self.device)
            edges_attr_ = torch.tensor(edges_attr[i], dtype=torch.float).to(self.device)
            x = self.gat(nodes_, edges_index_)
            x=x.reshape((1, -1))
            data.append(x)
        data = torch.cat(data, dim=0)
        data=data.unsqueeze(0)
        # print(data.shape)torch.Size([1, 7, 72960])
        out=self.lstm(data)
        return out