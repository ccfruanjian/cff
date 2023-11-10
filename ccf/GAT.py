import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
import torch.nn.functional as F

class GAT(torch.nn.Module):
    def __init__(self, in_feats, h_feats, out_feats,edge_dim ):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_feats, h_feats, heads=8, edge_dim=edge_dim, concat=False)

    def forward(self, x,edge_index,edge_attr):
        x = self.conv1(x, edge_index, edge_attr)

        return x
# 123