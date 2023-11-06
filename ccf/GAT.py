import torch
import torch.nn as nn
from torch_geometric.nn import GATConv


class GATE(nn.Module):
    def __init__(self, in_channels, out_channels, edge_weight_dim, num_heads=1):
        super(GATE, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_weight_dim = edge_weight_dim
        self.num_heads = num_heads

        # 定义GATConv层
        self.conv = GATConv(in_channels=in_channels, out_channels=out_channels, heads=num_heads)

        # 定义边权重线性变换层
        self.edge_weight_lin = nn.Linear(edge_weight_dim, num_heads)

    def forward(self, x, edge_index, edge_weight):
        # 将边权重线性变换为注意力系数
        attention = self.edge_weight_lin(edge_weight).softmax(dim=1)

        # 使用注意力系数进行节点表征计算
        x = self.conv(x, edge_index, attention)

        return x
