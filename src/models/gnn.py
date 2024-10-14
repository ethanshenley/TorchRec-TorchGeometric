import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv

class GNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, gnn_type='GCN'):
        super().__init__()
        if gnn_type == 'GCN':
            self.conv = GCNConv(in_channels, out_channels)
        elif gnn_type == 'GraphSAGE':
            self.conv = SAGEConv(in_channels, out_channels)
        elif gnn_type == 'GAT':
            self.conv = GATConv(in_channels, out_channels)
        else:
            raise ValueError(f"Unsupported GNN type: {gnn_type}")

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)

class GNNStack(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, gnn_type='GCN'):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GNNLayer(in_channels, hidden_channels, gnn_type))
        for _ in range(num_layers - 2):
            self.convs.append(GNNLayer(hidden_channels, hidden_channels, gnn_type))
        self.convs.append(GNNLayer(hidden_channels, out_channels, gnn_type))

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)
        return self.convs[-1](x, edge_index)
