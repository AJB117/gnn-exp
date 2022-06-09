import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
import torch.nn.functional as F
from torch_geometric.nn.models import GIN, GAT, GCN

# class GCN(torch.nn.Module):
#     def __init__(self, num_features, num_hidden, num_classes) -> None:
#         super().__init__()
#         self.conv1 = pyg_nn.GCNConv(num_features, num_hidden)
#         self.conv2 = pyg_nn.GCNConv(num_hidden, num_hidden)
#         self.ll = nn.Linear(num_hidden, num_classes)

#     def forward(self, data):
#         x, edge_idx = data.x, data.edge_index
#         x = self.conv1(x, edge_idx)
#         x = x.relu()
#         x = self.conv2(x, edge_idx)
#         x = pyg_nn.global_add_pool(x, batch=None, size=None)
#         x = F.dropout(x, training=self.training, p=0.5)
#         x = self.ll(x)
#         return x

# class GAT(torch.nn.Module):
#     def __init__(self, num_features, num_hidden, num_classes, n_heads) -> None:
#         super().__init__()
#         self.n_heads = n_heads
#         self.conv1 = pyg_nn.GATConv(num_features, num_hidden, heads=self.n_heads, dropout=0.6)
#         self.conv2 = pyg_nn.GATConv(num_hidden*self.n_heads, num_hidden, concat=False)
#         self.ll = nn.Linear(num_hidden, num_classes)

#     def forward(self, data):
#         x, edge_idx = data.x, data.edge_index
#         x = self.conv1(x, edge_idx)
#         x = x.relu()
#         x = F.dropout(x, training=self.training, p=0.5)
#         x = self.conv2(x, edge_idx)
#         x = pyg_nn.global_add_pool(x, batch=None, size=None)
#         x = self.ll(x)
#         return x

class GIN_C(GIN):
    def __init__(self, **args):
        super(GIN_C, self).__init__(**args)
        self.ll = nn.Linear(self.hidden_channels, 2)

    def forward(self, x, edge_idx):
        x = super(GIN, self).forward(x, edge_idx)
        x = F.dropout(x, training=self.training, p=0.5)
        x = pyg_nn.global_add_pool(x, batch=None, size=None)
        x = self.ll(x)
        return x

class GAT_C(GAT):
    def __init__(self, **args):
        super(GAT_C, self).__init__(**args)
        self.ll = nn.Linear(self.hidden_channels, 2)

    def forward(self, x, edge_idx):
        x = super(GAT, self).forward(x, edge_idx)
        x = F.dropout(x, training=self.training, p=0.5)
        x = pyg_nn.global_add_pool(x, batch=None, size=None)
        x = self.ll(x)
        return x

class GCN_C(GCN):
    def __init__(self, **args):
        super(GCN_C, self).__init__(**args)
        self.ll = nn.Linear(self.hidden_channels, 2)

    def forward(self, x, edge_idx):
        x = super(GCN, self).forward(x, edge_idx)
        x = F.dropout(x, training=self.training, p=0.5)
        x = pyg_nn.global_add_pool(x, batch=None, size=None)
        x = self.ll(x)
        return x