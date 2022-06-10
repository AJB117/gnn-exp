import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.models import GAT
from torch_geometric.nn import GINConv, GCNConv, global_add_pool
from torch import Tensor
from torch_geometric.typing import Adj

class GCN(nn.Module):
    def __init__(self, num_features: int, num_hidden: int, num_classes: int, num_layers: int) -> None:
        super().__init__()
        self.conv0 = GCNConv(num_features, num_hidden)
        self.convs = nn.ModuleList(
            GCNConv(num_hidden, num_hidden) for _ in range(num_layers-1)
        )
        self.ll = nn.Linear(num_hidden, num_classes)
        self.num_layers = num_layers

    def forward(self, x: Tensor, edge_idx: Adj) -> Tensor:
        # apply convolutions
        x = self.conv0(x, edge_idx).relu()
        for i in range(self.num_layers-1):
            x = self.convs[i](x, edge_idx)
            if i < self.num_layers - 2:
                x = x.relu()

        x = global_add_pool(x, batch=None, size=None)
        x = F.dropout(x, training=self.training, p=0.5)
        x = self.ll(x)
        return x

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

class GIN(nn.Module):
    def __init__(self, num_features: int, num_hidden: int, num_classes: int, num_layers: int) -> None:
        super(GIN, self).__init__()
        self.conv0 = GINConv(
            nn.Sequential(
                nn.Linear(num_features, num_hidden),
                nn.BatchNorm1d(num_hidden), nn.ReLU(),
                nn.Linear(num_hidden, num_hidden), nn.ReLU()
            )
        )
        self.convs = nn.ModuleList(
            GINConv(
                nn.Sequential(
                    nn.Linear(num_hidden, num_hidden),
                    nn.BatchNorm1d(num_hidden), nn.ReLU(),
                    nn.Linear(num_hidden, num_hidden), nn.ReLU()
                )
            ) for _ in range(num_layers-1)
        )

        self.ll = nn.Linear(num_hidden*num_layers, num_classes)
        self.num_layers = num_layers

    def forward(self, x: Tensor, edge_idx: Adj) -> Tensor:
        # compute node representations
        emb = self.conv0(x, edge_idx)
        embs = [self.convs[0](emb, edge_idx)]
        embs.extend([self.convs[i](embs[i-1], edge_idx) for i in range(1, self.num_layers-1)])

        # readout
        h_0 = global_add_pool(emb, batch=None)
        hs = [global_add_pool(embs[i], batch=None) for i in range(self.num_layers-1)]

        # cat the representations
        h = torch.cat((h_0, *hs), dim=1)

        h = F.dropout(h, p=0.5, training=self.training)
        h = self.ll(h)

        return h

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
