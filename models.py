import torch
import torch.nn.functional as F
import torch.nn as nn
import torch_geometric.nn as gnn
from torch import Tensor
from torch_geometric.typing import Adj
from typing import List

# https://arxiv.org/abs/1609.02907
class GCN(gnn.models.GCN):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args)
        self.ll = nn.Linear(self.hidden_channels, kwargs['num_classes'])
    
    def forward(self, x: Tensor, edge_idx: Adj) -> Tensor:
        x = super(GCN, self).forward(x, edge_idx)
        x = gnn.global_add_pool(x, batch=None, size=None)
        x = F.dropout(x, training=self.training, p=0.5)
        x = self.ll(x)
        return x

# https://arxiv.org/abs/1710.10903
class GAT(gnn.models.GAT):
    def __init__(self, *args, **kwargs) -> None:
        super(GAT, self).__init__(*args, heads=kwargs['heads'])

        self.final_attn = gnn.GATConv(self.hidden_channels, self.hidden_channels, 1)
        self.ll = nn.Linear(self.hidden_channels, kwargs['num_classes'])

    def forward(self, x: Tensor, edge_idx: Adj) -> Tensor:
        x = super(GAT, self).forward(x, edge_idx)
        x = F.elu(x)
        x = self.final_attn(x, edge_idx)
        x = gnn.global_add_pool(x, batch=None, size=None)
        x = F.dropout(x, training=self.training, p=0.5)
        x = self.ll(x)
        return x

# https://arxiv.org/abs/1810.00826
class GIN(gnn.models.basic_gnn.BasicGNN):
    def __init__(self, *args, **kwargs) -> None:
        super(GIN, self).__init__(*args)
        self.ll = nn.Linear(self.num_layers*self.hidden_channels, kwargs['num_classes'])

    def init_conv(self, in_channels: int, out_channels: int, **kwargs) -> gnn.conv.MessagePassing:
        return gnn.GINConv(
            nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.BatchNorm1d(out_channels), nn.ReLU(),
                nn.Linear(out_channels, out_channels), nn.ReLU()
            ), **kwargs
        )

    def forward(self, x: Tensor, edge_idx: Adj, *args, **kwargs) -> Tensor:
        # compute node representations
        embs: List[Tensor] = []
        for i, conv in enumerate(self.convs):
            if i == 0:
                embs.append(conv(x, edge_idx, *args, **kwargs))
            else:
                embs.append(conv(embs[i-1], edge_idx, *args, **kwargs))

        # readout
        hs = [gnn.global_add_pool(embs[i], batch=None) for i in range(self.num_layers)]

        # cat the representations
        h = torch.cat(hs, dim=1)

        h = F.dropout(h, p=0.5, training=self.training)
        h = self.ll(h)

        return h
