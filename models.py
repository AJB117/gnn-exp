"""
Wrappers around popular GNNs. Adjusted for graph classification.
"""
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch_geometric.nn as gnn
from torch import LongTensor, Tensor
from torch_geometric.typing import Adj
from typing import Callable, Optional

class GraphClassHead(torch.nn.Module):
    def __init__(self, hidden_channels: int, num_layers: int, num_classes: int, pooling_fn: Callable) -> None:
        super().__init__()
        self.lls = nn.ModuleList(
            nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels), nn.ReLU()
            )
            for _ in range(num_layers-1)
        )
        self.final_ll = nn.Linear(hidden_channels, num_classes)
        self.pooling_fn = pooling_fn

    def forward(self, x: Tensor, batch: Optional[LongTensor]=None) -> Tensor:
        x = self.pooling_fn(x, batch=batch)
        x = F.dropout(x, training=self.training, p=0.5)
        for ll in self.lls:
            x = ll(x)
        x = self.final_ll(x)
        return x

# https://arxiv.org/abs/1609.02907
class GCN(gnn.models.GCN):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args)
        self.class_head = GraphClassHead(self.hidden_channels, kwargs['num_lls'], kwargs['num_classes'], gnn.global_add_pool)

    def forward(self, x: Tensor, edge_idx: Adj, *args, **kwargs) -> Tensor:
        x = super(GCN, self).forward(x, edge_idx, *args, **kwargs)
        return self.class_head(x)

# https://arxiv.org/abs/1710.10903
class GAT(gnn.models.GAT):
    def __init__(self, *args, **kwargs) -> None:
        super(GAT, self).__init__(*args, heads=kwargs['heads'])

        self.final_attn = gnn.GATConv(self.hidden_channels, self.hidden_channels, 1)
        self.class_head = GraphClassHead(self.hidden_channels, kwargs['num_lls'], kwargs['num_classes'], gnn.global_add_pool)

    def forward(self, x: Tensor, edge_idx: Adj, *args, **kwargs) -> Tensor:
        x = super(GAT, self).forward(x, edge_idx, *args, **kwargs)
        x = F.elu(x)
        x = self.final_attn(x, edge_idx)
        return self.class_head(x)

# https://arxiv.org/pdf/1706.02216.pdf
class GraphSAGE(gnn.models.GraphSAGE):
    def __init__(self, *args, **kwargs) -> None:
        super(GraphSAGE, self).__init__(*args)
        self.class_head = GraphClassHead(self.hidden_channels, kwargs['num_lls'], kwargs['num_classes'], gnn.global_add_pool)
    
    def forward(self, x: Tensor, edge_idx: Adj, *args, **kwargs) -> Tensor:
        x = super(GraphSAGE, self).forward(x, edge_idx, *args, **kwargs)
        return self.class_head(x)

# https://arxiv.org/abs/1810.00826
class GIN(gnn.models.GIN):
    def __init__(self, *args, **kwargs) -> None:
        super(GIN, self).__init__(*args)
        self.class_head = GraphClassHead(self.hidden_channels, kwargs['num_lls'], kwargs['num_classes'], gnn.global_add_pool)

    def forward(self, x: Tensor, edge_idx: Adj) -> Tensor:
        x = super(GIN, self).forward(x, edge_idx)
        return self.class_head(x)
