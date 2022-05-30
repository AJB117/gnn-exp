import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.data
import dgl.function as fn
from dgl.nn import GraphConv

dataset = dgl.data.CoraGraphDataset()
graph = dataset[0]
print(graph.edata)

class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, num_classes)

    def forward(self, g, in_feats):
        h = self.conv1(g, in_feats)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h

class SAGEConv(nn.Module):
    def __init__(self, in_feat, out_feat) -> None:
        super(SAGEConv, self).__init__()
        self.linear = nn.Linear(in_feat*2, out_feat)

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.update_all(message_func=fn.copy_u('h', 'm'), reduce_func=fn.mean('m', 'h_N'))
            h_N = g.ndata['h_N']
            h_total = torch.cat([h, h_N], dim=1)
            return self.linear(h_total)

class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes) -> None:
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats)
        self.conv2 = SAGEConv(h_feats, num_classes)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h

def train(g, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    best_val_acc = 0
    best_test_acc = 0

    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']

    for e in range(100):
        logits = model(g, features)
        pred = logits.argmax(1)
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])

        train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
        val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
        test_acc = (pred[test_mask] == labels[test_mask]).float().mean()

        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if e % 5 == 0:
            print('In epoch {}, loss: {:.3f}, val acc: {:.3f} (best {:.3f}), test acc: {:.3f} (best {:.3f})'.format(
                e, loss, val_acc, best_val_acc, test_acc, best_test_acc))

graph = graph.to('cuda')
model = GCN(graph.ndata['feat'].shape[1], 16, dataset.num_classes).to('cuda')
train(graph, model)
