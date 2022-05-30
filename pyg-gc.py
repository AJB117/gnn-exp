import pickle
import torch
import torch.nn.functional as F
from torch_geometric.utils.convert import from_networkx
from torch_geometric.nn import GCNConv
from argparse import ArgumentParser

# implement readout, export to file with other models
class GCN(torch.nn.Module):
    def __init__(self, num_features, num_hidden, num_classes) -> None:
        super().__init__()
        self.conv1 = GCNConv(num_features, num_hidden)
        self.conv2 = GCNConv(num_hidden, num_classes)

    def forward(self, data):
        x, edge_idx = data.x, data.edge_index
        x = self.conv1(x, edge_idx)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_idx)

        return F.sigmoid(x)

def split(data, split=(0.7, 0.15, 0.15)):
    n = len(data)
    t, v, te = split
    train = data[:int(n*t)]
    val = data[int(n*t):int(n*t)+int(n*v)]
    test = data[int(n*t)+int(n*v):]

    return train, val, test

def main(args):
    trees = pickle.load(open('./trees.pkl', 'rb'))
    non_trees = pickle.load(open('./non_trees.pkl', 'rb'))

    tree_data = [from_networkx(t) for t in trees]
    non_tree_data = [from_networkx(n) for n in non_trees]

    train_t, val_t, test_t = split(tree_data)
    train_n, val_n, test_n = split(non_tree_data)

    train = train_t + train_n
    val = val_t + val_n
    test = test_t + test_n

    num_hidden = args.hidden

    model = GCN(num_hidden, num_hidden, 2)

if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("--hidden", type=int)
    main(p.parse_args())