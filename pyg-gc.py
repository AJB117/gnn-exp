import pickle
from sklearn.utils import shuffle
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.utils.convert import from_networkx
import torch_geometric.nn as pyg_nn
from argparse import ArgumentParser

class GCN(torch.nn.Module):
    def __init__(self, num_features, num_hidden, num_classes) -> None:
        super().__init__()
        self.conv1 = pyg_nn.GCNConv(num_features, num_hidden)
        self.conv2 = pyg_nn.GCNConv(num_hidden, num_hidden)
        self.ll = nn.Linear(num_hidden, num_classes)

    def forward(self, data):
        x, edge_idx = data.x, data.edge_index
        x = self.conv1(x, edge_idx)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_idx)
        x = pyg_nn.global_mean_pool(x, batch=None, size=None)
        x = self.ll(x)
        return torch.sigmoid(x)

def split(data, split=(0.7, 0.15, 0.15)):
    n = len(data)
    t, v, te = split
    train = data[:int(n*t)]
    val = data[int(n*t):int(n*t)+int(n*v)]
    test = data[int(n*t)+int(n*v):]

    return train, val, test

def to_device(data, device):
    new = [(d[0].to(device), d[1].to(device)) for d in data]
    return new

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trees = pickle.load(open('./trees.pkl', 'rb'))
    non_trees = pickle.load(open('./non_trees.pkl', 'rb'))

    trees_labels = pickle.load(open('./trees_labels.pkl', 'rb'))
    non_trees_labels = pickle.load(open('./non_trees_labels.pkl', 'rb'))

    tree_data = [(from_networkx(t), trees_labels[i]) for i, t in enumerate(trees)]
    non_tree_data = [(from_networkx(n), non_trees_labels[i]) for i, n in enumerate(non_trees)]

    train_t, val_t, test_t = split(tree_data)
    train_n, val_n, test_n = split(non_tree_data)

    train = to_device(train_t + train_n, device)
    val = to_device(val_t + val_n, device)
    test = to_device(test_t + test_n, device)

    num_hidden = args.hidden
    model = GCN(train[0][0].x.shape[0], num_hidden, 2).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    loss_fn = nn.CrossEntropyLoss()

    for e, epoch in enumerate(range(200)):
        model.train()
        train = shuffle(train)
        train_loss = 0.
        for i, d in enumerate(train):
            optimizer.zero_grad()
            out = model(d[0])
            loss = loss_fn(out, d[1])
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

        model.eval()
        val_correct = 0
        for i, d in enumerate(val):
            pred = model(d[0])
            if pred.argmax().item() == torch.where(d[1][0] > 0)[0].item():
                val_correct += 1
        
        print(f'Epoch {e} train_loss: {train_loss}, val_acc: {val_correct/len(val)}')

if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("--hidden", type=int)
    p.add_argument("--batch_size", type=int)
    main(p.parse_args())