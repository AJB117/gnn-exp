import pickle
import torch
import torch.nn as nn
from torch_geometric.utils.convert import from_networkx
# from torch_geometric.nn.models import GIN, GAT, GCN
from sklearn.utils import shuffle
from argparse import ArgumentParser
from models import GCN_C, GIN_C, GAT_C, GCN

def split(data, split=(0.8, None)):
    n = len(data)
    t, v = split
    train = data[:int(n*t)]
    val = data[int(n*t):]

    return train, val

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

    train_t, val_t = split(tree_data)
    train_n, val_n = split(non_tree_data)

    train = to_device(train_t + train_n, device)
    val = to_device(val_t + val_n, device)

    num_hidden = args.hidden

    if args.model == "gcn":
        # model = GCN(train[0][0].x.shape[1], num_hidden, 2).to(device)
        model = GCN_C(in_channels=train[0][0].x.shape[1], hidden_channels=num_hidden, num_layers=2, out_channels=num_hidden).to(device)
    elif args.model == "gat":
        # model = GAT(train[0][0].x.shape[1], num_hidden, 2, 8).to(device)
        model = GAT_C(in_channels=train[0][0].x.shape[1], hidden_channels=num_hidden, num_layers=2, out_channels=num_hidden).to(device)
    elif args.model == "gin":
        model = GIN_C(in_channels=train[0][0].x.shape[1], hidden_channels=num_hidden, num_layers=2, out_channels=num_hidden).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        train = shuffle(train)
        train_loss = 0.
        for i, d in enumerate(train):
            optimizer.zero_grad()
            out = model(d[0].x, d[0].edge_index)
            # out = model(d[0])
            loss = loss_fn(out, d[1])
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

        model.eval()
        val_correct = 0
        for i, d in enumerate(val):
            # pred = model(d[0]).argmax(dim=1)
            pred = model(d[0].x, d[0].edge_index).argmax(dim=1)
            if pred == torch.where(d[1][0] > 0)[0].item():
                val_correct += 1

        print(f'Epoch {epoch} train_loss: {train_loss}, val_acc: {val_correct/len(val)}')

if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("--hidden", type=int, default=16)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--model", type=str, choices=["gcn", "gat", "gin"])
    main(p.parse_args())