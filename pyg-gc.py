import pickle
import torch
import torch.nn as nn
from torch_geometric.utils.convert import from_networkx
from sklearn.utils import shuffle
from argparse import ArgumentParser
from models import GCN, GIN, GAT, GraphSAGE

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
    num_features = train[0][0].x.shape[1]
    num_layers = args.layers
    model = args.model
    num_lls = args.lls

    if model == "gcn":
        model = GCN(num_features, num_hidden, num_layers, num_classes=2, num_lls=num_lls).to(device)
    elif model == "gat":
        model = GAT(num_features, num_hidden, num_layers, num_hidden, num_classes=2, heads=args.heads, num_lls=num_lls).to(device)
    elif model == "gin":
        model = GIN(num_features, num_hidden, num_layers, num_classes=2, num_lls=num_lls, jk="cat").to(device)
    elif model == "graphsage":
        model = GraphSAGE(num_features, num_hidden, num_layers, num_classes=2, num_lls=num_lls).to(device)

    trainable_params = [p.numel() for p in model.parameters() if p.requires_grad]
    print(f"number of trainable parameters: {sum(trainable_params)}")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        train = shuffle(train)
        train_loss = 0.
        for i, d in enumerate(train):
            optimizer.zero_grad()
            out = model(d[0].x, d[0].edge_index)
            loss = loss_fn(out, d[1])
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

        model.eval()
        val_correct = 0
        for i, d in enumerate(val):
            pred = model(d[0].x, d[0].edge_index).argmax(dim=1).item()
            label = d[1].argmax(dim=1).item()
            if i % 200 == 0:
                print(f"pred: {pred}, real: {label}")
            if pred == label:
                val_correct += 1

        print(f'Epoch {epoch} train_loss: {train_loss}, val_acc: {val_correct/len(val)} {val_correct}/{len(val)}')

if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("--hidden", type=int, default=16, help="Hidden unit size.")
    p.add_argument("--epochs", type=int, default=100, help="Number of epochs to train for.")
    p.add_argument("--model", type=str, choices=["gcn", "gat", "gin", "graphsage"])
    p.add_argument("--layers", type=int, default=3, help="# of graph convolutional layers to be used.")
    p.add_argument("--heads", type=int, default=8, help="# of attention heads for GAT.")
    p.add_argument("--lls", type=int, default=1, help="""# of linear layers in graph classification head.
                                                        Final layer has no activation, but the rest are separated by ReLUs.
                                                        Default=1.""")
    main(p.parse_args())