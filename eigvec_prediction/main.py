import argparse
import networkx as nx
import numpy as np
import pickle
import torch
import pickle
import seaborn as sb
import matplotlib.pyplot as plt

from tqdm import tqdm
from model import AutomatonPELayer
from util import load_files, construct_data, write_files


def main(args):
    model = AutomatonPELayer(args)
    device = 'cpu'

    if args.train:
        data_names = ['train', 'val', 'test']
        try:
            data = load_files(data_names)
        except FileNotFoundError:
            graph = nx.erdos_renyi_graph(20, 0.7) if args.rand_graph else None
            data = construct_data(args.num_states, graph, args.data_type)
            write_files(data, data_names)

        best_loss = torch.inf
        best_pe = None
        train, val, test = data

        criterion = torch.nn.MSELoss()
        for epoch in range(args.num_epochs):
            train_loss = 0.
            val_loss = 0.
            test_loss = 0.

            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

            for data in tqdm(train):
                optimizer.zero_grad()
                num_nodes, data = data['num_nodes'], data['matrix']
                pe, pos_init, pos_transition = model(num_nodes)
                loss = criterion(pe, torch.from_numpy(data).to(device).float())
                loss.backward()
                train_loss += loss.detach().item()
                optimizer.step()

            if train_loss < best_loss:
                model.eval()
                best_loss = train_loss
                best_pe = pe
                print('saving...')
                torch.save(best_pe, 'best_pe.pt')
                torch.save(pos_init, 'best_pos_init.pt')
                torch.save(pos_transition, 'best_pos_transition.pt')
                torch.save(model, 'best_model.pt')

            print(f'epoch {epoch}, train loss: {train_loss}, val loss: {val_loss}, test loss: {test_loss}')

    if args.gnn:
        matrix = torch.load('mat.pt')
        _, eigvecs = torch.linalg.eig(matrix)
        eigvecs = np.real(eigvecs.to(device).numpy())
        best_pe = torch.load('pe.pt').detach().to(device).numpy()
    else:
        if args.data_type == "graph":
            fname = 'eigenvectors.pkl'
        else:
            fname = 'sinusoidal_matrix.pkl'
        eigvecs = pickle.load(open(fname, 'rb'))
        best_pe = torch.load('best_pe.pt').detach().numpy()

    plt.figure("Actual eigenvectors")
    ax = sb.heatmap(eigvecs)
    ax.invert_yaxis()

    plt.figure("PE")
    ax = sb.heatmap(best_pe)
    ax.invert_yaxis()

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_states', type=int, default=20)
    parser.add_argument('--directed', action='store_false')
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--rand_graph', action='store_true', help="use random erdos-renyi graph, default is cycle graph")
    parser.add_argument('--gnn', action='store_true')
    parser.add_argument('--pe_type', type=str, default='learned', choices=['learned', 'random'])
    parser.add_argument('--num_initials', type=int, default=1, help="number of initial state vectors")
    parser.add_argument('--data_type', type=str, default='graph', choices=['graph', 'sine'])
    main(parser.parse_args())