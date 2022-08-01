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

def gen_data(n, num_states, graph=None):
    data = []
    for _ in range(n):
        if graph is None:
            graph = nx.cycle_graph(num_states)
        num_nodes = len(graph.nodes())
        laplacian = nx.normalized_laplacian_matrix(graph).toarray()
        _, eigenvectors = np.linalg.eig(laplacian)
        eigenvectors = np.real(eigenvectors)
        data.append({
            'num_nodes': num_nodes,
            'eigvecs': eigenvectors[:, :num_states],
        })

    pickle.dump(eigenvectors, open('eigenvectors.pkl', 'wb'))
    return data

def construct_data(num_states, graph=None):
    train, val, test = [], [], []
    train = gen_data(1000, num_states, graph)
    val = gen_data(1000, num_states, graph)
    test = gen_data(1000, num_states, graph)
    return train, val, test


def load_data():
    with open('./train.pkl', 'rb') as f:
        train = pickle.load(f)
    with open('./val.pkl', 'rb') as f:
        val = pickle.load(f)
    with open('./test.pkl', 'rb') as f:
        test = pickle.load(f)
    return train, val, test

def load_files(fnames):
    data = []
    for fname in fnames:
        with open(fname, 'rb') as f:
            data.append(pickle.load(f))
    return data

def write_files(data, fnames):
    for i, fname in enumerate(fnames):
        with open(fname, 'wb') as f:
            pickle.dump(data[i], f)


def main(args):
    model = AutomatonPELayer(args)
    device = 'cpu'

    if args.train:
        data_names = ['train', 'val', 'test']
        try:
            data = load_files(data_names)
        except FileNotFoundError:
            graph = nx.erdos_renyi_graph(20, 0.7) if args.rand_graph else None
            data = construct_data(args.num_states, graph)
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
                num_nodes, data = data['num_nodes'], data['eigvecs']
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
                torch.save(best_pe, 'best_eigvec_pe.pt')
                torch.save(pos_init, 'best_eigvec_pos_init.pt')
                torch.save(pos_transition, 'best_eigvec_pos_transition.pt')
                torch.save(model, 'best_eigvec_model.pt')

            print(f'epoch {epoch}, train loss: {train_loss}, val loss: {val_loss}, test loss: {test_loss}')

    if args.gnn:
        matrix = torch.load('mat.pt')
        _, eigvecs = torch.linalg.eig(matrix)
        eigvecs = np.real(eigvecs.to(device).numpy())
        best_pe = torch.load('pe.pt').detach().to(device).numpy()
    else:
        eigvecs = pickle.load(open('eigenvectors.pkl', 'rb'))
        best_pe = torch.load('best_eigvec_pe.pt').detach().numpy()

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
    main(parser.parse_args())