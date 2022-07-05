import torch
import networkx as nx
import numpy as np
from argparse import ArgumentParser
from sys import float_info


def main(args):
    n_nodes = args.n_nodes
    n_states = args.n_states

    initial_weights = torch.Tensor([[0]]*n_states)
    initial_weights[0][0] = 1.
    initial_weights = torch.cat((initial_weights, torch.zeros(n_states, n_nodes-1)), dim=1)

    path_graph = nx.path_graph(n_nodes)

    # mu = torch.eye(n_states)
    mu = torch.zeros((n_states, n_states))
    # mu = torch.rand((n_states, n_states))
    A = torch.from_numpy(nx.adjacency_matrix(path_graph).A).type(torch.float)

    # flatten in column-major order
    vec_init = initial_weights.transpose(1, 0).flatten()

    # solve Bx = c
    #   B = (I-A^T \otimes \mu)
    #   x = \alpha
    #   c = vec_init

    kron_prod = torch.kron(A.t().contiguous(), mu)
    B = torch.eye(kron_prod.shape[1]) - kron_prod
    c = vec_init

    _, info = torch.linalg.inv_ex(B)
    if info.item() != 0:
        print(f"Singular matrix, zero elements at: {info}")
        exit()

    weights = torch.linalg.solve(B, c)
    print(f"weights for {n_nodes} nodes with {n_states} states: \n", torch.stack(weights.split(n_states), dim=1))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--n_nodes", help="number of nodes", default=5, type=int)
    parser.add_argument("--n_states", help="number of states", default=3, type=int)
    main(parser.parse_args())