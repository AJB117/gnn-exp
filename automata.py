import torch
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sb
from argparse import ArgumentParser


def main(args):
    n_nodes = args.n_nodes
    n_states = args.n_states

    # initial_weights = torch.Tensor([[0]]*n_states)
    # initial_weights[0][0] = 1.
    # initial_weights = torch.cat((initial_weights, torch.zeros(n_states, n_nodes-1)), dim=1)

    initial_weights = torch.rand(n_states, n_nodes)
    # initial_weights = torch.ones(n_states, n_nodes)

    path_graph = nx.path_graph(n_nodes)

    # mu = torch.zeros((n_states, n_states))
    # mu = torch.rand((n_states, n_states))
    mu = torch.ones((n_states, n_states))
    A = torch.from_numpy(nx.adjacency_matrix(path_graph).A).type(torch.float)

    # flatten in column-major order
    vec_init = initial_weights.transpose(1, 0).flatten()

    # solve Bx = c
    #   B = I-A^T \otimes \mu
    #   x = \alpha
    #   c = vec_init

    kron_prod = torch.kron(A.t().contiguous(), mu)
    B = torch.eye(kron_prod.shape[1]) - kron_prod
    c = vec_init

    weights = torch.linalg.solve(B, c)

    # undo vectorization
    weights = torch.stack(weights.split(n_states), dim=1)
    print(f"weights for {n_nodes} nodes with {n_states} states: \n", weights)

    plt.figure("encoding")
    ax = sb.heatmap(weights, cmap='plasma')
    ax.set_xlabel('State idx')
    ax.set_ylabel('Position idx')
    ax.invert_yaxis()

    plt.figure("similarity")
    ax = sb.heatmap(weights.T @ weights, cmap='hot')
    ax.set_xlabel("Enc at node i")
    ax.set_ylabel("Enc at node i")
    ax.invert_yaxis()

    plt.show()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--n_nodes", help="number of nodes", default=50, type=int)
    parser.add_argument("--n_states", help="number of states", default=20, type=int)
    main(parser.parse_args())