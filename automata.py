import torch
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sb
from argparse import ArgumentParser


def main(args):
    n_nodes = args.n_nodes
    n_states = args.n_states

    initial_weights = torch.Tensor([[1]]*n_states)
    # initial_weights = torch.rand((n_states, 1))
    initial_weights = torch.cat((initial_weights, torch.zeros(n_states, n_nodes-1)), dim=1)

    # initial_weights = torch.zeros(n_states, n_nodes)
    # initial_weights = torch.rand(n_states, n_nodes)
    # initial_weights = torch.ones(n_states, n_nodes)

    graph = nx.DiGraph()
    edges = []
    for i in range(n_nodes-1):
        edges.append((i, i+1))
    graph.add_edges_from(edges)

    # graph = nx.path_graph(n_nodes)
    # graph = nx.cycle_graph(n_nodes)
    # graph = nx.random_powerlaw_tree(n_nodes, tries=100)
    # graph = nx.erdos_renyi_graph(n_nodes, 0.1)

    # mu = torch.zeros((n_states, n_states))
    mu = torch.rand((n_states, n_states))
    # mu = torch.ones((n_states, n_states))
    # mu = torch.eye(n_states)
    # mu = torch.ones(n_states, n_states).fill_diagonal_(0)
    A = torch.from_numpy(nx.adjacency_matrix(graph).A).type(torch.float)

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
    weights = torch.stack(weights.split(n_states), dim=1).type(torch.double)
    print(f"weights for {n_nodes} nodes with {n_states} states: \n", weights)

    plt.figure("encoding")
    ax = sb.heatmap(weights, cmap='plasma')
    ax.set_xlabel('State idx')
    ax.set_ylabel('Position idx')
    ax.invert_yaxis()

    plt.figure("similarity")
    similarity = weights.T @ weights
    ax = sb.heatmap(similarity, cmap='hot')
    ax.set_xlabel("Enc at node i")
    ax.set_ylabel("Enc at node i")
    ax.invert_yaxis()

    plt.show()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--n_nodes", help="number of nodes", default=10, type=int)
    parser.add_argument("--n_states", help="number of states", default=3, type=int)
    main(parser.parse_args())