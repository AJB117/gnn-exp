import networkx as nx
import pickle
import numpy as np
import torch
from argparse import ArgumentParser
from random import randint

def get_empty_edge(A):
    zeros = np.argwhere(A == 0)
    x = np.random.choice(zeros.shape[0], 1, replace=False)
    return zeros[x][0]

def embed_ohe(G, num_features):
    n = []
    for node in G.nodes:
        ohe = np.zeros(num_features, dtype=np.float32)
        ohe[node] = 1
        n.append((node, {
            'x': ohe
        }))
    H = nx.Graph()
    H.add_nodes_from(n)
    H.add_edges_from(G.edges)
    return H

def main(args):
    trees = []
    non_trees = []

    tree_num_nodes = args.n_tree_nodes
    no_tree_num_nodes = args.n_no_tree_nodes

    num_features = max(tree_num_nodes, no_tree_num_nodes)

    for _ in range(args.n_graphs):
        G = nx.random_tree(tree_num_nodes)
        G = embed_ohe(G, num_features)
        trees.append(G)

        H = nx.random_tree(no_tree_num_nodes)
        nodes = H.nodes
        num_edges = randint(1, 5)
        A = nx.adjacency_matrix(H)
        edges = [get_empty_edge(A) for _ in range(num_edges)]
        for (u, v) in edges:
            H.add_edge(u, v)
        H = embed_ohe(H, num_features)
        non_trees.append(H)

    pickle.dump([torch.tensor([[1, 0]], dtype=torch.float)]*len(trees), open('./trees_labels.pkl', 'wb'))
    pickle.dump([torch.tensor([[0, 1]], dtype=torch.float)]*len(non_trees), open('./non_trees_labels.pkl', 'wb'))

    pickle.dump(trees, open('./trees.pkl', 'wb'))
    pickle.dump(non_trees, open('./non_trees.pkl', 'wb'))

if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("--n_features", type=int, default=20)
    p.add_argument("--n_graphs", type=int, default=300)
    p.add_argument("--n_tree_nodes", type=int, default=20)
    p.add_argument("--n_no_tree_nodes", type=int, default=20)
    main(p.parse_args())