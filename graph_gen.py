import networkx as nx
import pickle
import numpy as np
import torch
from argparse import ArgumentParser
from random import randint, sample

def add_one_hots(G, num_features):
    n = []
    for node in G.nodes:
        one_hot = np.zeros(num_features, dtype=np.float32)
        one_hot[node] = 1
        n.append((node, {
            'x': one_hot
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
        G = add_one_hots(G, num_features)
        trees.append(G)

        H = nx.random_tree(no_tree_num_nodes)
        n_edges_to_add = randint(3, 3)
        edges = sample(list(nx.non_edges(H)), n_edges_to_add)
        for (u, v) in edges:
            H.add_edge(u, v)

        H = add_one_hots(H, num_features)
        non_trees.append(H)

    pickle.dump([torch.tensor([[0, 1]], dtype=torch.float)]*len(trees), open('./trees_labels.pkl', 'wb'))
    pickle.dump([torch.tensor([[1, 0]], dtype=torch.float)]*len(non_trees), open('./non_trees_labels.pkl', 'wb'))

    pickle.dump(trees, open('./trees.pkl', 'wb'))
    pickle.dump(non_trees, open('./non_trees.pkl', 'wb'))

if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("--n_features", type=int, default=20)
    p.add_argument("--n_graphs", type=int, default=1200)
    p.add_argument("--n_tree_nodes", type=int, default=20)
    p.add_argument("--n_no_tree_nodes", type=int, default=20)
    main(p.parse_args())