import networkx as nx
import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch
from random import randint

def get_empty_edge(A):
    zeros = np.argwhere(A == 0)
    x = np.random.choice(zeros.shape[0], 1, replace=False)
    return zeros[x][0]

def embed_ohe(G):
    n = []
    for node in G.nodes:
        ohe = np.zeros(len(G.nodes), dtype=np.float32)
        ohe[node] = 1
        n.append((node, {
            'x': ohe
        }))
    H = nx.Graph()
    H.add_nodes_from(n)
    H.add_edges_from(G.edges)
    return H

trees = []
non_trees = []

for _ in range(300):
    G = nx.random_tree(20)
    G = embed_ohe(G)
    trees.append(G)

    H = nx.random_tree(20)
    nodes = H.nodes
    num_edges = randint(10, 15)
    A = nx.adjacency_matrix(H)
    edges = [get_empty_edge(A) for _ in range(num_edges)]
    for (u, v) in edges:
        H.add_edge(u, v)
    H = embed_ohe(H)
    non_trees.append(H)

pickle.dump([torch.tensor([[1, 0]], dtype=torch.float)]*len(trees), open('./trees_labels.pkl', 'wb'))
pickle.dump([torch.tensor([[0, 1]], dtype=torch.float)]*len(non_trees), open('./non_trees_labels.pkl', 'wb'))

pickle.dump(trees, open('./trees.pkl', 'wb'))
pickle.dump(non_trees, open('./non_trees.pkl', 'wb'))