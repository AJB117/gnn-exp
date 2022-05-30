from pprint import pprint
import networkx as nx
import pickle
import matplotlib.pyplot as plt
import numpy as np
from random import sample, randint

def get_empty_edge(A):
    zeros = np.argwhere(A == 0)
    x = np.random.choice(zeros.shape[0], 1, replace=False)
    return zeros[x][0]

trees = []
non_trees = []
k = 0
for _ in range(200):
    G = nx.random_tree(20)
    trees.append(G)

    H = nx.random_tree(20)
    nodes = H.nodes
    num_edges = randint(1, 5)
    A = nx.adjacency_matrix(H)
    edges = [get_empty_edge(A) for _ in range(num_edges)]
    for (u, v) in edges:
        H.add_edge(u, v)
    if nx.is_tree(H):
        k += 1
    non_trees.append(H)

pickle.dump(trees, open('./trees.pkl', 'wb'))
pickle.dump(non_trees, open('./non_trees.pkl', 'wb'))

trees = pickle.load(open('./trees.pkl', 'rb'))
non_trees = pickle.load(open('./non_trees.pkl', 'rb'))
tree = trees[0]
non_tree = non_trees[0]

fig, axes = plt.subplots(2)
nx.draw(tree, with_labels=True, ax=axes[0])
nx.draw(non_tree, with_labels=True, ax=axes[1])
plt.show()