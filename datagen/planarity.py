import sys
import networkx as nx
import matplotlib.pyplot as plt
import torch
import numpy as np
from argparse import ArgumentParser
from pickle import dump

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

def generate_planarity_dataset(num_graphs, n):
    planar_graphs = []
    non_planar_graphs = []
    tries = 0
    while len(planar_graphs) != num_graphs or len(non_planar_graphs) != num_graphs:
        sys.stdout.write(f"\r try: {tries}")

        candidate = nx.erdos_renyi_graph(n, 0.15)
        is_planar = nx.is_planar(candidate)
        candidate = add_one_hots(candidate, n)
        if is_planar:
            planar_graphs.append(candidate)
        if not is_planar and len(non_planar_graphs) != num_graphs:
            non_planar_graphs.append(candidate)
        tries += 1
    return planar_graphs, non_planar_graphs

def main(args):
    planar_graphs, non_planar_graphs = generate_planarity_dataset(args.num_graphs, args.n)

    dump([torch.tensor([[0, 1]], dtype=torch.float)]*len(planar_graphs), open(args.filenames[0] + '.labels', 'wb'))
    dump([torch.tensor([[1, 0]], dtype=torch.float)]*len(non_planar_graphs), open(args.filenames[1] + '.labels', 'wb'))

    dump(planar_graphs, open(args.filenames[0], 'wb'))
    dump(non_planar_graphs, open(args.filenames[1], 'wb'))

    if not args.verify: return

    for i, (planar, nonplanar) in enumerate(zip(planar_graphs, non_planar_graphs)):
        if i % 10 != 0:
            continue
        plt.figure("planar")
        nx.draw(planar)
        plt.figure("non-planar")
        nx.draw(nonplanar)
        plt.show()

if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument("--n", help="number of nodes", default=20)
    p.add_argument("--num_graphs", help="number of graphs to generate (planar and non-planar)", default=1000)
    p.add_argument("--verify", help="view the graphs after generation", action="store_true")
    p.add_argument("--filenames", default=["../data/planar.pkl", "../data/non_planar.pkl"])
    main(p.parse_args())