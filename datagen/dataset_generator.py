import sys
import numpy as np
import networkx as nx
import torch
from pickle import dump
from abc import abstractmethod

class DatasetGenerator:
    def __init__(self, dataset_name: str) -> None:
        self.dataset_name = dataset_name
        self.positive = []
        self.negative = []

    @staticmethod
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

    @abstractmethod
    def generate(self, num_graphs, n):
        raise NotImplementedError

    def save(self):
        dump([torch.tensor([[0, 1]], dtype=torch.float)]*len(self.positive), open("../data/" + self.dataset_name + '.labels', 'wb'))
        dump([torch.tensor([[1, 0]], dtype=torch.float)]*len(self.negative), open("../data/" + "non_" + self.dataset_name + '.labels', 'wb'))

        dump(self.positive, open("../data/" + self.dataset_name, 'wb'))
        dump(self.negative, open("../data/" + "non_" + self.dataset_name, 'wb'))

class K3ColorableDatasetGenerator(DatasetGenerator):
    def generate(self, num_graphs, n):
        k3_colorable_graphs = []
        non_k3_colorable_graphs = []
        tries = 0
        while len(k3_colorable_graphs) != num_graphs or len(non_k3_colorable_graphs) != num_graphs:
            sys.stdout.write(f"\r try: {tries}")

            candidate = nx.erdos_renyi_graph(n, 0.2)
            coloring = nx.coloring.greedy_color(candidate)
            is_k3_colorable = len(set(coloring.values())) <= 3

            candidate = self.add_one_hots(candidate, n)

            if is_k3_colorable:
                k3_colorable_graphs.append(candidate)

            if not is_k3_colorable and len(non_k3_colorable_graphs) != num_graphs:
                non_k3_colorable_graphs.append(candidate)

            tries += 1

        self.positive = k3_colorable_graphs
        self.negative = non_k3_colorable_graphs

        return k3_colorable_graphs, non_k3_colorable_graphs
