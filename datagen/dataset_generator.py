import sys
from turtle import position
import numpy as np
import networkx as nx
import torch
import dgl
from pickle import dump, load
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

    @staticmethod
    def add_one_hots_dgl(G):
        # G.ndata['feat'] = torch.nn.functional.one_hot(torch.arange(0, len(G)))
        G.ndata['feat'] = torch.ones(G.num_nodes()).reshape(G.num_nodes(), -1)
        G.edata['feat'] = torch.ones(G.num_edges()).reshape(G.num_edges(), -1)
        return G

    @abstractmethod
    def generate(self, num_graphs, n):
        raise NotImplementedError

    def save_to_dgl(self):
        positive = load(open('../data/' + self.dataset_name, 'rb'))
        negative = load(open('../data/non_' + self.dataset_name, 'rb'))

        negative = map(self.add_one_hots_dgl, map(dgl.from_networkx, negative))
        positive = map(self.add_one_hots_dgl, map(dgl.from_networkx, positive))
        all_data = []

        for p, n in zip(positive, negative):
            all_data.append((p, torch.tensor([1])))
            all_data.append((n, torch.tensor([0])))
        
        # dump(all_data, open(f"../data/{self.dataset_name}.pkl", "wb"))
        torch.save(all_data, f"../data/{self.dataset_name}.pkl",)

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

class PlanarDatsetGenerator(DatasetGenerator):
    def generate(self, num_graphs, n):
        planar_graphs = []
        non_planar_graphs = []
        tries = 0
        while len(planar_graphs) != num_graphs or len(non_planar_graphs) != num_graphs:
            sys.stdout.write(f"\r try: {tries}")

            candidate = nx.erdos_renyi_graph(n, 0.15)
            is_planar = nx.is_planar(candidate)
            candidate = self.add_one_hots(candidate, n)

            if is_planar:
                planar_graphs.append(candidate)
            if not is_planar and len(non_planar_graphs) != num_graphs:
                non_planar_graphs.append(candidate)
            tries += 1

        self.positive = planar_graphs
        self.negative = non_planar_graphs

        return planar_graphs, non_planar_graphs

    def check_planarity_proportions(self):
        positive = load(open('../data/planarity', 'rb'))
        count = 0
        for graph in positive:
            n = len(graph.nodes)
            e = len(graph.edges)
            if (3*n - 6) < e:
                count += 1
        print("Ratio of positive graphs that violate e <= 3v-6: ", count/len(positive))
