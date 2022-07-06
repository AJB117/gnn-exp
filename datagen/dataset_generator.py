import sys
import numpy as np
import networkx as nx
import torch
import dgl
from pickle import dump, load
from abc import abstractmethod
from random import randint
import matplotlib.pylab as plt 

class DatasetGenerator:
    def __init__(self, prefix: str) -> None:
        self.prefix = prefix
        try:
            self.positive = load(open(f'../data/{prefix}', 'rb'))
        except FileNotFoundError:
            self.positive = []
        try:
            self.negative = load(open(f'../data/non_{prefix}', 'rb'))
        except FileNotFoundError:
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

    @abstractmethod
    def check_proportions(self):
        raise NotImplementedError

    def save_to_dgl(self):
        negative = map(self.add_one_hots_dgl, map(dgl.from_networkx, self.negative))
        positive = map(self.add_one_hots_dgl, map(dgl.from_networkx, self.positive))
        all_data = []

        for p, n in zip(positive, negative):
            all_data.append((p, torch.tensor([1])))
            all_data.append((n, torch.tensor([0])))
        
        # dump(all_data, open(f"../data/{self.dataset_name}.pkl", "wb"))
        torch.save(all_data, f"../data/{self.prefix}.pkl",)

    def save(self):
        dump([torch.tensor([[0, 1]], dtype=torch.float)]*len(self.positive), open("../data/" + self.prefix + '.labels', 'wb'))
        dump([torch.tensor([[1, 0]], dtype=torch.float)]*len(self.negative), open("../data/" + "non_" + self.prefix + '.labels', 'wb'))

        dump(self.positive, open("../data/" + self.prefix, 'wb'))
        dump(self.negative, open("../data/" + "non_" + self.prefix, 'wb'))

class K3ColorableDatasetGenerator(DatasetGenerator):
    def generate(self, num_graphs, lower_n, upper_n):
        k3_colorable_graphs = []
        non_k3_colorable_graphs = []

        while len(k3_colorable_graphs) != num_graphs or len(non_k3_colorable_graphs) != num_graphs:
            sys.stdout.write(f"\r k3colorable - non_k3colorable: {len(k3_colorable_graphs)} - {len(non_k3_colorable_graphs)}")

            positive_found = False
            negative_found = False
            n = randint(lower_n, upper_n)

            while not negative_found or not positive_found:
                candidate = nx.erdos_renyi_graph(n, 0.2)
                coloring = nx.coloring.greedy_color(candidate)
                is_k3_colorable = len(set(coloring.values())) <= 3
                candidate = self.add_one_hots(candidate, n)

                if is_k3_colorable and len(k3_colorable_graphs) != num_graphs and not positive_found:
                    k3_colorable_graphs.append(candidate)
                    positive_found = True
                if not is_k3_colorable and len(non_k3_colorable_graphs) != num_graphs and not negative_found:
                    non_k3_colorable_graphs.append(candidate)
                    negative_found = True

            # candidate = nx.erdos_renyi_graph(n, 0.2)
            # coloring = nx.coloring.greedy_color(candidate)
            # is_k3_colorable = len(set(coloring.values())) <= 3

            # candidate = self.add_one_hots(candidate, n)

            # if is_k3_colorable:
            #     k3_colorable_graphs.append(candidate)

            # if not is_k3_colorable and len(non_k3_colorable_graphs) != num_graphs:
            #     non_k3_colorable_graphs.append(candidate)

        self.positive = k3_colorable_graphs
        self.negative = non_k3_colorable_graphs

        return k3_colorable_graphs, non_k3_colorable_graphs

    def check_proportions(self):
        positive_sizes = [len(x.nodes) for x in self.positive]
        negative_sizes = [len(x.nodes) for x in self.negative]
        plt.figure("3-colorable graph node number dist")
        plt.hist(positive_sizes)
        plt.figure("Non-3-colorable graph node number dist")
        plt.hist(negative_sizes)
        plt.show()

class PlanarDatsetGenerator(DatasetGenerator):
    def generate(self, num_graphs, lower_n, upper_n):
        planar_graphs = []
        non_planar_graphs = []

        while len(planar_graphs) != num_graphs or len(non_planar_graphs) != num_graphs:
            sys.stdout.write(f"\r planar - non_planar: {len(planar_graphs)} - {len(non_planar_graphs)}")

            positive_found = False
            negative_found = False
            n = randint(lower_n, upper_n)

            while not negative_found or not positive_found:
                candidate = nx.erdos_renyi_graph(n, 0.1)
                is_planar = nx.is_planar(candidate)
                candidate = self.add_one_hots(candidate, n)

                if is_planar and len(planar_graphs) != num_graphs and not positive_found:
                    planar_graphs.append(candidate)
                    positive_found = True
                if not is_planar and len(non_planar_graphs) != num_graphs and not negative_found:
                    non_planar_graphs.append(candidate)
                    negative_found = True

        self.positive = planar_graphs
        self.negative = non_planar_graphs

        return planar_graphs, non_planar_graphs

    def check_proportions(self):
        count = 0
        for graph in self.positive:
            n = len(graph.nodes)
            e = len(graph.edges)
            if (3*n - 6) < e:
                count += 1
        
        positive_sizes = [len(x.nodes) for x in self.positive]
        negative_sizes = [len(x.nodes) for x in self.negative]
        plt.figure("Planar graph node number dist")
        plt.hist(positive_sizes)
        plt.figure("Non-planar graph node number dist")
        plt.hist(negative_sizes)
        print("Ratio of positive graphs that violate e <= 3v-6: ", count/len(self.positive))
        plt.show()