import sys
import networkx as nx
import matplotlib.pyplot as plt
from argparse import ArgumentParser

def generate_planarity_er(num_graphs, n):
    planar_graphs = []
    non_planar_graphs = []
    tries = 0
    while len(planar_graphs) != num_graphs or len(non_planar_graphs) != num_graphs:
        sys.stdout.write(f"\r try: {tries}")

        candidate = nx.erdos_renyi_graph(n, 0.15)
        is_planar = nx.is_planar(candidate)
        if is_planar:
            planar_graphs.append(candidate)
        if not is_planar and len(non_planar_graphs) != num_graphs:
            non_planar_graphs.append(candidate)
        tries += 1
    return planar_graphs, non_planar_graphs

def main(args):
    planar_graphs, non_planar_graphs = generate_planarity_er(args.num_graphs, args.n)

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
    main(p.parse_args())