import networkx as nx
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from dataset_generator import K3ColorableDatasetGenerator, PlanarDatsetGenerator

def main(args):
    if args.prefix == "k3colorable":
        gen = K3ColorableDatasetGenerator(args.prefix)
    if args.prefix == "planarity":
        gen = PlanarDatsetGenerator(args.prefix)

    positive, negative = gen.generate(args.num_graphs, args.n)

    gen.save_to_dgl()

    if not args.verify: return

    for i, (p, n) in enumerate(zip(positive, negative)):
        if i % 10 != 0:
            continue
        plt.figure(f"{args.prefix}")
        nx.draw(p)
        plt.figure(f"not-{args.prefix}")
        nx.draw(n)
        plt.show()

if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument("--n", help="number of nodes", default=20)
    p.add_argument("--num_graphs", help="number of graphs to generate (planar and non-planar)", default=1000)
    p.add_argument("--verify", help="view the graphs after generation", action="store_true")
    p.add_argument("--prefix", default="k3colorable")
    p.add_argument("--view", action="store_true")
    main(p.parse_args())