import networkx as nx
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from dataset_generator import K3ColorableDatasetGenerator

def main(args):
    gen = K3ColorableDatasetGenerator(args.prefix)
    is_3_colorable, not_3_colorable = gen.generate(args.num_graphs, args.n)

    gen.save()

    if not args.verify: return

    for i, (colorable, not_colorable) in enumerate(zip(is_3_colorable, not_3_colorable)):
        if i % 10 != 0:
            continue
        plt.figure(f"{args.prefix}")
        nx.draw(colorable)
        plt.figure(f"not-{args.prefix}")
        nx.draw(not_colorable)
        plt.show()

if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument("--n", help="number of nodes", default=20)
    p.add_argument("--num_graphs", help="number of graphs to generate (planar and non-planar)", default=1000)
    p.add_argument("--verify", help="view the graphs after generation", action="store_true")
    p.add_argument("--prefix", default="3colorable")
    main(p.parse_args())