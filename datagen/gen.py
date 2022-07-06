import networkx as nx
import matplotlib.pyplot as plt
import torch
from argparse import ArgumentParser
from dataset_generator import K3ColorableDatasetGenerator, PlanarDatsetGenerator

def main(args):
    if args.prefix == "k3colorable":
        gen = K3ColorableDatasetGenerator(args.prefix)
    if args.prefix == "planarity":
        gen = PlanarDatsetGenerator(args.prefix)
    
    if args.view:
        gen.check_proportions()
        return
    
    if args.dglView:
        graphs = torch.load(open(f"../data/{args.prefix}.pkl", 'rb'))
        print(graphs[0])
        return

    positive, negative = gen.generate(args.num_graphs, args.lower_n, args.upper_n)

    gen.save()
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
    p.add_argument("--lower_n", help="lower bound for number of nodes to include in graph", default=20, type=int)
    p.add_argument("--upper_n", help="upper bound for number of nodes to include in graph", default=20, type=int)
    p.add_argument("--num_graphs", help="number of graphs to generate (planar and non-planar)", default=1000, type=int)
    p.add_argument("--verify", help="view the graphs after generation", action="store_true")
    p.add_argument("--prefix", default="k3colorable", choices=["k3colorable", "planarity"])
    p.add_argument("--view", action="store_true")
    p.add_argument("--dglView", action="store_true")
    main(p.parse_args())