import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import networkx as nx

from argparse import ArgumentParser


def sinusoidal_encoding(max_position, d_model, min_freq=1e-4):
    position = np.arange(max_position)
    freqs = min_freq**(2*(np.arange(d_model)//2)/d_model)
    pos_enc = position.reshape(-1,1)*freqs.reshape(1,-1)
    pos_enc[:, ::2] = np.cos(pos_enc[:, ::2])
    pos_enc[:, 1::2] = np.sin(pos_enc[:, 1::2])
    return pos_enc


def lap_enc(max_position, d_model, mode="cycle"):
    graph = None
    if mode == "cycle":
        graph = nx.cycle_graph(max_position)
    elif mode == "path":
        graph = nx.path_graph(max_position)
    L = nx.normalized_laplacian_matrix(graph)
    evals, evecs = np.linalg.eig(L.toarray())
    idx = np.argsort(evals)
    evals, evecs = evals[idx], evecs[:, idx]
    evecs = evecs[:d_model, :]
    return evecs, evals


def main(args):
    d_model = args.d_model
    max_pos = args.max_pos

    if args.enc == "sine":
        mat = sinusoidal_encoding(max_pos, d_model) 
        map_type = "sinusoidal"
    elif args.enc == "lap-path":
        mat, evals = lap_enc(max_pos, d_model, mode="path")
        map_type = "Laplacian eigenvectors (path graph)"
    elif args.enc == "lap-cycle":
        mat, evals = lap_enc(max_pos, d_model, mode="cycle")
        map_type = "Laplacian eigenvectors (cycle graph)"

    plt.figure(map_type)
    ax = sb.heatmap(mat, cmap='plasma')
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Position')
    ax.invert_yaxis()
    plt.show()

    similarities = mat @ mat.transpose()

    ax = sb.heatmap(similarities, cmap='hot')
    ax.set_xlabel("Encoding at word_i")
    ax.set_ylabel("Encoding at word_i")
    ax.invert_yaxis()
    plt.show()


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("--enc", choices=["sine", "lap-path", "lap-cycle"], default="sine")
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--max_pos", type=int, default=128)
    main(p.parse_args())