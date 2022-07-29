import argparse
import pprint
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import torch

def sinusoidal_encoding(seq_len, d_model, min_freq=1e-4):
    position = np.arange(seq_len)
    freqs = min_freq ** (2*(np.arange(d_model)//2)/d_model)
    pos_enc = position.reshape(-1, 1)*freqs.reshape(1, -1)
    pos_enc[:, ::2] = np.cos(pos_enc[:, ::2])
    pos_enc[:, 1::2] = np.sin(pos_enc[:, 1::2])
    return pos_enc

def main(args):
    # file = open('./test.en.best_trans_fw.pkl', 'rb')
    # file = open('../pe.pt', 'rb')
    file = open('../best_eigvec_pe.pt', 'rb')
    # file = open('../best_eigvec_pos_transition.pt', 'rb')
    # file = open('../best_eigvec_pos_init.pt', 'rb')
    # fw = pickle.load(file).to('cpu')
    fw = torch.load(file).detach().to('cpu')
    pprint.pprint(fw)
    # fw = fw/np.linalg.norm(fw)
    seq_len, dim = fw.shape[1], fw.shape[0]

    if args.sine:
        # encs = np.transpose(sinusoidal_encoding(seq_len, dim))
        encs = sinusoidal_encoding(seq_len, dim)
        # encs = encs/np.linalg.norm(encs)
        plt.figure("sinusoidal encoding")
        # plt.plot(encs, label="sinusoidal encoding")
        ax = sb.heatmap(encs, cmap="RdBu_r")
        ax.invert_yaxis()
        plt.figure("sinusoidal similarities")
        ax = sb.heatmap(encs @ encs.T, cmap="hot")
        ax.invert_yaxis()

    if args.eigs:
        graph = nx.cycle_graph(seq_len)
        normalized_L = nx.normalized_laplacian_matrix(graph).A
        unnormalized_L = nx.laplacian_matrix(graph).A

        nL_eigvals, nL_eigvecs = np.linalg.eig(normalized_L)
        uL_eigvals, uL_eigvecs = np.linalg.eig(unnormalized_L)
        nL_eigvecs, uL_eigvecs = np.real(nL_eigvecs), np.real(uL_eigvecs)
        nL_eigvecs, uL_eigvecs = nL_eigvecs[:dim, :], uL_eigvecs[:dim, :]

        plt.figure("normalized eigenvectors")
        # plt.plot(nL_eigvecs, label='normalized')
        ax = sb.heatmap(nL_eigvecs, cmap="RdBu_r")
        ax.invert_yaxis()
        plt.figure("unnormalized eigenvectors")
        # plt.plot(uL_eigvecs, label='unnormalized')
        ax = sb.heatmap(uL_eigvecs, cmap="RdBu_r")
        ax.invert_yaxis()

    plt.figure("forward weights")
    # plt.plot(fw, label='forward')
    ax = sb.heatmap(fw, cmap="RdBu_r")
    # ax = sb.heatmap(fw, cmap="RdBu_r", vmin=-0.3, vmax=0.3)
    ax.invert_yaxis()
    plt.figure("forward weight similarities")
    ax = sb.heatmap((fw @ torch.transpose(fw, 1, 0)).numpy(), cmap="hot")
    ax.invert_yaxis()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--sine", action='store_true', help="plot sinusoidal encoding")
    parser.add_argument("--eigs", action='store_true', help="plot eigenvectors")
    main(parser.parse_args())