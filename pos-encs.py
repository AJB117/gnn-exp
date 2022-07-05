import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
from argparse import ArgumentParser


def sinusoidal_encoding(max_position, d_model, min_freq=1e-4):
    position = np.arange(max_position)
    freqs = min_freq**(2*(np.arange(d_model)//2)/d_model)
    pos_enc = position.reshape(-1,1)*freqs.reshape(1,-1)
    pos_enc[:, ::2] = np.cos(pos_enc[:, ::2])
    pos_enc[:, 1::2] = np.sin(pos_enc[:, 1::2])
    return pos_enc

def laplacian_encoding(max_position, d_model):
    position = np.arange(max_position)
    freq_term_1 = np.pi*np.arange(d_model)/max_position
    freq_term_2 = np.pi/(2*max_position)

    freqs = freq_term_1 - freq_term_2

    pos_enc = position.reshape(-1,1)*freqs.reshape(1,-1)
    eigvecs = np.cos(pos_enc)
    eigvals = [2*(1-np.cos(np.pi*k/max_position)) for k in position]
    return eigvecs, eigvals

def main(args):
    d_model = 128
    max_pos = 128
    if args.enc == "sine":
        mat = sinusoidal_encoding(max_pos, d_model) 
        map_type = "sinusoidal"
    elif args.enc == "lap": 
        mat, eigvals = laplacian_encoding(max_pos, d_model)
        map_type = "Laplacian eigenvectors"

    plt.figure()
    ax = sb.heatmap(mat, cmap='plasma')
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Position')
    ax.invert_yaxis()
    plt.show()

    if args.enc == "lap":
        idx = np.array(eigvals).argsort()
        mat = mat[:, idx]

    similarities = []
    for enc in mat:
        similarity = np.array([np.dot(enc, x) for x in mat])
        similarity /= max(similarity)
        # similarity[np.where(similarity == 1.0)] = 0.0
        similarities.append(similarity)

    # plt.pcolormesh(similarities, cmap='hot')
    # plt.colorbar()
    # # plt.title(f"Normalized encoding similarity (dot product) heat map ({map_type})")
    ax = sb.heatmap(similarities, cmap='hot')
    ax.set_xlabel("Encoding at word_i")
    ax.set_ylabel("Encoding at word_i")
    ax.invert_yaxis()
    # plt.xlabel("Encoding at word_i")
    # plt.ylabel("Encoding at word_i")
    plt.show()


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("--enc", choices=["sine", "lap"])
    main(p.parse_args())