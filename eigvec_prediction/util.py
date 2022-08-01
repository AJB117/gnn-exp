import numpy as np
import networkx as nx
import pickle

def sinusoidal_encoding(seq_len, d_model, min_freq=1e-4):
    position = np.arange(seq_len)
    freqs = min_freq ** (2*(np.arange(d_model)//2)/d_model)
    pos_enc = position.reshape(-1, 1)*freqs.reshape(1, -1)
    pos_enc[:, ::2] = np.cos(pos_enc[:, ::2])
    pos_enc[:, 1::2] = np.sin(pos_enc[:, 1::2])
    return pos_enc

def gen_sine_data(n, num_states):
    data = []
    for _ in range(n):
        seq_len = num_states
        sinusoidal_matrix = sinusoidal_encoding(seq_len, num_states)
        data.append({
            'num_nodes': seq_len,
            'matrix': sinusoidal_matrix[:, :num_states],
        })

    pickle.dump(sinusoidal_matrix, open('sinusoidal_matrix.pkl', 'wb'))
    return data

def gen_graph_data(n, num_states, graph=None):
    data = []
    for _ in range(n):
        if graph is None:
            graph = nx.cycle_graph(num_states)
        num_nodes = len(graph.nodes())
        laplacian = nx.normalized_laplacian_matrix(graph).toarray()
        _, eigenvectors = np.linalg.eig(laplacian)
        eigenvectors = np.real(eigenvectors)
        data.append({
            'num_nodes': num_nodes,
            'matrix': eigenvectors[:, :num_states],
        })

    pickle.dump(eigenvectors, open('eigenvectors.pkl', 'wb'))
    return data

def construct_data(num_states, graph=None, data_type='graph'):
    train, val, test = [], [], []
    if data_type == 'graph':
        train = gen_graph_data(1000, num_states, graph)
        val = gen_graph_data(1000, num_states, graph)
        test = gen_graph_data(1000, num_states, graph)
    elif data_type == 'sine':
        train = gen_sine_data(1000, num_states)
        val = gen_sine_data(1000, num_states)
        test = gen_sine_data(1000, num_states)
    return train, val, test


def load_data():
    with open('./train.pkl', 'rb') as f:
        train = pickle.load(f)
    with open('./val.pkl', 'rb') as f:
        val = pickle.load(f)
    with open('./test.pkl', 'rb') as f:
        test = pickle.load(f)
    return train, val, test

def load_files(fnames):
    data = []
    for fname in fnames:
        with open(fname, 'rb') as f:
            data.append(pickle.load(f))
    return data

def write_files(data, fnames):
    for i, fname in enumerate(fnames):
        with open(fname, 'wb') as f:
            pickle.dump(data[i], f)
