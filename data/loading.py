import numpy as np
import scipy.sparse as sp
from tensorflow import keras


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def deflected_cosmic_rays():
    file = np.load("/net/scratch/deeplearning/cosmic_ray_sphere/dataset_HAP.npz")
    x_train, x_test = file['data'][:-10000], file['data'][-10000:]
    labels = keras.utils.to_categorical(file['label'], num_classes=2)
    y_train, y_test = labels[:-10000], labels[-10000:]

    return x_train, x_test, y_train, y_test


def karate_club(edge_path="./", label_path=None):
    """Load karate club dataset
    returns: features, adj, edges, labels
    """

    print('Loading karate club dataset...')

    edges = np.loadtxt(edge_path, dtype=np.int32) - 1  # 0-based indexing
    features = sp.eye(np.max(edges + 1), dtype=np.float32).tocsr()
    num_nodes = edges.max() + 1  # idx starts at 0
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(num_nodes, num_nodes), dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    if label_path is not None:
        idx_labels = np.loadtxt(label_path, dtype=np.int32)
        idx_labels = idx_labels[idx_labels[:, 0].argsort()]
        labels = encode_onehot(idx_labels[:, 1])
        print('Dataset has {} nodes, {} edges, {} features.'.format(adj.shape[0], edges.shape[0], features.shape[1]))
        return features.toarray(), adj.toarray(), edges, labels
    else:
        print('Dataset has {} nodes, {} edges, {} features.'.format(adj.shape[0], edges.shape[0], features.shape[1]))
        return features.toarray(), adj.toarray(), edges


def make_mnist_super_data(path=""):
    import torch
    features, edge_index, edge_slice, pos, labels = torch.load(path)
    features, edge_index, edge_slice, pos, labels = np.array(features), np.array(edge_index), np.array(edge_slice), np.array(pos), np.array(labels)

    edges = edge_index.swapaxes(0, 1)
    num_nodes = edges.max() + 1  # idx starts at 0
    n_samples = features.shape[0]
    adj = np.zeros((n_samples, num_nodes, num_nodes))
    for i in range(n_samples):
        edge = edges[edge_slice[i]: edge_slice[i+1], :]  # idx starts at 0
        a = sp.coo_matrix((np.ones(edge.shape[0]), (edge[:, 0], edge[:, 1])),
                          shape=(num_nodes, num_nodes), dtype=np.float32)
        adj[i] = (a + a.T.multiply(a.T > a) - a.multiply(a.T > a)).toarray()

    return features, adj, edges, pos, labels
