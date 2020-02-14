import numpy as np
import scipy.sparse as sp
from tensorflow import keras
from spektral.datasets import mnist


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def mnist_regular_graph(k=8):
    ''' load mnist dataset as graph. The graph is created using knn.
        You can change the number of neighbors per node by changing k.
        params:
            k: number of neighbors per node
        returns:
            X_train, y_train, X_val, y_val, X_test, y_test, A, node_positions
    '''
    X_train, y_train, X_val, y_val, X_test, y_test, A = mnist.load_data()
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    n_x, n_y = (28, 28)
    x = np.linspace(0, 10, n_x)
    y = np.linspace(0, 10, n_y)
    y = np.flip(y)
    xv, yv = np.meshgrid(x, y)
    pos = np.stack([xv.flatten(), yv.flatten()], axis=-1)
    return X_train, y_train, X_val, y_val, X_test, y_test, A.toarray(), pos


def deflected_cosmic_rays():
    ''' load dataset of deflected cosmic rays.
        returns:
            x_train, x_test, y_train, y_test
    '''
    file = np.load("/net/scratch/deeplearning/cosmic_ray_sphere/dataset_HAP.npz")
    x_train, x_test = file['data'][:-10000], file['data'][-10000:]
    labels = keras.utils.to_categorical(file['label'], num_classes=2)
    y_train, y_test = labels[:-10000], labels[-10000:]

    return x_train, x_test, y_train, y_test


def karate_club(edge_path="/net/scratch/JGlombitza/edges.txt", label_path="/net/scratch/JGlombitza/labels.txt"):
    """ load karate club dataset
        returns: features, A, edges, labels
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
        return features.toarray(), np.sign(adj.toarray()), edges


def make_mnist_super_data(path=""):
    ''' Create mnist mnist_superpixel dataset
        returns:
            features, A, edges, pos, labels
    '''
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
