import numpy as np
from sklearn.neighbors import kneighbors_graph
import matplotlib.pyplot as plt
import networkx as nx
from tensorflow import keras
from astrotools import skymap

K = keras.backend


def one_hot_to_labels(labels_one_hot):
    return np.sum([(labels_one_hot[:, i] == 1) * (i + 1) for i in range(4)], axis=0)


def get_edge_graph_in_model(input_data, model, sample_idx=0):
    layer_names = [l.name for l in model.layers if "edge_conv" in l.name]
    coord_mask = [np.sum(np.linalg.norm(inp_d[sample_idx], axis=-1)) == 500 for inp_d in input_data]
    assert True in coord_mask, "For plotting the spherical graph of the cosmic ray on at least one input has to have 3 dimensions XYZ"
    fig, axes = plt.subplots(ncols=len(layer_names), figsize=(5 * len(layer_names), 6))
    for i, lay_name in enumerate(layer_names):
        points_in, feats_in = model.get_input_at(0)
        coordinates = model.get_layer(lay_name).get_input_at(0)
        functor = K.function([points_in, feats_in], coordinates)
        sample_input = [inp[np.newaxis, sample_idx] for inp in input_data]
        layer_points, layer_features = functor(sample_input)
        layer_points = np.squeeze(layer_points)
        adj = kneighbors_graph(layer_points, model.get_layer(lay_name).K)
        g = nx.from_scipy_sparse_matrix(adj)
        for c, s in zip(coord_mask, sample_input):
            if c == True:  # no 'is' here!
                pos = s
                break
        axes[i].set_title("XY-Projection of Graph formed in %s" % lay_name)

        nx.draw(
            g,
            cmap=plt.get_cmap('viridis'),
            pos=pos.squeeze()[:, :-1],
            node_size=10,
            width=0.3,
            ax=axes[i])
        axes[i].axis('equal')
    fig.tight_layout()
    return fig


def draw_signal_contribution(model, test_input_data, test_id=0):
    # Draw Signal Contribution
    coord_mask = [np.sum(np.linalg.norm(inp_d[test_id], axis=-1)) == 500 for inp_d in test_input_data]
    assert True in coord_mask, "For plotting the spherical graph of the cosmic ray on at least one input has to have 3 dimensions XYZ"

    emb = model.get_layer("embedding").get_output_at(0)
    functor = K.function([points_input, feats_input], emb)
    test_sample_points = [arr[np.newaxis, test_id] for arr in test_input_data]
    F = functor(test_sample_points).squeeze()  # output of last conv layer
    W, b = model.get_layer("classification").get_weights()  # weights of final dense layer
    score = F[..., np.newaxis] * W  # leave out bias for better visualization
    for c, s in zip(coord_mask, test_sample_points):
        if c == True:  # no 'is' here!
            pos = s
            break
    fig, ax = skymap.eventmap(pos.squeeze().T,
                              c=score[:, 1],
                              cblabel="class score",
                              cmap="YlOrRd")
    fig.tight_layout()
    return fig
