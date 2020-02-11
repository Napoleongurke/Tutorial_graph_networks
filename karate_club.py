import tensorflow as tf
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from data import loading
from spektral.layers import GraphConv
import utils

K = tf.keras.backend

features, adj, edges, labels_one_hot = loading.karate_club()


g = nx.from_numpy_matrix(adj)

# plot graph
fig, axes = plt.subplots(1)
nx.draw(
    g,
    cmap=plt.get_cmap('jet'),
    node_color=np.log(utils.one_hot_to_labels(labels_one_hot)),
    layout="random_layout")
fig.savefig("./initial_graph_spring.png")


adj = nx.adj_matrix(g).toarray()
X = np.identity(34)

# Pick one at random from each class
labels_to_keep = np.array([np.random.choice(np.nonzero(labels_one_hot[:, c])[0]) for c in range(4)])
train_mask = np.zeros(shape=labels_one_hot.shape[0], dtype=np.bool)
train_mask[labels_to_keep] = ~train_mask[labels_to_keep]
val_mask = ~train_mask

y_train = labels_one_hot * train_mask[..., np.newaxis]
y_val = labels_one_hot * val_mask[..., np.newaxis]

# Get important parameters of adjacency matrix
N = adj.shape[0]
F = 16
learning_rate = 0.01
epochs = 300

# Preprocessing operations
fltr = GraphConv.preprocess(adj).astype('f4')

# Model definition
X_in = Input(shape=(N,))
fltr_in = Input(shape=(N,))
x = GraphConv(F, activation='tanh', use_bias=False)([X_in, fltr_in])
x = GraphConv(F, activation='tanh', use_bias=False)([x, fltr_in])
x = GraphConv(2, activation='tanh', use_bias=False, name="embedding")([x, fltr_in])
output = GraphConv(4, activation='softmax', use_bias=False)([x, fltr_in])

# Build model
model = Model(inputs=[X_in, fltr_in], outputs=output)
model.compile(optimizer=Adam(lr=learning_rate),
              loss='categorical_crossentropy',
              weighted_metrics=['acc'])
model.summary()

embeddings = {}
for i in range(epochs):
    log = model.train_on_batch([X, fltr],
                               labels_one_hot,
                               sample_weight=train_mask,
                               )
    if i % 10 == 0:
        print("iteration:", i, "loss:", log[0], "accuracy:", log[1])

    if i % 50 == 0:
        val_log = model.test_on_batch([X, fltr], labels_one_hot, sample_weight=val_mask)
        print("iteration:", i, "val_loss:", val_log[0], "val_accuracy:", val_log[1])

        emb = model.get_layer("embedding").get_output_at(0)
        functor = K.function([X_in, fltr_in], emb)
        embeddings[i] = functor([X, fltr])


fig, axes = plt.subplots(nrows=2, ncols=3, sharey=True, sharex=True, figsize=(16, 10))
axes = axes.flatten()
keys = list(embeddings.keys())
keys.sort()

for i, k_epoch in enumerate(keys):
    axes[i].set_title("iterations %i" % k_epoch)
    nx.draw(
        g,
        cmap=plt.get_cmap('jet'),
        node_color=np.log(np.sum([(labels_one_hot[:, j] == 1) * (j + 1) for j in range(4)], axis=0)),
        pos=embeddings[k_epoch], ax=axes[i])
    axes[i].set_xlabel("embedding x")
    axes[i].set_ylabel("embedding y")

fig.savefig("./after_training.png")
