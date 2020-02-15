"""
This example implements the experiments on citation networks using convolutional
layers from the paper:
Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering (https://arxiv.org/abs/1606.09375)
Michaël Defferrard, Xavier Bresson, Pierre Vandergheynst
"""
import tensorflow
from tensorflow.keras import models, layers, optimizers
from matplotlib import pyplot as plt
import networkx as nx
from spektral.layers import ChebConv
from spektral.utils.convolution import chebyshev_filter
from data import loading
import utils
import numpy as np

# Load data
X_train, y_train, X_val, y_val, X_test, y_test, A, pos = loading.mnist_regular_graph(k=8)  # k=number of next neighbors

# plot example of single graph sample
fig, ax = plt.subplots(1, figsize=(9,9))
ax.axis('equal')

g = nx.from_numpy_matrix(A)
nx.draw(
    g,
    cmap=plt.get_cmap('jet'),
    pos=pos,
    node_size=100,
    node_color=X_train[5].squeeze(),  # set node color ~ to pixel intensity
    width=2,
    )

fig.tight_layout()
fig.savefig("./mnist_graph.png")

# plot first 20 eigenvectors of graph / most important eigenvectors in fourierdomain
fig = utils.plot_eigenvectors(A)
fig.savefig("./mnist_eigen.png")

# Parameters
channels = 16           # Number of channels in the first layer
cheb_k = 2              # Max degree of the Chebyshev approximation
support = cheb_k + 1    # Total number of filters (k + 1)
learning_rate = 1e-2
epochs = 2
batch_size = 64

N_samples = X_train.shape[0]  # Number of samples in the training dataset
test_samples = X_test.shape[0]
N_nodes = X_train.shape[-2]   # Number of nodes in the graph


# Preprocessing operations
fltr = chebyshev_filter(A, cheb_k) # normalaize adjacency matrix


# Model definition
X_in = layers.Input(shape=(N_nodes, 1))

# One input filter for each degree of the Chebyshev approximation
fltr_in = [layers.Input((N_nodes, N_nodes)) for _ in range(support)]

graph_conv_1 = ChebConv(channels,
                        activation='relu',
                        use_bias=False)([X_in] + fltr_in)
graph_conv_2 = ChebConv(2 * channels,
                        activation='softmax',
                        use_bias=False)([graph_conv_1] + fltr_in)
flatten = layers.Flatten()(graph_conv_2)
output = layers.Dense(10, activation='softmax')(flatten)

# Build model
model = models.Model(inputs=[X_in] + fltr_in, outputs=output)
optimizer = optimizers.Adam(lr=learning_rate)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              weighted_metrics=['acc'])
model.summary()


A_train = [np.repeat(f[np.newaxis,...], batch_size, axis=0) for f in fltr]  # create batch of adjacency matrices as ht graph is constant for all samples!

print('Train model.\n') # Train model
history = []
for epoch in range(epochs):

    print("epoch", epoch, "\n")
    progbar = tensorflow.keras.utils.Progbar(N_samples, verbose=1)  # create fancy keras progressbar
    for i in range(N_samples // batch_size):

        beg = i*batch_size
        end = (i+1) * batch_size
        loss, acc = model.train_on_batch([X_train[beg:end]] + A_train, y_train[beg:end])
        progbar.add(batch_size, values=[("train loss", loss), ("acc", acc)])  # update fancy keras progress bar

        history.append([loss, acc])


fig = utils.plot_history(history)
fig.savefig("./history_mnist.png")


print('Evaluating model.\n')  # Evaluate model

progbar = tensorflow.keras.utils.Progbar(test_samples, verbose=1)  # create fancy keras progressbar
for a in range(test_samples // batch_size):

    beg = a*batch_size
    end = (a+1) * batch_size
    eval_loss, eval_acc = model.test_on_batch([X_test[beg:end]] + A_train, y_test[beg:end])
    progbar.add(batch_size, values=[("eval loss", eval_loss), ("acc", eval_acc)])  # update fancy keras progress bar
