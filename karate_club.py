import time
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
import spektral
from spektral.layers import GraphConv

g = nx.read_graphml('data/karate.graphml')


def plot_graph(g, name=""):
    f, axes = plt.subplots(1)
    nx.draw(
        g,
        cmap=plt.get_cmap('jet'),
        node_color=np.log(list(nx.get_node_attributes(g, 'membership').values())))
    f.savefig("inital_graph.png")


adj = nx.adj_matrix(g).toarray()
X = np.ones(shape=adj.shape[0])[..., np.newaxis]
X=features
# x = np.diagonal(spektral.utils.degree(adj))[..., np.newaxis]

# Semi-supervised
memberships = [m - 1 for m in nx.get_node_attributes(g, 'membership').values()]
nb_classes = len(set(memberships))
labels = keras.utils.to_categorical(memberships, nb_classes)

# X = np.array(memberships) # WARNING THIS IS STUPID _ ONLY FOR BUG FIXING

# Pick one at random from each class
labels_to_keep = np.array([np.random.choice(np.nonzero(labels[:, c])[0]) for c in range(nb_classes)])
train_mask = np.zeros(shape=labels.shape[0], dtype=np.bool)
train_mask[labels_to_keep] = ~train_mask[labels_to_keep]
val_mask = ~train_mask

y_train = labels * train_mask[..., np.newaxis]
y_val = labels * val_mask[..., np.newaxis]

# Get important parameters of adjacency matrix
N = adj.shape[0]
F = 16
learning_rate = 1e-2
epochs = 20000

# Preprocessing operations
fltr = GraphConv.preprocess(adj).astype('f4')

# Model definition
X_in = Input(shape=(N,))
fltr_in = Input(shape=(N,))
dropout_1 = Dropout(0.5)(X_in)
graph_conv_1 = GraphConv(4, activation='tanh', use_bias=False)([dropout_1, fltr_in])
dropout_2 = Dropout(0.5)(graph_conv_1)
graph_conv_4 = GraphConv(4, activation='tanh', use_bias=False)([dropout_2, fltr_in])
output = GraphConv(nb_classes, activation='softmax', use_bias=False)([graph_conv_4, fltr_in])

# Build model
model = Model(inputs=[X_in, fltr_in], outputs=output)
optimizer = Adam(lr=learning_rate)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              weighted_metrics=['acc'])
model.summary()

# Train model
# fltr = fltr.toarray()
validation_data = ([X, fltr], labels, val_mask)
model.fit([X, fltr],
          labels,
          epochs=epochs,
          sample_weight=train_mask,
          batch_size=N,
          validation_data=validation_data,
          shuffle=False,  # Shuffling data means shuffling the whole graph
          )

# Evaluate model
print('Evaluating model.')
eval_results = model.evaluate([X, fltr],
                              labels,
                              sample_weight=val_mask,
                              batch_size=N)
print('Done.\n'
      'Test loss: {}\n'
      'Test accuracy: {}'.format(*eval_results))


epochs = 600
save_every = 100

t = time.time()
outputs = {}
# Train model
for epoch in range(epochs):
    # Construct feed dictionary

    # Training step
    _, train_loss, train_acc = sess.run(
        (opt_op, loss, accuracy), feed_dict=feed_dict_train)

    if epoch % save_every == 0:
        # Validation
        val_loss, val_acc = sess.run((loss, accuracy), feed_dict=feed_dict_val)

        # Print results
        print("Epoch:", '%04d' % (epoch + 1),
              "train_loss=", "{:.5f}".format(train_loss),
              "train_acc=", "{:.5f}".format(train_acc),
              "val_loss=", "{:.5f}".format(val_loss),
              "val_acc=", "{:.5f}".format(val_acc),
              "time=", "{:.5f}".format(time.time() - t))

        feed_dict_output = {ph['adj_norm']: adj_norm_tuple,
                            ph['x']: feat_x_tuple}

        output = sess.run(o_fc3, feed_dict=feed_dict_output)
        outputs[epoch] = output

node_positions = {o: {n: tuple(outputs[o][j])
                      for j, n in enumerate(nx.nodes(g))}
                  for o in outputs}
plot_titles = {o: 'epoch {o}'.format(o=o) for o in outputs}

# Two subplots, unpack the axes array immediately
f, axes = plt.subplots(nrows=2, ncols=3, sharey=True, sharex=True, figsize=(16, 10))

e = list(node_positions.keys())

for i, ax in enumerate(axes.flat):
    pos = node_positions[e[i]]
    ax.set_title(plot_titles[e[i]])

    nx.draw(
        g,
        cmap=plt.get_cmap('jet'),
        node_color=np.log(
            list(nx.get_node_attributes(g, 'membership').values())),
        pos=pos, ax=ax)

f.savefig("karathe_graph_final.png")
