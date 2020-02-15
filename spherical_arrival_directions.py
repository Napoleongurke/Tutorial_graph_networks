#!/usr/bin/env pygpu
import numpy as np
from astrotools import skymap
from tensorflow import keras
from tensorflow.keras import layers
from edgeconv import EdgeConv
from data import loading
import utils

test_id = 0

x_train, x_test, y_train, y_test = loading.deflected_cosmic_rays()

# define coordinates for very first EdgeConv
train_points, test_points = x_train[..., :3], x_test[..., :3]

# Use normalized Energy as features for convolution
train_features, test_features = x_train[..., -1, np.newaxis], x_test[..., -1, np.newaxis]

train_input_data = [train_points, train_features]
test_input_data = [test_points, test_features]

# plot example
example_map = x_test[0]
skymap.eventmap(example_map[:, 0:3].T, c=example_map[:, 3], cblabel="Energy (normed)", opath="skymap_%i.png" % test_id)


# build kernel network
def kernel_nn(data, nodes=16):
    d1, d2 = data  # get xi ("central" pixel) and xj ("neighborhood" pixels)

    dif = layers.Subtract()([d1, d2])
    x = layers.Concatenate(axis=-1)([d1, dif])

    x = layers.Dense(nodes, use_bias=False, activation="relu")(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dense(nodes, use_bias=False, activation="relu")(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dense(nodes, use_bias=False, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    return x


# build complete model
points_input = layers.Input((500, 3))
feats_input = layers.Input((500, 1))

x = EdgeConv(lambda a: kernel_nn(a, nodes=8), next_neighbors=5)([points_input, feats_input])  # conv with fixed graph
x = layers.Activation("relu")(x)
x = EdgeConv(lambda a: kernel_nn(a, nodes=16), next_neighbors=8)([points_input, x])  # conv with fixed graph
x = layers.Activation("relu")(x)
y = EdgeConv(lambda a: kernel_nn(a, nodes=32), next_neighbors=16)([points_input, x])  # conv with fixed graph
x = layers.Activation("relu")(y)
x = layers.GlobalAveragePooling1D(data_format='channels_first', name="embedding")(x)
out = layers.Dense(2, name="classification", activation="softmax")(x)

model = keras.models.Model([points_input, feats_input], out)
model.summary()

model.compile(loss="binary_crossentropy",
              optimizer=keras.optimizers.Adam(3E-3, decay=1E-4),
              metrics=['acc'])

history = model.fit(train_input_data,
          y_train,
          batch_size=64,
          epochs=2)


fig = utils.plot_history(history.history)
fig.savefig("./history_sphere.png")


# Draw Graph in each EdgeConv layerser
fig = utils.get_edge_graph_in_model(test_input_data, model, sample_idx=test_id)
fig.savefig("./cr_sphere_dynamic.png")


# Draw contribution (class score) of each individual cosmic ray
fig = utils.draw_signal_contribution(model, test_input_data, test_id=test_id)
fig.savefig("./signal_contributions_of_cosmic_ray.png")
