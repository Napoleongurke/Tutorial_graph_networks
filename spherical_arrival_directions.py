import numpy as np
from astrotools import skymap
from tensorflow import keras
from edgeconv import EdgeConv
from data import loading
import utils

K = keras.backend
lay = keras.layers

skymap_id = 0

x_train, x_test, y_train, y_test = loading.deflected_cosmic_rays()

# define coordinates for very first EdgeConv
train_points, test_points = x_train[..., :3], x_test[..., :3]

# Use normalized Energy as features for convolution
train_features, test_features = x_train[..., -1, np.newaxis], x_test[..., -1, np.newaxis]

train_input_data = [train_points, train_features]
test_input_data = [test_points, test_features]

# plot example
example_map = x_train[0]
skymap.eventmap(example_map[:, 0:3].T, c=example_map[:, 3], cblabel="Energy (normed)", opath="skymap_%i.png" % skymap_id)


# build kernel network
def kernel_nn(data, nodes=16):
    d1, d2 = data
    dif = lay.Subtract()([d1, d2])
    x = lay.Concatenate(axis=-1)([d1, dif])

    x = lay.Dense(nodes, use_bias=False, activation="relu")(x)
    x = lay.BatchNormalization()(x)

    x = lay.Dense(nodes, use_bias=False, activation="relu")(x)
    x = lay.BatchNormalization()(x)

    x = lay.Dense(nodes, use_bias=False, activation="relu")(x)
    x = lay.BatchNormalization()(x)
    return x


# build complete model
points_input = lay.Input((500, 3))
feats_input = lay.Input((500, 1))

x = EdgeConv(lambda a: kernel_nn(a, nodes=8), next_neighbors=5)([points_input, feats_input])
x = lay.Activation("elu")(x)
x = EdgeConv(lambda a: kernel_nn(a, nodes=16), next_neighbors=10)([points_input, x])
x = lay.Activation("elu")(x)
y = EdgeConv(lambda a: kernel_nn(a, nodes=32), next_neighbors=16)([points_input, x])
x = lay.Activation("elu")(y)
x = lay.GlobalAveragePooling1D(data_format='channels_first', name="embedding")(x)
out = lay.Dense(2, name="classification", activation="softmax")(x)

model = keras.models.Model([points_input, feats_input], out)
model.summary()

model.compile(loss="binary_crossentropy",
              optimizer=keras.optimizers.Adam(1E-3, decay=1E-4),
              metrics=['acc'])

model.fit(train_input_data,
          y_train,
          epochs=10)


# Draw Graph in each EdgeConv layer
fig = utils.get_edge_graph_in_model(test_input_data, model, sample_idx=0)
fig.savefig("./cr_sphere.png")


# Draw contribution (class score) of each individual cosmic ray
fig = utils.draw_signal_contribution(model, test_input_data, test_id=0)
fig.savefig("./signal_contributions_of_cosmic_ray.png")
