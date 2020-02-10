"""Implementation of EdgeConv using arbitrary functions as h"""
import tensorflow as tf
import tensorflow.keras.layers as lay
from tensorflow import keras


class SplitLayer(lay.Layer):
    """ Custom layer: split layer along specific axis.
    eg. split (1,9) into 9 x (1,1)

    Parameters
    ----------
    n_splits : int
        number of splits
    split_axis : int
        axis where to split tensor
    **kwargs : type
        Description of parameter `**kwargs`.

    Attributes
    ----------
    n_splits
    split_axis

    """

    def __init__(self, n_splits=12, split_axis=-1, **kwargs):
        self.n_splits = n_splits
        self.split_axis = split_axis
        super(SplitLayer, self).__init__(**kwargs)

    def get_config(self):
        config = {'n_splits': self.n_splits,
                  'split_axis': self.split_axis}
        base_config = super(SplitLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, x):
        ''' return array of splitted tensors '''
        sub_tensors = tf.split(x, self.n_splits, axis=self.split_axis)
        return sub_tensors

    def compute_output_shape(self, input_shape):
        sub_tensor_shape = list(input_shape)
        num_channels = sub_tensor_shape[-1]
        sub_tensor_shape[-1] = int(num_channels / self.n_splits)
        sub_tensor_shape = tuple(sub_tensor_shape)
        list_of_output_shape = [sub_tensor_shape] * self.n_splits
        return list_of_output_shape

    def compute_mask(self, inputs, mask=None):
        return self.n_splits * [None]


class EdgeConv(lay.Layer):
    '''
    Keras layer implementation of EdgeConv.
    # Arguments
        h_func: h-function applied on the points and it's k nearest neighbors. The function should take a list
            of two tensors. The first tensor is the vector v_i of the central point, the second tensor is the vector
            of one of its neighbors v_j.
            :param list: [v_i, v_j] with v_i and v_j being Keras tensors with shape (C_f, ).
            :return: Keras tensor of shape (C', ).
        next_neighbors: number k of nearest neighbors to consider
        agg_func: Aggregation function applied after h. Must take argument "axis=2" to
            aggregate over all neighbors.
    # Input shape
        List of two tensors [points, features] with shape:
        `[(batch, P, C_p), (batch, P, C_f)]`.
        or tensor with shape:
        `(batch, P, C)`
        if points (coordinates) and features are supposed to be the same.
    # Output shape
        Tensor with shape:
        `(batch, P, C_h)`
        with C_h being the output dimension of the h-function.
    '''

    def __init__(self, h_func, next_neighbors, agg_func=tf.reduce_mean, **kwargs):
        self.h_func = h_func
        self.K = next_neighbors
        self.agg_func = agg_func
        super(EdgeConv, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        try:
            p_shape, f_shape = input_shape
        except ValueError:
            f_shape = input_shape

        x = lay.Input((f_shape.as_list()[-1] * 2,))
        a = lay.Reshape((f_shape.as_list()[-1], 2))(x)
        x1, x2 = SplitLayer(n_splits=2, split_axis=-1)(a)  # (2, C)
        x1 = lay.Reshape((f_shape.as_list()[-1],))(x1)
        x2 = lay.Reshape((f_shape.as_list()[-1],))(x2)
        y = self.h_func([x1, x2])
        self.h_func = keras.models.Model(x, y)

        super(EdgeConv, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        try:
            points, features = x
        except TypeError:
            points = features = x

        # distance
        D = batch_distance_matrix_general(points, points)  # (N, P, P)
        _, indices = tf.nn.top_k(-D, k=self.K + 1)  # (N, P, K+1)
        indices = indices[:, :, 1:]  # (N, P, K) remove self connection

        knn_fts = knn(indices, features)  # (N, P, K, C)
        knn_fts_center = tf.tile(tf.expand_dims(features, axis=2), (1, 1, self.K, 1))  # (N, P, K, C)
        knn_fts = tf.concat([knn_fts_center, knn_fts], axis=-1)  # (N, P, K, 2*C)

        res = lay.TimeDistributed(lay.TimeDistributed(self.h_func))(knn_fts)  # (N, P, K, C')
        # aggregation
        agg = self.agg_func(res, axis=2)  # (N, P, C')
        return agg

    def compute_output_shape(self, input_shape):
        self.output_shape = self.h_func.get_output_shape_at(-1)
        return self.output_shape


def batch_distance_matrix_general(A, B):
    ''' Calculate elements-wise distance between entries in two tensors '''
    with tf.name_scope('dmat'):
        r_A = tf.reduce_sum(A * A, axis=2, keepdims=True)
        r_B = tf.reduce_sum(B * B, axis=2, keepdims=True)
        m = tf.matmul(A, tf.transpose(B, perm=(0, 2, 1)))
        D = r_A - 2 * m + tf.transpose(r_B, perm=(0, 2, 1))
        return D


def knn(topk_indices, features):
    # topk_indices: (N, P, K)
    # features: (N, P, C)
    # return: (N, P, K, C)
    with tf.name_scope('knn'):
        k = tf.shape(topk_indices)[-1]
        num_points = tf.shape(features)[-2]
        queries_shape = tf.shape(features)
        batch_size = queries_shape[0]
        batch_indices = tf.tile(tf.reshape(tf.range(batch_size), (-1, 1, 1, 1)), (1, num_points, k, 1))
        indices = tf.concat([batch_indices, tf.expand_dims(topk_indices, axis=3)], axis=3)  # (N, P, K, 2)
        return tf.gather_nd(features, indices)
