# Tensorflow model for prediction

from tensorflow.keras import Model
from tensorflow.math import reduce_max, unsorted_segment_mean, unsorted_segment_sum
from spektral.layers.pooling.global_pool import GlobalAvgPool
from layers_gvp import MPNN, GlobalUpdate, RBFExpansion
from tensorflow import zeros, ones, expand_dims, concat, constant, squeeze, shape, constant_initializer
from tensorflow.keras.layers import Embedding, Dense

class anisotropy(Model):
    def __init__(
        self,
        embedding=256,
        num_elements=84,
        rbf_dim=10
    ):
        super().__init__()
        self.pool=GlobalAvgPool()
        self.element_embedding=Embedding(num_elements, embedding, mask_zero=True)
        self.graph_embedding=Embedding(num_elements, 3, mask_zero=True)
        self.embedding=embedding

        self.mpnn=[ # vi is the number of vectors in each node
            MPNN(vi=1, vo=3),
            MPNN(vi=3, vo=3),
            MPNN(vi=3, vo=3),
            MPNN(vi=3, vo=3)
        ]
        self.glob_u=GlobalUpdate(vi=3)

        self.rbf=RBFExpansion(rbf_dim)
        self.dense_e=Dense(embedding)

    def call(self, inputs, training=False):
        x, a, e, i = inputs

        ##########
        # Preprocessing and initialization
        ##########
        n_nodes = shape(x)[0]
        batch_size = reduce_max(i)+1

        # Edge features
        e_s = self.dense_e(self.rbf(e[:, 3]))
        e_v = e[:, 0:3]

        # Node features
        x_v = expand_dims(unsorted_segment_mean(e_v, a.indices[:, 1], n_nodes), axis = -1)
        x_v = ones([n_nodes, 3, 3])
        x_s = self.element_embedding(x)
        x_s = squeeze(x_s, axis = 1)

        # Global features
        u_s = self.pool([self.graph_embedding(x), i])
        u_s = squeeze(u_s, axis = 1)
        x_e = unsorted_segment_sum(e_v, a.indices[:, 1], n_nodes) # sum of edges pointing to each vertex
        u_v = self.pool([expand_dims(x_e, axis = -1), i])

        ##########
        # Message blocks
        ##########
        for j, mpnn in enumerate(self.mpnn):
            if j != 0:
                old_s, old_v = x_s, x_v
            x_s, x_v = mpnn([x_s, x_v, a, e_s, e_v])
            if j != 0:
                x_s += old_s
                x_v += old_v

        out_s, out_v = self.glob_u([x_s, x_v, i, u_s, u_v])

        out_s = expand_dims(out_s, axis = -1)
        return concat([out_v, out_s], axis = -1)
