from tensorflow.keras import Model
from tensorflow.math import reduce_max
from spektral.layers.pooling.global_pool import GlobalAvgPool
from layers_gvp import MPNN, GlobalUpdate, RBFExpansion
from tensorflow import zeros, ones, expand_dims, concat, constant, squeeze
from tensorflow.keras.layers import Embedding, Dense

import tensorflow_probability as tfp

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
        self.embedding=embedding

        self.mpnn=[
            MPNN(vi=3, vo=6),
            MPNN(vi=6, vo=9),
            MPNN(vi=9, vo=6),
            MPNN(vi=6, vo=3)
        ]
        self.glob_u=[
            GlobalUpdate(vi=6),
            GlobalUpdate(vi=9),
            GlobalUpdate(vi=6),
            GlobalUpdate(vi=3)
        ]

        self.rbf=RBFExpansion(rbf_dim)
        self.dense_e=Dense(embedding)
        self.dense_n=Dense(embedding)

    def call(self, inputs):
        x, a, e, i = inputs

        ##########
        # Preprocessing and initialization
        ##########
        n_nodes = x.shape[0]
        batch_size = max(i)+1

        # Edge features
        e_s = self.dense_e(self.rbf(e[:, 3]))
        e_v = e[:, 0:3]

        # Node features
        x_v = zeros([n_nodes, 3, 3]) # Each vertex gets 3 vectors of dim 3
        x_s = self.dense_n(self.element_embedding(x))
        x_s = squeeze(x_s, axis = 1)

        # Global features
        u_s = zeros([batch_size, 3])
        I = constant([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        I = expand_dims(I, 0)
        #u_v = concat([I for _ in range(batch_size)], axis = 0)
        u_v = zeros([batch_size, 3, 3])

        ##########
        # Message blocks
        ##########
        for mpnn, glob in zip(self.mpnn, self.glob_u):
            x_s, x_v = mpnn([x_s, x_v, a, e_s, e_v])
            u_s, u_v = glob([x_s, x_v, i, u_s, u_v])
            #u_v = tfp.math.gram_schmidt(u_v)

        u_s = expand_dims(u_s, axis = -1)
        return concat([u_s, u_v], axis = -1)