# This file creates a model as described in the paper:
# https://www.sciencedirect.com/science/article/pii/S2666389921002233

from tensorflow.keras import Model 
from tensorflow.keras.layers import Embedding, Dense
from tensorflow import squeeze

from base_layer import base, RBFExpansion
from spektral.layers.pooling.global_pool import GlobalAvgPool

class base_model(Model):
    def __init__(
        self,
        embedding=256,
        message_passing=4,
        num_elements=84,
        rbf_dim=10
    ):
        super().__init__()
        self.pool=GlobalAvgPool
        self.element_embedding=Embedding(num_elements, embedding, mask_zero=True)
        self.element_mean=Embedding(num_elements, 1, mask_zero=True)
        self.embedding=embedding

        self.gnn=[
            base(embedding)
            for _ in range(message_passing)
        ]
        self.rbf=RBFExpansion(rbf_dim)
        self.dense_e=Dense(embedding)
        self.dense_n=Dense(1)

    def call(self, inputs):
        x, a, e, i = inputs
        x=squeeze(x, axis=1)
        e=squeeze(e, axis=1)

        ##########
        # Preprocessing
        ##########
        e_mean=self.element_mean(x)
        x=self.element_embedding(x)
        e=self.dense_e(self.rbf(e))

        ##########
        # Message blocks
        ##########
        for layer in self.gnn:
            x=layer([x, a, e])+x
        
        ##########
        # Pooling
        ##########
        x = self.dense_n(x)+e_mean
        return self.pool(x, i)
