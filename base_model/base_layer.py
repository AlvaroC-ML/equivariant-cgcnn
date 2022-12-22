# This file creates a Spektral layer that does message passing in a crystal graph.

import tensorflow as tf 
import numpy as np
from tensorflow.keras.layers import Dense, Layer, Concatenate

from spektral.layers.convolutional.message_passing import MessagePassing

class base(MessagePassing):
    def __init__(self, embedding=256, aggregate="mean", **kwargs):
        super().__init__()
        self.embedding=embedding
        self.dense_e1=Dense(embedding)
        self.dense_e2=Dense(embedding)
        self.dense_n1=Dense(embedding)
        self.dense_n2=Dense(embedding)
        self.dense_n3=Dense(embedding)
        self.concat=Concatenate(axis=-1)

    def propagate(self, x, a, e=None, **kwargs):
        self.n_nodes = tf.shape(x)[-2]
        self.index_targets = a.indices[:, 1]  # Nodes receiving the message
        self.index_sources = a.indices[:, 0]  # Nodes sending the message

        concatenate = self.concat(
            [self.get_sources(x), self.get_targets(x), e]
        )
        e=self.dense_e1(self.dense_e2(concatenate))+e
        x=self.dense_n1(x)

        # Message
        msg_kwargs = self.get_kwargs(x, a, e, self.msg_signature, kwargs)
        messages = self.message(x, e, **msg_kwargs)

        # Aggregate
        agg_kwargs = self.get_kwargs(x, a, e, self.agg_signature, kwargs)
        embeddings = self.aggregate(messages, **agg_kwargs)

        # Update
        upd_kwargs = self.get_kwargs(x, a, e, self.upd_signature, kwargs)
        output = self.update(embeddings, **upd_kwargs)

        #return (output, e)
        return output

    def message(self, x, e):
        return self.get_sources(x)*e

    def update(self, embeddings, **kwargs):
        return self.dense_n2(self.dense_n3(embeddings))

class RBFExpansion(tf.keras.layers.Layer):
    def __init__(self, new_dimensions):
        super().__init__()
        self.dim = new_dimensions
        self.eta = self.add_weight(
            shape = (1),
            initializer = tf.constant_initializer(7.0),
            trainable = True
        )

        self.c = self.add_weight(
            shape = (1, self.dim),
            initializer = tf.constant_initializer(np.arange(0.0, 7.0, 7/self.dim)),
            trainable = True
        )
    
    def call(self, inputs):
        inputs = tf.expand_dims(inputs, axis = -1)
        gaps = inputs-self.c
        
        return tf.exp((-self.eta * gaps ** 2))
