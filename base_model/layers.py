import tensorflow as tf
from tensorflow.keras.layers import Dense, Layer

from spektral.layers.convolutional.message_passing import MessagePassing

class base(MessagePassing):
    def __init__(self, embedding=256):
        super().__init__()
        self.embedding=embedding
        self.dense_e1=Dense(embedding)
        self.dense_e2=Dense(embedding)
        self.dense_n1=Dense(embedding)
        self.dense_n2=Dense(embedding)
        self.dense_n3=Dense(embedding)

    # x.shape=(n_nodes, n_ef)
    # a.shape=(n_nodes, n_nodes)
    # e.shape=(edges, n_ef)
    def call(self, inputs):
        x, a, e = inputs
        concatenate = tf.concat(
            [self.get_sources(x), self.get_targets(x), e],
            axis = -1
        )
        e=self.dense_e1(self.dense_e2(concatenate))+e

        return self.propagate(x, a, e)

    def message(self, x, e):
        return self.get_sources(self.dense_n1(x))*e

    def update(self, embeddings, **kwargs):
        return self.dense_n2(self.dense_n3(embeddings)) # + original_x

class RBFExpansion(Layer):
    def __init__(self, new_dimensions):
        super().__init__()
        self.dim = new_dimensions
        self.eta = self.add_weight(
            shape = (1), # A single number
            initializer = tf.constant_initializer(7.0), # Initialized to 7
            trainable = True
        )
        self.c = []
        for i in range(self.dim):
            self.c.append(
                self.add_weight(
                    shape = (1, ),
                    initializer = tf.constant_initializer(0.7*i),
                    trainable = True
                )
            )
    
    def build(self, input_shape):
        self.batch_size = input_shape[0]
        self.num_edges = input_shape[1]
    
    # Input is (batch_size, num_edges), output is (batch_size, num_edges, dim)
    def call(self, inputs): # how can we parametrize this?
        inputs = tf.expand_dims(inputs, axis = -1) # Expand dimension
        inputs = tf.concat([inputs - self.c[i] for i in range(self.dim)], axis = -1)
        
        return tf.exp((-self.eta * inputs))