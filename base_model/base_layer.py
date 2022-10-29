import tensorflow as tf 
from tensorflow.keras.layers import Dense, Layer, Concatenate

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
        self.concat=Concatenate(axis=-2)

    # x.shape=(n_nodes, n_ef)
    # a.shape=(n_nodes, n_nodes)
    # e.shape=(edges, n_ef)
    def propagate(self, x, a, e=None, **kwargs):
        self.n_nodes = tf.shape(x)[-2]
        self.index_targets = a.indices[:, 1]  # Nodes receiving the message
        self.index_sources = a.indices[:, 0]  # Nodes sending the message (ie neighbors)

        print("Okay here I am.")
        print(x.shape)
        print(a.values.shape)
        print(e.shape)
        print(self.get_sources(x).shape)
        print(self.get_targets(x).shape)

        concatenate = self.concat(
            [self.get_sources(x), self.get_targets(x), e]
        )
        e=self.dense_e1(self.dense_e2(concatenate))+e

        # Message
        msg_kwargs = self.get_kwargs(x, a, e, self.msg_signature, kwargs)
        messages = self.message(x, **msg_kwargs)

        # Aggregate
        agg_kwargs = self.get_kwargs(x, a, e, self.agg_signature, kwargs)
        embeddings = self.aggregate(messages, **agg_kwargs)

        # Update
        upd_kwargs = self.get_kwargs(x, a, e, self.upd_signature, kwargs)
        output = self.update(embeddings, **upd_kwargs)

        return output

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
    
    def call(self, inputs): # how can we parametrize this?
        inputs = tf.expand_dims(inputs, axis = -1) # Expand dimension
        inputs = tf.concat([inputs - self.c[i] for i in range(self.dim)], axis = -1)
        
        return tf.exp((-self.eta * inputs))