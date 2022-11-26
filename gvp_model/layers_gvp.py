from tensorflow.keras.layers import Dense, Layer
from tensorflow.math import sigmoid, unsorted_segment_mean
from tensorflow.nn import relu
from tensorflow import norm, concat, zeros, expand_dims, constant_initializer, exp, shape, gather
from tensorflow.linalg import matmul, diag

from numpy import arange

from spektral.layers.convolutional.message_passing import MessagePassing
from spektral.layers.pooling.global_pool import GlobalAvgPool

class GlobalUpdate(Layer):
    def __init__(self, vi):
        super().__init__()
        self.pool=GlobalAvgPool_modified()
        self.dense=Dense(16)        
        self.gvp1=GVPg(vi=vi+3, vo=int((vi+3)/2), so=10)
        self.gvp2=GVPg(vi=int((vi+3)/2), vo=3, so=3)

    def call(self, inputs):
        x_s, x_v, i, u_s, u_v = inputs

        avg_s = self.dense(self.pool([x_s, i]))
        avg_v = self.pool([x_v, i])

        update_s = concat(
            [avg_s, u_s], axis = -1
        )
        update_v = concat(
            [avg_v, u_v], axis = -1
        )
        output_s, output_v = self.gvp2(*self.gvp1(update_s, update_v))

        return output_s, output_v

class GlobalAvgPool_modified(GlobalAvgPool):
    def build(self, input_shape):
        self.data_mode = "disjoint"
        super().build(input_shape)

class MPNN(MessagePassing):
    def __init__(self, vi, vo, embedding=256, **kwargs):
        super().__init__()
        self.embeding=embedding
        self.m1 = GVPg(vi=2*vi + 1, vo=vi + vo, so=768)
        self.m2 = GVPg(vi=vi + vo, vo=vo + int(vi/2), so=512)
        self.m3 = GVPg(vi=vo + int(vi/2), vo=vo, so=256)

    def call(self, inputs, **kwargs):
        x_s, x_v, a, e_s, e_v = inputs
        return self.propagate(x_s, x_v, a, e_s, e_v)

    def propagate(self, x_s, x_v, a, e_s, e_v, **kwargs):
        self.n_nodes = shape(x_s)[-2]
        self.index_targets = a.indices[:, 1]  # Nodes receiving the message
        self.index_sources = a.indices[:, 0]  # Nodes sending the message (ie neighbors)

        # Message
        msg_kwargs = self.get_kwargs(x_s, a, e_s, self.msg_signature, kwargs)
        messages_s, messages_v = self.message(x_s, x_v, e_s, e_v, **msg_kwargs)

        # Aggregate
        agg_kwargs = self.get_kwargs(x_s, a, e_s, self.agg_signature, kwargs)
        embeddings_s, embeddings_v = self.aggregate(messages_s, messages_v, **agg_kwargs)

        # Update
        upd_kwargs = self.get_kwargs(x_s, a, e_s, self.upd_signature, kwargs)
        output_s, output_v = self.update(embeddings_s, embeddings_v, **upd_kwargs)

        return output_s, output_v

    # x_s.shape = [n_nodes, 256]
    # x_v.shape = [n_nodes, 3, 12]
    # e.shape = [n_edges, 10]
    def message(self, x_s, x_v, e_s, e_v):
        x_s = concat(
            [self.get_sources(x_s), self.get_targets(x_s), e_s],
            axis = -1
        )
        
        e_v = expand_dims(e_v, axis = -1)
        x_v = concat(
            [self.get_sources_v(x_v), self.get_targets_v(x_v), e_v],
            axis = -1
        )

        return self.m3(*self.m2(*self.m1(x_s, x_v)))

    def get_sources_v(self, x):
        return gather(x, self.index_sources, axis=-3)
    
    def get_targets_v(self, x):
        return gather(x, self.index_targets, axis=-3)

    def aggregate(self, messages_s, messages_v, **agg_kwargs):
        out_s = unsorted_segment_mean(messages_s, self.index_targets, self.n_nodes)
        out_v = unsorted_segment_mean(messages_v, self.index_targets, self.n_nodes)
        return out_s, out_v
    
    def update(self, embeddings_s, embeddings_v, **kwargs):
        return embeddings_s, embeddings_v


class GVPg(Layer):
    def __init__(self, vi, vo, so, v_act = sigmoid, s_act = relu, g_act = sigmoid):
        super().__init__()
        self.h = max(vi, vo)
        self.Wh = Dense(self.h, use_bias = False)
        self.Wvo = Dense(vo, use_bias = False)
        self.Wso = Dense(so, activation = s_act)
        self.Wg = Dense(vo, activation = g_act)
        self.v_act = v_act
    
    # V.shape = [..., 3, vi]
    # s.shape = [..., si]
    def call(self, s, V):
        Vh = self.Wh(V) # Vh.shape = [..., 3, h]
        Vvo = self.Wvo(Vh) # Vvo.shape = [..., 3, vo]
        sh = norm(Vh, axis = -2) # sh.shape = [..., h]
        sn = concat([sh, s], axis = -1) # sn.shape = [..., si+h]
        s_out = self.Wso(sn) # s_out.shape = [..., so]

        v_scalars = self.v_act(s_out) # v_scalars.shape = [..., so]
        gate = self.Wg(v_scalars) # gate.shape = [..., vo]
        V_out = self.column_wise_mult(gate, Vvo) # V_out.shape = [..., 3, vo]

        return s_out, V_out
    
    def column_wise_mult(self, scalars, matrix):
        diagonals = diag(scalars)
        return matmul(matrix, diagonals)

class RBFExpansion(Layer):
    def __init__(self, new_dimensions):
        super().__init__()
        self.dim = new_dimensions
        self.eta = self.add_weight(
            shape = (1), # A single number
            initializer = constant_initializer(7.0), # Initialized to 7
            trainable = True
        )

        self.c = self.add_weight(
            shape = (1, self.dim),
            initializer = constant_initializer(arange(0.0, 7.0, 7/self.dim)),
            trainable = True
        )
    
    def call(self, inputs):
        inputs = expand_dims(inputs, axis = -1) # Expand dimension
        gaps = inputs-self.c
        
        return exp((-self.eta * gaps ** 2))