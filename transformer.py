import keras
import numpy as np
from keras.layers import Lambda, Layer, Concatenate, Dense, Add
import keras.backend as K
from keras import initializers
from keras.initializers import Ones, Zeros


class LayerNormalization(Layer):

    def __init__(self, eps=1e-6, **kwargs):
        self.eps = eps
        super(LayerNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:], initializer=Ones(), trainable=True)
        self.beta = self.add_weight(name='beta', shape=input_shape[-1:], initializer=Zeros(), trainable=True)
        super(LayerNormalization, self).build(input_shape)

    def call(self, x):
        mean = K.mean(x, axis=-1, keepdims=True)
        std = K.std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

    def compute_output_shape(self, input_shape):
        return input_shape


class SelfAttention(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(SelfAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create all trainable weight variable for this layer.
        self.W_Q = self.add_weight(name='query_transform_weight',
                                   shape=(input_shape[1], self.output_dim),
                                   dtype='float32',
                                   initializer='glorot_uniform',
                                   trainable=True)
        self.W_K = self.add_weight(name='key_transform_weight',
                                   shape=(input_shape[1], self.output_dim),
                                   dtype='float32',
                                   initializer='glorot_uniform',
                                   trainable=True)
        self.W_V = self.add_weight(name='value_transform_weight',
                                   shape=(input_shape[1], self.output_dim),
                                   dtype='float32',
                                   initializer='glorot_uniform',
                                   trainable=True)

        super(SelfAttention, self).build(input_shape)

    def call(self, X):
        Q = K.dot(X, self.W_Q)
        Key = K.dot(X, self.W_K)
        V = K.dot(X, self.W_V)
        scaled_weights_mat = K.dot(Q, K.transpose(Key))/np.sqrt(self.output_dim) # check it
        attn_weights_mat = K.softmax(scaled_weights_mat)
        Z = K.dot(attn_weights_mat, V)
        return Z

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim) # check shape (batch size?)


class MultiHeadSelfAttention(Layer):

    def __init__(self, output_dim, heads, **kwargs):
        self.output_dim = output_dim
        self.heads = heads
        super(MultiHeadSelfAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        # create all needed weights here
        self.W_O = self.add_weight(name='weight_combining_multiheads',
                                   shape=(input_shape[1]*self.heads, input_shape[0]),
                                   dtype='float32',
                                   initializer='glorot_uniform',
                                   trainable=True)
        super(MultiHeadSelfAttention, self).build(input_shape)

    def call(self, X):
        attn_heads = []
        for i in range(self.heads):
            attn_heads.append(SelfAttention(X.shape[1].value)(X))
        concat_layer = Concatenate(axis=1)
        attn_heads_mat = concat_layer(attn_heads)
        output = K.dot(attn_heads_mat, self.W_O)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[0]) # check for shape


class PositionWiseFeedForwardNet(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        self.ff_net1 = Dense(output_dim, activation='relu')
        self.ff_net2 = Dense(output_dim)
        super(PositionWiseFeedForwardNet, self).__init__(**kwargs)

    def call(self, X):
        output = self.ff_net1(X)
        output = self.ff_net2(output)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1])


class Encoder(Layer):

    def __init__(self, output_dim, attn_heads, **kwargs):
        self.output_dim = output_dim
        self.mutihead_attn = MultiHeadSelfAttention(output_dim, attn_heads)
        self.layer_norm = LayerNormalization()
        self.ff_net = PositionWiseFeedForwardNet(output_dim)
        self.add_layer = Add()
        super(Encoder, self).__init__(**kwargs)

    def call(self, X):
        output = self.mutihead_attn(X)
        output = self.add_layer([X, output])
        ff_net_inp = self.layer_norm(output)
        ff_net_opt = self.ff_net(ff_net_inp)
        output = self.add_layer([ff_net_opt, ff_net_inp])
        output = self.layer_norm(output)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[0])


class TransformerEncoder(Layer):

    def __init__(self, output_dim, nencoder, attn_heads, **kwargs):
        self.output_dim = output_dim
        self.num_encoders = nencoder
        self.encoders = [Encoder(self.output_dim, attn_heads) for i in range(self.num_encoders)]
        super(TransformerEncoder, self).__init__(**kwargs)

    def call(self, X):
        inp = X
        for encoder in self.encoders:
            opt = encoder(inp)
            inp = opt
        return inp

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[0])
