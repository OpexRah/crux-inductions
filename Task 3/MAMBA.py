import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from einops import rearrange, repeat
import math

class MambaBlock(layers.Layer):
    def __init__(self, model_input_dims, model_states):
        super(MambaBlock, self).__init__()
        self.model_internal_dim = int(2 * model_input_dims)
        self.delta_t_rank = math.ceil(model_input_dims/16)
        self.in_projection = layers.Dense(self.model_internal_dim * 2, input_shape=(model_input_dims,), use_bias=False)

        self.conv1d = layers.Conv1D(filters=self.model_internal_dim, use_bias=True, kernel_size=4, groups=self.model_internal_dim, data_format='channels_first', padding='causal')

        # this layer takes in current token 'x' and outputs the input-specific Δ, B, C 
        self.x_projection = layers.Dense(self.delta_t_rank + model_states * 2, use_bias=False)

        # this layer projects Δ from delta_t_rank to the mamba internal dimension
        self.delta_t_projection = layers.Dense(self.model_internal_dim,input_shape=(self.delta_t_rank,), use_bias=True)
        self.A = repeat(tf.range(1, model_states+1, dtype=tf.float32),'n -> d n', d=self.model_internal_dim)
        self.A_log = tf.Variable(tf.math.log(self.A),trainable=True, dtype=tf.float32)
        self.D = tf.Variable(np.ones(self.model_internal_dim),trainable=True, dtype=tf.float32)
        self.out_projection = layers.Dense(model_input_dims,input_shape=(self.model_internal_dim,),use_bias=False)

    def call(self, x):
        (batch_size, seq_len, dimension) = x.shape

        x_and_result = self.in_projection(x)
        (x, result) = tf.split(x_and_result, [self.model_internal_dim, self.model_internal_dim], axis=-1)

        x = rearrange(x, 'b l d_in -> b d_in l')
        x = self.conv1d(x)[:, :, :seq_len]
        x = rearrange(x, 'b d_in l -> b l d_in')

        x = tf.nn.swish(x)
        y = self.ssm(x)
        y = y * tf.nn.swish(result)
        return self.out_projection(y)
    
    def ssm(self, x):
        (d_in, n) = self.A_log.shape


        A = -tf.exp(tf.cast(self.A_log, tf.float32)) 
        D = tf.cast(self.D, tf.float32)

        x_dbl = self.x_projection(x) 

        (delta, B, C) = tf.split(
                x_dbl,
                num_or_size_splits=[self.delta_t_rank, n, n],
                axis=-1) 

        delta = tf.nn.softplus(self.delta_t_projection(delta)) 

        # selective scan process
        
        dA = tf.einsum('bld,dn->bldn', delta, A)
        dB_u = tf.einsum('bld,bld,bln->bldn', delta, x, B)

        dA_cumsum = tf.pad(
            dA[:, 1:], [[0, 0], [1, 1], [0, 0], [0, 0]])[:, 1:, :, :]

        dA_cumsum = tf.reverse(dA_cumsum, axis=[1]) 
        dA_cumsum = tf.math.cumsum(dA_cumsum, axis=1)

        dA_cumsum = tf.exp(dA_cumsum)
        dA_cumsum = tf.reverse(dA_cumsum, axis=[1]) 

        x_ss = dB_u * dA_cumsum
        # 1e-12 to avoid division by 0
        x_ss = tf.math.cumsum(x_ss, axis=1)/(dA_cumsum + 1e-12)

        y = tf.einsum('bldn,bln->bld', x_ss, C)

        return y + x * D
    
class SkipConnect(layers.Layer):
    def __init__(self,model_input_dims,model_states):
        super(SkipConnect, self).__init__()
        self.mixer = MambaBlock(model_input_dims,model_states)
        self.norm = layers.LayerNormalization(epsilon=1e-5)

    def call(self, x):
        return self.mixer(self.norm(x)) + x