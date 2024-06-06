import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping
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
        pass