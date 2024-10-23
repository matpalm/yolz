from typing import List, Dict

import jax.numpy as jnp

from keras.layers import Input, Dense, Conv2D, GlobalMaxPooling2D
from keras.layers import Layer, BatchNormalization, Activation
from keras.models import Model
from keras.initializers import TruncatedNormal

def conv_bn_relu(filters, y):
    main = Conv2D(
        filters=filters, strides=2, kernel_size=3,
        activation=None, padding='same')(y)
    main = BatchNormalization()(main)
    main = Activation('relu')(main)
    branch = Conv2D(
        filters=filters, strides=1, kernel_size=3,
        activation=None, padding='same')(main)
    branch = BatchNormalization()(branch)
    branch = Activation('relu')(branch)
    return main + branch

class L2Normalisation(Layer):
    def call(self, x):
        norm = jnp.linalg.norm(x, axis=-1, keepdims=True)
        return x / norm

def construct_embedding_model(
        height_width: int,
        filter_sizes: List[int],
        embedding_dim: int
        ):

    input = Input((height_width, height_width, 3))

    y = input
    for f in filter_sizes:
        y = conv_bn_relu(filters=f, y=y)
    y = GlobalMaxPooling2D()(y)  # (B, E)

    # embed, with normalisation
    embeddings = Dense(
        embedding_dim,
        use_bias=False,
        kernel_initializer=TruncatedNormal(),
        name='embeddings')(y)  # (B, E)
    embeddings = L2Normalisation()(embeddings)

    return Model(input, embeddings)

def construct_image_backbone(
    height_width: int,
    base_filter_size: int
    ):

    input = Input((height_width, height_width, 3))

    y = input
    for depth in range(6):
        num_filters = base_filter_size * (depth**2)
        print('depth', depth, 'num_filters', num_filters)
        y = conv_bn_relu(filters=num_filters, y=y)

    spatial_size = y.shape[-2]

    return Model(input, y), spatial_size







