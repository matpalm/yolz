from typing import List, Tuple
import math

from keras.layers import Input, Dense, Conv2D, GlobalMaxPooling2D, Reshape
from keras.layers import Layer, BatchNormalization, Activation, Concatenate
from keras.layers import UpSampling2D
from keras.models import Model
from keras.initializers import TruncatedNormal, Constant
from keras import ops

import jax.numpy as jnp

def conv_bn_relu(filters, y, name, strides=2, kernel_size=3):
    main = Conv2D(
        filters=filters, strides=strides, kernel_size=kernel_size,
        activation=None, padding='same',
        name=f"{name}_conv_m")(y)

    main = BatchNormalization(name=f"{name}_bn_m")(main)
    main = Activation('relu', name=f"{name}_relu_m")(main)

    # branch = Conv2D(
    #     filters=filters, strides=1, kernel_size=3,
    #     activation=None, padding='same',
    #     name=f"{name}_conv_b")(main)
    # branch = BatchNormalization(name=f"{name}_bn_b")(branch)
    # branch = Activation('relu', name=f"{name}_relu_b")(branch)

    return main #+ branch

class L2Normalisation(Layer):
    def call(self, x):
        norm = jnp.linalg.norm(x, axis=-1, keepdims=True)
        return x / norm

class Broadcast(Layer):

    def __init__(self, scene_hwc, name):
        super().__init__(name=name)
        self.scene_hwc = scene_hwc

    def call(self, x):
        # batch dim for self.scene_hwc will be None ( from time it was
        # recorded ) so we need to determine the concrete value
        batch_size = x.shape[0]
        scene_shape_without_batch = self.scene_hwc[1:]
        scene_shape = (batch_size, *scene_shape_without_batch)
        return jnp.broadcast_to(x[:,None,None,:], scene_shape)

def construct_embedding_model(
        height_width: int,
        filter_sizes: List[int],
        embedding_dim: int
        ):

    input = Input((height_width, height_width, 3))

    y = input
    for i, f in enumerate(filter_sizes):
        y = conv_bn_relu(filters=f, y=y, name=f"obj_e_{i}")
    y = GlobalMaxPooling2D(name='obj_e_gp')(y)  # (B, E)

    # embed, with normalisation
    embeddings = Dense(
        embedding_dim,
        use_bias=False,
        kernel_initializer=TruncatedNormal(),
        name='obj_embeddings')(y)  # (B, E)
    embeddings = L2Normalisation(name='obj_e_l2')(embeddings)

    return Model(input, embeddings)

def construct_scene_model(
    height_width: int,
    filter_sizes: List[int],
    feature_dim: int,
    classifier_filter_sizes: List[int],
    init_classifier_bias: int=-5,
    mixing_strategy: str='elementwise_add'
    ):

    # scene backbone; downsampling
    scene_input = Input((height_width, height_width, 3), name='scene_input')
    scene_features = scene_input
    downscaling_outputs = []
    for i, f in enumerate(filter_sizes):
        out = conv_bn_relu(filters=f, y=scene_features, name=f"scene_down_{i}")
        downscaling_outputs.append(out)
        scene_features = out

    # inject the obj_embeddings by broadcasting them to match the spatial size of
    # the scene features and element wise adding them
    obj_embedding_input = Input((feature_dim,), name='obj_embedding_inp')
    # currently all mixing_strategy approaches require the following...
    if obj_embedding_input.shape[-1] != feature_dim:
        raise Exception(f"expected obj_embedding_input embedding dim [{obj_embedding_input.shape}]"
                        f" to match scene feature dim [{feature_dim}]")

    if mixing_strategy == 'elementwise_add':
        scene_features = Dense(feature_dim, use_bias=False, activation=None,
                               kernel_initializer=TruncatedNormal(), name='scene_features')(scene_features)
        obj_embeddings = Broadcast(scene_features.shape, name='broadcast_e')(obj_embedding_input)
        obj_scene_features = scene_features + obj_embeddings

    elif mixing_strategy == 'attention':
        query = obj_embedding_input
        keys = Dense(feature_dim, use_bias=False, activation=None,
                     kernel_initializer=TruncatedNormal(), name='keys')(scene_features)
        values = Dense(feature_dim, use_bias=False, activation=None,
                     kernel_initializer=TruncatedNormal(), name='values')(scene_features)
        query_key_scale = 1.0 / math.sqrt(feature_dim)
        scores = ops.einsum('cf,chwf->chw', query, keys)
        scores /= query_key_scale
        scores = ops.softmax(scores, axis=(-1, -2))
        obj_scene_features = ops.einsum('chwv,chw->chwv', values, scores)

    else:
        raise Exception(f"unknown mixing_strategy [{mixing_strategy}]")

    # upsampling back up, only 3 layers
    y = obj_scene_features
    down_idx = -2
    for i in range(3):
        y = UpSampling2D()(y)
        y = conv_bn_relu(filters=filter_sizes[down_idx], y=y,
                         strides=1, kernel_size=1,
                         name=f"scene_up_{i}")
        y += downscaling_outputs[down_idx]
        down_idx -= 1

    # TODO: could also use squeeze excite like approach? i.e. small MLP applied to
    #       obj embeddings, sigmoid, and use that that to scale scene_features

    # add classifier ( logits )
    classifier = y
    for i, f in enumerate(classifier_filter_sizes):
        classifier = conv_bn_relu(filters=f, y=classifier,
                                  strides=1, kernel_size=1,
                                  name=f"classifier_{i}")
    classifier = Dense(1, name='classifier',
                       bias_initializer=Constant(init_classifier_bias))(classifier)

    return Model(inputs=[scene_input, obj_embedding_input],
                 outputs=classifier)


