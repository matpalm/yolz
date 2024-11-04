from typing import List, Tuple

from keras.layers import Input, Dense, Conv2D, GlobalMaxPooling2D, Reshape
from keras.layers import Layer, BatchNormalization, Activation, Concatenate
from keras.models import Model
from keras.initializers import TruncatedNormal, Constant

import jax.numpy as jnp

def conv_bn_relu(filters, y, name, one_by_one=False):
    if one_by_one:
        main = Conv2D(
            filters=filters, strides=1, kernel_size=1,
            activation=None, padding='same',
            name=f"{name}_conv1x1_m")(y)
    else:
        main = Conv2D(
            filters=filters, strides=2, kernel_size=3,
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
        print("Broadcasting x", x.shape, 'to shape', scene_shape)
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
    expected_obj_embedding_dim: int,
    classifier_filter_sizes: List[int],
    init_classifier_bias: int=-5
    ):

    # scene backbone
    scene_input = Input((height_width, height_width, 3), name='scene_input')
    y = scene_input
    for i, f in enumerate(filter_sizes):
        y = conv_bn_relu(filters=f, y=y, name=f"scene_{i}")

    # final feature layer ( projection, no relu )
    scene_features = Dense(
        feature_dim,
        use_bias=False, activation=None,
        kernel_initializer=TruncatedNormal(),
        name='scene_features')(y)  # (B, F)

    # input branch from obj_embeddings
    obj_embedding_input = Input((expected_obj_embedding_dim,), name='obj_embedding_inp')
    if obj_embedding_input.shape[-1] != feature_dim:
        raise Exception("needed obj_embedding_input final dim to match feature_dim")

    # broadcast the embeddings to match the spatial size of the features
    # from the scene backbone
    obj_embeddings = Broadcast(scene_features.shape, name='broadcast_e')(obj_embedding_input)

    # element wise add scene_features with object embeddings
    obj_scene_features = scene_features + obj_embeddings

    # TODO: could also use squeeze excite like approach? i.e. small MLP applied to
    #       obj embeddings, sigmoid, and use that that to scale scene_features

    # add classifier ( logits )
    classifier = obj_scene_features
    for i, f in enumerate(classifier_filter_sizes):
        classifier = conv_bn_relu(filters=f, y=classifier,
                                  name=f"classifier_{i}", one_by_one=True)
    classifier = Dense(1, name='classifier',
                       bias_initializer=Constant(init_classifier_bias))(classifier)

    return Model(inputs=[scene_input, obj_embedding_input],
                 outputs=classifier)

