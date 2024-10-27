
import os
os.environ['KERAS_BACKEND'] = 'jax'

import pickle

import jax.numpy as jnp
from jax import vmap, jit, value_and_grad, nn
from jax.lax import stop_gradient

from models.models import construct_embedding_model
from models.models import construct_scene_model

import optax

class Yolz(object):

    def __init__(self, models_config,
                 initial_weights_pkl: str=None,
                 stop_anchor_gradient: bool=False,
                 contrastive_loss_weight: int=1,
                 classifier_loss_weight: int=100):

        self.embedding_model = construct_embedding_model(**models_config['embedding'])
        self.scene_model = construct_scene_model(**models_config['scene'])
        if initial_weights_pkl is not None:
            with open(initial_weights_pkl, 'rb') as f:
                e_weights, s_weights = pickle.load(f)
                self.embedding_model.set_weights(e_weights)
                self.scene_model.set_weights(s_weights)
        self.stop_anchor_gradient = stop_anchor_gradient
        self.contrastive_loss_weight = contrastive_loss_weight
        self.classifier_loss_weight = classifier_loss_weight

    def get_params(self):
        e_params = self.embedding_model.trainable_variables
        e_nt_params = self.embedding_model.non_trainable_variables
        s_params = self.scene_model.trainable_variables
        s_nt_params = self.scene_model.non_trainable_variables
        params = e_params, s_params
        nt_params = e_nt_params, s_nt_params
        return params, nt_params

    def classifier_spatial_size(self):
        _, h, w, _ = self.scene_model.output.shape
        assert h == w
        return h

    @staticmethod
    def main_diagonal_softmax_cross_entropy(logits):
        # cross entropy assuming "labels" are just (0, 1, 2, ...) i.e. where
        # one_hot mask for log_softmax ends up just being the main diagonal
        return -jnp.sum(jnp.diag(nn.log_softmax(logits)))

    def mean_embeddings(self, params, nt_params, x, training):
        # x (N, H, W, 3)
        embeddings, nt_params = self.embedding_model.stateless_call(
            params, nt_params, x, training=training)  # (N, E)
        # average over N
        embeddings = jnp.mean(embeddings, axis=0)  # (E)
        # (re) L2 normalise
        embeddings /= jnp.linalg.norm(embeddings, axis=-1, keepdims=True)
        return embeddings, nt_params  # (E,)

    def forward(self, params, nt_params, obj_x, scene_x, training):
        # obj_x    (C, 2, N, oHW, oHW, 3)
        # scene_x  (C, sHW, sHW, 3)
        # scene_y  (C, G, G, 1)

        e_params, s_params = params
        e_nt_params, s_nt_params = nt_params

        # first run obj reference branch

        # flatten obj_x to single 2C "batch" over N to get common batch norm stats
        # TODO: how are these stats skewed w.r.t to fact we'll call over N during inference
        C = obj_x.shape[0]
        nhwc = obj_x.shape[-4:]
        obj_x = obj_x.reshape((-1, *nhwc))  # (2C, N, oHW, oHW, 3)

        # run through mean embeddings which reduces over N
        # ( and average non trainables )
        v_mean_embeddings = vmap(self.mean_embeddings, in_axes=(None, None, 0, None))
        obj_embeddings, e_nt_params = v_mean_embeddings(
            e_params, e_nt_params, obj_x, training)  # (2C, E)
        e_nt_params = [jnp.mean(p, axis=0) for p in e_nt_params]

        # reshape back to split anchors and positives
        obj_embeddings = obj_embeddings.reshape((C, 2, -1))  # (C, 2, E)
        anchors = obj_embeddings[:,0]
        positives = obj_embeddings[:,1]

        # second; run scene branch runs ( with just anchors for obj references )

        # classifier_logits (C, G, G, 1)
        if self.stop_anchor_gradient:
            anchors = stop_gradient(anchors)
        classifier_logits, s_nt_params = self.scene_model.stateless_call(
            s_params, s_nt_params, [scene_x, anchors], training=training)

        nt_params = e_nt_params, s_nt_params
        return anchors, positives, classifier_logits, nt_params

    def test_step(self, params, nt_params, obj_x, scene_x):
        _anchors, _positives, classifier_logits, _nt_params = self.forward(
            params, nt_params, obj_x, scene_x, training=False)
        return classifier_logits

    def calculate_individual_losses(self, params, nt_params, obj_x, scene_x, scene_y_true):
        # obj_x    (C, 2, N, oHW, oHW, 3)
        # scene_x  (C, sHW, sHW, 3)
        # scene_y  (C, G, G, 1)

        # run forward through two networks
        anchors, positives, classifier_logits, nt_params = self.forward(
            params, nt_params, obj_x, scene_x, training=True)

        # calculate contrastive loss from obj embeddings
        gram_ish_matrix = jnp.einsum('ae,be->ab', anchors, positives)
        metric_losses = self.main_diagonal_softmax_cross_entropy(logits=gram_ish_matrix)
        metric_loss = jnp.mean(metric_losses)

        # calculate classifier loss is binary cross entropy ( mean across all instances )
        scene_losses = optax.losses.sigmoid_binary_cross_entropy(
            logits=classifier_logits.flatten(),
            labels=scene_y_true.flatten())
        scene_loss = jnp.mean(scene_losses)

        # return losses ( with nt_params updated from forward call )
        return metric_loss, scene_loss, nt_params

    def calculate_single_loss(self, params, nt_params, obj_x, scene_x, scene_y_true):
        metric_loss, scene_loss, nt_params = self.calculate_individual_losses(
            params, nt_params, obj_x, scene_x, scene_y_true)
        loss = metric_loss * self.contrastive_loss_weight
        loss += scene_loss * self.classifier_loss_weight
        return loss,  nt_params

    def calculate_gradients(self, params, nt_params, obj_x, scene_x, scene_y_true):
        # obj_x    (C, 2, N, oHW, oHW, 3)
        # scene_x  (C, sHW, sHW, 3)
        # scene_y  (C, G, G, 1)
        grad_fn = value_and_grad(self.calculate_single_loss, has_aux=True)
        (loss, nt_params), grads = grad_fn(
            params, nt_params, obj_x, scene_x, scene_y_true)
        return (loss, nt_params), grads

    def write_weights(self, params, nt_params, weights_pkl):
        # set values back in model
        e_params, s_params = params
        e_nt_params, s_nt_params = nt_params
        for variable, value in zip(self.embedding_model.trainable_variables, e_params):
            variable.assign(value)
        for variable, value in zip(self.embedding_model.non_trainable_variables, e_nt_params):
            variable.assign(value)
        for variable, value in zip(self.scene_model.trainable_variables, s_params):
            variable.assign(value)
        for variable, value in zip(self.scene_model.non_trainable_variables, s_nt_params):
            variable.assign(value)
        # write weights as pickle
        with open(weights_pkl, 'wb') as f:
            weights = (self.embedding_model.get_weights(),
                       self.scene_model.get_weights())
            pickle.dump(weights, f)