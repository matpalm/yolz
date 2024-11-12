import pickle
from typing import Tuple

import jax.numpy as jnp
from jax import vmap, value_and_grad, nn

from models.models import construct_embedding_model
from models.models import construct_scene_model

import optax

class Yolz(object):

    def __init__(self, models_config,
                 initial_weights_pkl: str=None,
                 contrastive_loss_weight: int=1,
                 classifier_loss_weight: int=10,
                 focal_loss_alpha: float=0.25,
                 focal_loss_gamma: float=2.0):

        # clumsy
        assert models_config['embedding']['embedding_dim'] == \
            models_config['scene']['feature_dim']

        self.embedding_model = construct_embedding_model(**models_config['embedding'])
        self.scene_model = construct_scene_model(**models_config['scene'])
        if initial_weights_pkl is not None:
            with open(initial_weights_pkl, 'rb') as f:
                e_weights, s_weights = pickle.load(f)
                self.embedding_model.set_weights(e_weights)
                self.scene_model.set_weights(s_weights)
        self.contrastive_loss_weight = contrastive_loss_weight
        self.classifier_loss_weight = classifier_loss_weight
        self.focal_loss_alpha = focal_loss_alpha
        self.focal_loss_gamma = focal_loss_gamma

    @staticmethod
    def main_diagonal_softmax_cross_entropy(logits):
        # cross entropy assuming "labels" are just (0, 1, 2, ...) i.e. where
        # one_hot mask for log_softmax ends up just being the main diagonal
        return -jnp.sum(jnp.diag(nn.log_softmax(logits)))

    def get_params(self):
        e_params = self.embedding_model.trainable_variables
        e_nt_params = self.embedding_model.non_trainable_variables
        s_params = self.scene_model.trainable_variables
        s_nt_params = self.scene_model.non_trainable_variables
        params = e_params, s_params
        nt_params = e_nt_params, s_nt_params
        return params, nt_params

    def mean_embeddings(self, e_params, e_nt_params, x, training):
        # x (N, H, W, 3)
        embeddings, e_nt_params = self.embedding_model.stateless_call(
            e_params, e_nt_params, x, training=training)  # (N, E)
        # average over N
        embeddings = jnp.mean(embeddings, axis=0)  # (E)
        # (re) L2 normalise
        embeddings /= jnp.linalg.norm(embeddings, axis=-1, keepdims=True)
        return embeddings, e_nt_params  # (E,)

    def train_step(self,
                   params, nt_params,
                   anchors_a, positives_a, scene_img_a):

        # split sub model params and nt_params
        e_params, s_params = params
        e_nt_params, s_nt_params = nt_params
        TRAINING = True

        # calculate mean embeddings for anchors and positives
        # we collect stats re: training for anchors, but drop nt params for positives
        v_mean_embeddings = vmap(self.mean_embeddings, in_axes=(None, None, 0, None))
        positive_embeddings, _ = v_mean_embeddings(e_params, e_nt_params, positives_a, TRAINING)
        anchor_embeddings, e_nt_params = v_mean_embeddings(e_params, e_nt_params, anchors_a, TRAINING)
        e_nt_params = [jnp.mean(p, axis=0) for p in e_nt_params]

        # run scene, using both rgb input and anchors as the embeddings
        y_pred_logits, s_nt_params = self.scene_model.stateless_call(
            s_params, s_nt_params,
            [scene_img_a, anchor_embeddings], TRAINING)

        # return
        nt_params = e_nt_params, s_nt_params
        return anchor_embeddings, positive_embeddings, y_pred_logits, nt_params

    def test_step(self,
                  params, nt_params,
                  anchors_a, scene_img_a):

        # split sub model params and nt_params
        e_params, s_params = params
        e_nt_params, s_nt_params = nt_params
        NOT_TRAINING = False

        # calculate mean embeddings for anchors and positives
        # we collect stats re: training for anchors, but drop nt params for positives
        v_mean_embeddings = vmap(self.mean_embeddings, in_axes=(None, None, 0, None))
        anchor_embeddings, _ = v_mean_embeddings(e_params, e_nt_params, anchors_a, NOT_TRAINING)

        # run scene, using both rgb input and anchors as the embeddings
        y_pred_logits, _ = self.scene_model.stateless_call(
            s_params, s_nt_params,
            [scene_img_a, anchor_embeddings], NOT_TRAINING)

        # return
        return anchor_embeddings, y_pred_logits

    def calculate_individual_losses(
            self,
            params, nt_params,
            anchors_a, positives_a, scene_img_a, masks_a):

        # run forward through two networks
        anchor_embeddings, positive_embeddings, y_pred_logits, nt_params = self.train_step(
            params, nt_params,
            anchors_a, positives_a, scene_img_a)

        # calculate contrastive loss from obj embeddings
        gram_ish_matrix = jnp.einsum('ae,be->ab', anchor_embeddings, positive_embeddings)
        metric_losses = self.main_diagonal_softmax_cross_entropy(logits=gram_ish_matrix)
        metric_loss = jnp.mean(metric_losses)

        # calculate classifier loss is binary cross entropy ( mean across all instances )
        scene_losses = optax.losses.sigmoid_focal_loss(
            logits=y_pred_logits.flatten(),
            labels=masks_a.flatten(),
            alpha=self.focal_loss_alpha,  # how much we weight loss for positives ( vs negatives )
            gamma=self.focal_loss_gamma
            )
        scene_loss = jnp.mean(scene_losses)

        # return losses ( with nt_params updated from forward call )
        return metric_loss, scene_loss, nt_params

    def calculate_single_weighted_loss(
            self, params, nt_params,
            anchors_a, positives_a, scene_img_a, masks_a):

        metric_loss, scene_loss, nt_params = self.calculate_individual_losses(
            params, nt_params,
            anchors_a, positives_a, scene_img_a, masks_a)

        loss = metric_loss * self.contrastive_loss_weight
        loss += scene_loss * self.classifier_loss_weight

        return loss, nt_params

    def calculate_gradients(self, params, nt_params,
                            anchors_a, positives_a, scene_img_a, masks_a):

        grad_fn = value_and_grad(self.calculate_single_weighted_loss, has_aux=True)
        (loss, nt_params), grads = grad_fn(
            params, nt_params,
            anchors_a, positives_a, scene_img_a, masks_a)
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