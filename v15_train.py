import os
os.environ['KERAS_BACKEND'] = 'jax'

import json, tqdm

import jax.numpy as jnp
from jax import vmap, jit, value_and_grad, nn

import optax

from data import ObjIdsHelper, ContrastiveExamples
from models.models import construct_embedding_model

import numpy as np
np.set_printoptions(precision=5, threshold=10000, suppress=True, linewidth=10000)

import argparse
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--num-batches', type=int, default=100,
                    help='effective epoch length')
parser.add_argument('--num-obj-references', type=int, default=16,
                    help='(N). number of reference samples for each instance in C')
parser.add_argument('--num-contrastive-examples', type=int, default=32,
                    help='(C). inner batch size, 2x for a,p pair.'
                         ' if None, use |--eg-obj-ids-json|')
parser.add_argument('--model-config-json', type=str, required=True,
                    help='embedding model config json file')
parser.add_argument('--eg-root-dir', type=str,
                    default='data/train/reference_patches',
                    help='.')
parser.add_argument('--eg-obj-ids-json', type=str, default=None,
                    help='ids to use, json str. if None use all'
                         ' entries from --eg-root-dir')
parser.add_argument('--learning-rate', type=float, default=1e-3,
                    help='adam learning rate')
parser.add_argument('--weights-pkl', type=str, default=None,
                    help='where to save final weights')
parser.add_argument('--losses-json', type=str, default=None,
                    help='where to write json list of losses')
opts = parser.parse_args()
print("opts", opts)

if (opts.eg_obj_ids_json is None) or (opts.eg_obj_ids_json == ''):
    obj_ids = None
else:
    obj_ids = json.loads(opts.eg_obj_ids_json)

with open(opts.model_config_json, 'r') as f:
    model_config = json.load(f)
print('model_config', model_config)
embedding_dim = model_config['embedding_dim']

obj_ids_helper = ObjIdsHelper(
    root_dir=opts.eg_root_dir,
    obj_ids=obj_ids,
    seed=123
)

c_egs = ContrastiveExamples(obj_ids_helper)

dataset = c_egs.dataset(
    num_batches=opts.num_batches,                            # B
    num_obj_references=opts.num_obj_references,              # N
    num_contrastive_examples=opts.num_contrastive_examples)  # C

embedding_model = construct_embedding_model(**model_config)
print(embedding_model.summary())

def mean_embeddings(params, nt_params, x):
    # x (N, H, W, 3)
    embeddings, nt_params = embedding_model.stateless_call(
        params, nt_params, x, training=True)  # (N, E)
    # average over N
    embeddings = jnp.mean(embeddings, axis=0)  # (E)
    # (re) L2 normalise
    embeddings /= jnp.linalg.norm(embeddings, axis=-1, keepdims=True)
    return embeddings, nt_params

def main_diagonal_softmax_cross_entropy(logits):
    # cross entropy assuming "labels" are just (0, 1, 2, ...) i.e. where
    # one_hot mask for log_softmax ends up just being the main diagonal
    return -jnp.sum(jnp.diag(nn.log_softmax(logits)))

def contrastive_loss(params, nt_params, x):
    # x (2C, N, H, W, 3) -> embeddings (2C, E)
    embeddings, nt_params = vmap(mean_embeddings, in_axes=(None, None, 0))(
        params, nt_params, x)
    embeddings = embeddings.reshape((-1, 2, embedding_dim))  # (C, 2, E)
    anchors = embeddings[:, 0]
    positives = embeddings[:, 1]
    gram_ish_matrix = jnp.einsum('ae,be->ab', anchors, positives)
    xent = main_diagonal_softmax_cross_entropy(logits=gram_ish_matrix)
    return jnp.mean(xent), nt_params

def calculate_gradients(params, nt_params, x):
    # x (2C, N, H, W, 3)
    grad_fn = value_and_grad(contrastive_loss, has_aux=True)
    (loss, nt_params), grads = grad_fn(params, nt_params, x)
    return (loss, nt_params), grads

opt = optax.adam(learning_rate=opts.learning_rate)

@jit
def train_step(params, nt_params, opt_state, x):
    (loss, nt_params), grads = calculate_gradients(params, nt_params, x)
    nt_params = [jnp.mean(p, axis=0) for p in nt_params]
    updates, opt_state = opt.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, nt_params, opt_state, loss

params = embedding_model.trainable_variables
nt_params = embedding_model.non_trainable_variables
opt_state = opt.init(params)

losses = []

with tqdm.tqdm(dataset, total=opts.num_batches) as progress:
    for e, (x,_y) in enumerate(progress):
        x = jnp.array(x)
        # x (2C, N, H, W, 3)

        if e == 0:
            print("x", x.shape)

        params, nt_params, opt_state, loss = train_step(params, nt_params, opt_state, x)
        losses.append(float(loss))
        if e % 50 == 0:
            progress.set_description(f"loss {loss}")


# set values back in model
for variable, value in zip(embedding_model.trainable_variables, params):
    variable.assign(value)
for variable, value in zip(embedding_model.non_trainable_variables, nt_params):
    variable.assign(value)

# write weights as pickle
import pickle
if opts.weights_pkl is not None:
    with open(opts.weights_pkl, 'wb') as f:
        pickle.dump(embedding_model.get_weights(), f)

# write losses
if opts.losses_json is not None:
    with open(opts.losses_json, 'w') as f:
        json.dump(losses, f)
