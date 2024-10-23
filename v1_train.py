import os
os.environ['KERAS_BACKEND'] = 'jax'

import json

import jax.numpy as jnp
from jax import vmap, jit, value_and_grad, nn

import optax

from data import ContrastiveExamples
from models.models import construct_embedding_model

import numpy as np
np.set_printoptions(precision=5, threshold=10000, suppress=True, linewidth=10000)

import argparse
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--num-batches', type=int, default=100,
                    help='effective epoch length')
parser.add_argument('--batch-size', type=int, default=128,
                    help='(B) outer batch size')
# parser.add_argument('--num-obj-references', type=int, default=16,
#                     help='(N). number of reference samples for each instance in C')
parser.add_argument('--num-contrastive-objs', type=int, default=32,
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

c_egs = ContrastiveExamples(
    root_dir=opts.eg_root_dir,
    obj_ids=obj_ids
)
dataset = c_egs.dataset(
    num_batches=opts.num_batches,
    batch_size=opts.batch_size,                        # B
    num_obj_references=1, # single example per object
    num_contrastive_objs=opts.num_contrastive_objs)  # C

embedding_model = construct_embedding_model(**model_config)
print(embedding_model.summary())

def main_diagonal_softmax_cross_entropy(logits):
    # cross entropy assuming "labels" are just (0, 1, 2, ...) i.e. where
    # one_hot mask for log_softmax ends up just being the main diagonal
    return -jnp.sum(jnp.diag(nn.log_softmax(logits)))

def constrastive_loss(params, nt_params, x):
    # x (2C,H,W,3)
    embeddings, nt_params = embedding_model.stateless_call(params, nt_params, x, training=True)
    embeddings = embeddings.reshape((-1, 2, embedding_dim))
    anchors = embeddings[:, 0]
    positives = embeddings[:, 1]
    gram_ish_matrix = jnp.einsum('ae,be->ab', anchors, positives)
    xent = main_diagonal_softmax_cross_entropy(logits=gram_ish_matrix)
    return jnp.mean(xent), nt_params

def constrastive_loss_v(params, nt_params, x):
    # x (B,2C,H,W,3)
    loss_fn_v = vmap(constrastive_loss, in_axes=[None, None, 0])
    loss_v, nt_params_v = loss_fn_v(params, nt_params, x)
    loss = jnp.mean(loss_v)
    return loss, nt_params_v

def calculate_gradients(params, nt_params, x):
    # x (B,2C,H,W,3)
    grad_fn = value_and_grad(constrastive_loss_v, has_aux=True)
    (loss, nt_params_v), grads = grad_fn(params, nt_params, x)
    return (loss, nt_params_v), grads

opt = optax.adam(learning_rate=opts.learning_rate)

@jit
def train_step(params, nt_params, opt_state, x):
    (loss, nt_params_v), grads = calculate_gradients(params, nt_params, x)
    updates, opt_state = opt.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    nt_params = [jnp.mean(p, axis=0) for p in nt_params_v]
    return params, nt_params, opt_state, loss

params = embedding_model.trainable_variables
nt_params = embedding_model.non_trainable_variables
opt_state = opt.init(params)

losses = []
for e, (x, _y) in enumerate(dataset):
    x = jnp.array(x)
    if e == 0:
        print("x", x.shape)
    params, nt_params, opt_state, loss = train_step(params, nt_params, opt_state, x)
    losses.append(loss)
    if e % 50 == 0:
        print('e', e, 'loss', loss)


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

# # test against example from final batch
# embeddings = embedding_model(x[0], training=False)
# gramm_matrix = jnp.dot(embeddings, embeddings.T)  # (-1, 1)
# print("gram min", jnp.min(gramm_matrix))
# sims = (gramm_matrix + 1) / 2  # (0, 1)
# print("sims")
# print(np.around(sims, decimals=1))