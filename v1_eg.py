import os
os.environ['KERAS_BACKEND'] = 'jax'

import jax.numpy as jnp
from jax import vmap, jit, value_and_grad, nn

import optax

from data import ConstrastiveExamples
from models.embeddings import construct_embedding_model
from util import to_pil_img, collage

import argparse
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--num-batches', type=int, default=1000)     # epoch size
parser.add_argument('--batch-size', type=int, default=8)         # outer batch size
parser.add_argument('--num-egs-per-class', type=int, default=3)  # inner batch size
parser.add_argument('--height-width', type=int, default=64)
parser.add_argument('--embedding-dim', type=int, default=128)
parser.add_argument('--learning-rate', type=float, default=1e-3)

# root_dir
# obj_ids
opts = parser.parse_args()
print("opts", opts)

# start with simple case of x1 R, G, B example
c_egs = ConstrastiveExamples(
    root_dir='data/reference_egs',
    obj_ids=["061", # "135","182",  # x1 red
             "111", # "153","198",  # x1 green
             "000", # "017","019"   # x2 blue
            ]
)
dataset = c_egs.dataset(
    num_batches=opts.num_batches,
    batch_size=opts.batch_size,
    objs_per_batch=opts.num_egs_per_class)

embedding_model = construct_embedding_model(
    height=opts.height_width,
    width=opts.height_width,
    embedding_dim=opts.embedding_dim
)
#print(embedding_model.summary())

def main_diagonal_softmax_cross_entropy(logits):
    # cross entropy assuming "labels" are just (0, 1, 2, ...) i.e. where
    # one_hot mask for log_softmax ends up just being the main diagonal
    return -jnp.sum(jnp.diag(nn.log_softmax(logits)))

def constrastive_loss(params, nt_params, x):
    # x (2C,H,W,3)
    embeddings, nt_params = embedding_model.stateless_call(params, nt_params, x, training=True)
    embeddings = embeddings.reshape((opts.num_egs_per_class, 2, opts.embedding_dim))
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

def train_step(params, nt_params, opt_state, x):
    (loss, nt_params_v), grads = calculate_gradients(params, nt_params, x)
    updates, opt_state = opt.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    nt_params = [jnp.mean(p, axis=0) for p in nt_params_v]
    return params, nt_params, opt_state, loss

params = embedding_model.trainable_variables
nt_params = embedding_model.non_trainable_variables
opt_state = opt.init(params)

train_step = jit(train_step)

for e, (x, _y) in enumerate(dataset):
    x = jnp.array(x)
    params, nt_params, opt_state, loss = train_step(params, nt_params, opt_state, x)
    if e % 100 == 0:
        print('e', e, 'loss', loss)

# test against example from final batch
embeddings, _ = embedding_model.stateless_call(params, nt_params, x[0], training=False)
print(jnp.around(jnp.dot(embeddings, embeddings.T), 2))