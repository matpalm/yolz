import os
os.environ['KERAS_BACKEND'] = 'jax'

import json, tqdm

import jax.numpy as jnp
from jax import vmap, jit, value_and_grad, nn

import optax

from data import construct_datasets, jnp_arrayed
from models.yolz import Yolz
import util

import numpy as np
np.set_printoptions(precision=5, threshold=10000, suppress=True, linewidth=10000)

import argparse
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--run-dir', type=str, required=True,
                    help='where to store weights, losses.json, examples etc')
parser.add_argument('--num-batches', type=int, default=100,
                    help='effective epoch length')
parser.add_argument('--num-obj-references', type=int, default=16,
                    help='(N). number of reference samples for each instance in C')
parser.add_argument('--num-focus-objs', type=int, default=32,
                    help='(C). inner batch size, 2x for a,p pair.'
                         ' if None, use |--eg-obj-ids-json|')
parser.add_argument('--models-config-json', type=str, required=True,
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
parser.add_argument('--seed', type=int, default=123)
opts = parser.parse_args()
print("opts", opts)

if (opts.eg_obj_ids_json is None) or (opts.eg_obj_ids_json == ''):
    obj_ids = None
else:
    obj_ids = json.loads(opts.eg_obj_ids_json)

# load base model config
with open(opts.models_config_json, 'r') as f:
    models_config = json.load(f)
print('base models_config', models_config)

# create model and extract initial params
yolz = Yolz(
    models_config,
    initial_weights_pkl=None,
    stop_anchor_gradient=True,
    contrastive_loss_weight=1,
    classifier_loss_weight=0,  # back port to classifier=0 ignore for v1.5
    )
params, nt_params = yolz.get_params()

# datasets
dataset = construct_datasets(
    opts.eg_root_dir, opts.num_batches,
    obj_ids, yolz.classifier_spatial_size(), opts,
    opts.seed)

# set up training step & optimiser
opt = optax.adam(learning_rate=opts.learning_rate)

opt_state = opt.init(params)
def train_step(
    params, nt_params, opt_state,
    obj_x, scene_x, scene_y_true):
    # calculate gradients
    (loss, nt_params), grads = yolz.calculate_gradients(
        params, nt_params, obj_x, scene_x, scene_y_true)
    # calculate updates from optimiser
    updates, opt_state = opt.update(grads, opt_state, params)
    # apply updates to get new params
    params = optax.apply_updates(params, updates)
    # return
    return params, nt_params, opt_state, loss

# compile various utils
train_step = jit(train_step)
calculate_individual_losses = jit(yolz.calculate_individual_losses)

losses = []
with tqdm.tqdm(dataset, total=opts.num_batches) as progress:
    for step, (obj_x, scene_x, scene_y_true) in enumerate(jnp_arrayed(progress)):

        params, nt_params, opt_state, loss = train_step(
            params, nt_params, opt_state,
            obj_x, scene_x, scene_y_true)

        if step % 10 == 0:
            metric_loss, scene_loss, _ = calculate_individual_losses(
                params, nt_params, obj_x, scene_x, scene_y_true)
            metric_loss, scene_loss = map(float, (metric_loss, scene_loss))

            progress.set_description(
                f"step {step} losses (weighted)"
                f" metric {metric_loss} scene {scene_loss}")

        if step % 500 == 0:

            # write latest weights
            yolz.write_weights(params, nt_params,
                               os.path.join(opts.run_dir, 'models_weights.pkl'))
