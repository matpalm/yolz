import os
os.environ['KERAS_BACKEND'] = 'jax'

import json, tqdm
import pickle

import jax.numpy as jnp
from jax import vmap, jit, value_and_grad, nn
from jax.lax import stop_gradient

import optax

from data import construct_datasets, jnp_arrayed
from models.yolz import Yolz
from util import to_pil_img, smooth_highlight, highlight, collage, create_dir_if_required

import numpy as np
np.set_printoptions(precision=5, threshold=10000, suppress=True, linewidth=10000)

# from jaxopt
from jax.nn import softplus
def binary_logistic_loss(label: int, logit: float) -> float:
  return softplus(jnp.where(label, -logit, logit))


import argparse
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--run-dir', type=str, default=100,
                    help='where to store weights, losses.json, examples etc')
parser.add_argument('--num-batches', type=int, default=100,
                    help='effective epoch length')
parser.add_argument('--num-obj-references', type=int, default=16,
                    help='(N). number of reference samples for each instance in C')
parser.add_argument('--num-focus-objs', type=int, default=32,
                    help='(C). for embedding branch represents number of'
                        ' anchor/positive pairs. for scene branch represents'
                        ' the number of examples.')
parser.add_argument('--models-config-json', type=str, required=True,
                    help='embedding model config json file')
parser.add_argument('--eg-train-root-dir', type=str,
                    default='data/train/reference_patches',
                    help='.')
parser.add_argument('--eg-validate-root-dir', type=str,
                    default='data/validate/reference_patches',
                    help='.')
parser.add_argument('--eg-obj-ids-json', type=str, default=None,
                    help='ids to use, json str. if None use all'
                         ' entries from --eg-root-dir')
parser.add_argument('--learning-rate', type=float, default=1e-3,
                    help='adam learning rate')
parser.add_argument('--stop-anchor-gradient', action='store_true')
parser.add_argument('--contrastive-loss-weight', type=float, default=1.0)
parser.add_argument('--classifier-loss-weight', type=float, default=100.0)
opts = parser.parse_args()
print("opts", opts)

if (opts.eg_obj_ids_json is None) or (opts.eg_obj_ids_json == ''):
    obj_ids = None
else:
    obj_ids = json.loads(opts.eg_obj_ids_json)

with open(opts.models_config_json, 'r') as f:
    models_config = json.load(f)
print('models_config', models_config)

# create model and extract initial params
yolz = Yolz(models_config)
params, nt_params = yolz.get_params()

# datasets
train_ds = construct_datasets(
    opts.eg_train_root_dir, opts.num_batches,
    obj_ids, yolz.classifier_spatial_size(), opts)
validate_ds = construct_datasets(
    opts.eg_validate_root_dir, opts.num_batches,
    obj_ids, yolz.classifier_spatial_size(), opts)

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
test_step = jit(yolz.test_step)
calculate_individual_losses = jit(yolz.calculate_individual_losses)

print("running", opts.run_dir)

def generate_debug_imgs(step, obj_x, scene_x, scene_y_true, split):
    # obj_x = jnp.array(obj_x)
    # scene_x = jnp.array(scene_x)
    # scene_y_true = jnp.array(scene_y_true)
    create_dir_if_required(os.path.join(opts.run_dir, 'debug_imgs'))
    y_pred = nn.sigmoid(test_step(params, nt_params, obj_x, scene_x).squeeze())
    anchors0 = list(map(to_pil_img, obj_x[0, 0]))
    positives0 = list(map(to_pil_img, obj_x[0, 1]))
    anchor_positives = anchors0 + positives0
    collage(anchor_positives, 2, len(anchors0)).save(
        os.path.join(opts.run_dir, 'debug_imgs', f"s{step:06d}_{split}_anchor_positives.png"))
    scene0 = to_pil_img(scene_x[0])
    highlight(scene0, scene_y_true[0]).save(
        os.path.join(opts.run_dir, 'debug_imgs', f"s{step:06d}_{split}_y_true.png"))
    scene0 = to_pil_img(scene_x[0])
    smooth_highlight(scene0, y_pred[0]).save(
        os.path.join(opts.run_dir, 'debug_imgs', f"s{step:06d}_{split}_y_pred.png"))

def mean_log_loss(params, nt_params, root_dir, num_egs):
    ds_for_log_loss = construct_datasets(
        root_dir, num_egs,
        obj_ids, yolz.classifier_spatial_size(), opts)
    losses = []
    for obj_x, scene_x, scene_y_true in jnp_arrayed(ds_for_log_loss):
        y_true = scene_y_true.flatten()
        y_pred_logits = test_step(params, nt_params, obj_x, scene_x).flatten()
        log_loss = jnp.mean(binary_logistic_loss(y_true, y_pred_logits))
        losses.append(float(log_loss))
    return np.mean(losses)

losses = []
with tqdm.tqdm(train_ds, total=opts.num_batches) as progress:
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
                f" metric {(metric_loss * opts.contrastive_loss_weight):0.5f}"
                f" scene  {(scene_loss * opts.classifier_loss_weight):0.5f}")

            losses.append((step, metric_loss, scene_loss))

        if step % 1000 == 0:
            train_log_loss = mean_log_loss(
                params, nt_params, opts.eg_train_root_dir, num_egs=100)
            validation_log_loss = mean_log_loss(
                params, nt_params, opts.eg_validate_root_dir, num_egs=100)
            print("log_loss", step, 'train', train_log_loss, 'validate', validation_log_loss)

            # generate debug imgs for training with most recent batch
            generate_debug_imgs(step, obj_x, scene_x, scene_y_true, split='train')

            # grab next batch from validation for validation debug imgs
            (obj_x, _obj_y), (scene_x, scene_y_true) = next(validate_ds)
            generate_debug_imgs(step, obj_x, scene_x, scene_y_true, split='validate')

            # write latest weights
            yolz.write_weights(params, nt_params,
                               os.path.join(opts.run_dir, 'models_weights.pkl'))

            # flush losses
            with open(os.path.join(opts.run_dir, 'losses.json'), 'w') as f:
                json.dump(losses, f)



