
import os
os.environ['KERAS_BACKEND'] = 'jax'

import json
import os

import jax.numpy as jnp
from jax import jit
import numpy as np

from data import construct_datasets
from models.yolz import Yolz
from util import binary_logistic_loss

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
# parser.add_argument('--eg-train-root-dir', type=str,
#                     default='data/train/reference_patches',
#                     help='.')
parser.add_argument('--eg-validate-root-dir', type=str,
                    default='data/validate/reference_patches',
                    help='.')
parser.add_argument('--eg-obj-ids-json', type=str, default=None,
                    help='ids to use, json str. if None use all'
                         ' entries from --eg-root-dir')
# parser.add_argument('--learning-rate', type=float, default=1e-3,
#                     help='adam learning rate')
parser.add_argument('--stop-anchor-gradient', action='store_true')
# parser.add_argument('--contrastive-loss-weight', type=float, default=1.0)
# parser.add_argument('--classifier-loss-weight', type=float, default=100.0)
# parser.add_argument('--weights-pkl', type=str, default=None,
#                     help='where to save final weights')
# parser.add_argument('--losses-json', type=str, default=None,
#                     help='where to write json list of losses')
opts = parser.parse_args()
print("opts", opts)

if (opts.eg_obj_ids_json is None) or (opts.eg_obj_ids_json == ''):
    obj_ids = None
else:
    obj_ids = json.loads(opts.eg_obj_ids_json)

with open(opts.models_config_json, 'r') as f:
    models_config = json.load(f)
print('models_config', models_config)

yolz = Yolz(models_config,
            os.path.join(opts.run_dir, 'models_weights.pkl'))

validate_ds = construct_datasets(
    opts.eg_validate_root_dir, 100, obj_ids, yolz.classifier_spatial_size(), opts)

@jit
def test_step(obj_x, scene_x):
    params, nt_params = yolz.get_params()
    return yolz.test_step(params, nt_params, obj_x, scene_x)

import time

losses = []
times = []
for step, ((obj_x, _obj_y), (scene_x, scene_y_true)) in enumerate(validate_ds):
    obj_x = jnp.array(obj_x)
    scene_x = jnp.array(scene_x)
    scene_y_true = jnp.array(scene_y_true)
    s = time.perf_counter()
    y_pred_logit = test_step(obj_x, scene_x)
    e = time.perf_counter()
    times.append(e-s)
    y_pred_logit = y_pred_logit.flatten()
    y_true = scene_y_true.flatten()
    log_loss = float(jnp.mean(binary_logistic_loss(y_true, y_pred_logit)))
    print(log_loss)
    losses.append(log_loss)

print("mean log loss", np.mean(losses))
print("times", np.min(times), np.mean(times), np.max(times))
