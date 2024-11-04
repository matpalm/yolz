
import os
os.environ['KERAS_BACKEND'] = 'jax'

import json
import os

import jax.numpy as jnp
from jax import jit, nn
import numpy as np

from data import construct_datasets, jnp_arrayed
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
parser.add_argument('--embedding-dim', type=int, default=None)
parser.add_argument('--feature-dim', type=int, default=None)
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--threshold', type=float, default=0.1)

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

# clobber with any specifically set flags
if opts.embedding_dim is not None:
    models_config['embedding']['embedding_dim'] = opts.embedding_dim
    models_config['scene']['expected_obj_embedding_dim'] = opts.embedding_dim  # clumsy
if opts.feature_dim is not None:
    models_config['scene']['feature_dim'] = opts.feature_dim
print('models_config with --flag updates', models_config)

yolz = Yolz(models_config,
            os.path.join(opts.run_dir, 'models_weights.pkl'))

validate_ds = construct_datasets(
    opts.eg_validate_root_dir, 100,
    obj_ids, yolz.classifier_spatial_size(), opts,
    random_background_colours=False,
    seed=opts.seed)

@jit
def test_step(obj_x, scene_x):
    params, nt_params = yolz.get_params()
    return yolz.test_step(params, nt_params, obj_x, scene_x)

import time

from sklearn.metrics import *

y_true_all = []
y_pred_all = []

for step, (obj_x, scene_x, scene_y_true) in enumerate(jnp_arrayed(validate_ds)):
    s = time.perf_counter()
    y_pred_logit = test_step(obj_x, scene_x)
    y_pred_logit = y_pred_logit.flatten()
    y_pred = nn.sigmoid(y_pred_logit)
    y_true = scene_y_true.flatten()
    y_pred = (y_pred > opts.threshold).astype(float)
    y_true_all.append(y_true)
    y_pred_all.append(y_pred)
    if step > 3:
        break

y_true_all = np.concatenate(y_true_all)
y_pred_all = np.concatenate(y_pred_all)
print(confusion_matrix(y_true_all, y_pred_all))
print('average_p', average_precision_score(y_true_all, y_pred_all))
print('p', recall_score(y_true_all, y_pred_all))
print('r', precision_score(y_true_all, y_pred_all))
print('f1', f1_score(y_true_all, y_pred_all))
