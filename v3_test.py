
import os
os.environ['KERAS_BACKEND'] = 'jax'

import json
import os
import pickle

import jax.numpy as jnp
from jax import vmap, jit, nn
from jax.lax import stop_gradient


from data import construct_datasets
from models.models import construct_embedding_model
from models.models import construct_scene_model

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

embedding_model = construct_embedding_model(**models_config['embedding'])
embedding_model.summary()

scene_model = construct_scene_model(**models_config['scene'])
scene_model.summary()

classifier_shape = scene_model.output.shape
classifier_spatial_w = classifier_shape[1]
classifier_spatial_h = classifier_shape[2]
assert classifier_spatial_w == classifier_spatial_h

with open(os.path.join(opts.run_dir, 'models_weights.pkl'), 'rb') as f:
    e_weights, s_weights = pickle.load(f)
    embedding_model.set_weights(e_weights)
    scene_model.set_weights(s_weights)

validate_ds = construct_datasets(
    opts.eg_validate_root_dir, 100, obj_ids, classifier_spatial_w, opts)

def mean_embeddings(params, nt_params, x, training):
    # x (N, H, W, 3)
    embeddings, nt_params = embedding_model.stateless_call(
        params, nt_params, x, training=training)  # (N, E)
    # average over N
    embeddings = jnp.mean(embeddings, axis=0)  # (E)
    # (re) L2 normalise
    embeddings /= jnp.linalg.norm(embeddings, axis=-1, keepdims=True)
    return embeddings, nt_params  # (E,)

def forward(params, nt_params, obj_x, scene_x, training):
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
    v_mean_embeddings = vmap(mean_embeddings, in_axes=(None, None, 0, None))
    obj_embeddings, e_nt_params = v_mean_embeddings(
        e_params, e_nt_params, obj_x, training)  # (2C, E)
    e_nt_params = [jnp.mean(p, axis=0) for p in e_nt_params]

    # reshape back to split anchors and positives
    obj_embeddings = obj_embeddings.reshape((C, 2, -1))  # (C, 2, E)
    anchors = obj_embeddings[:,0]
    positives = obj_embeddings[:,1]

    # second; run scene branch runs ( with just anchors for obj references )

    # classifier_logits (C, G, G, 1)
    if opts.stop_anchor_gradient:
        anchors = stop_gradient(anchors)
    classifier_logits, s_nt_params = scene_model.stateless_call(
        s_params, s_nt_params, [scene_x, anchors], training=training)

    nt_params = e_nt_params, s_nt_params
    return anchors, positives, classifier_logits, nt_params

def test_step(
    params, nt_params,
    obj_x, scene_x):
    _anchors, _positives, classifier_logits, _nt_params = forward(
        params, nt_params, obj_x, scene_x, training=False)
    return classifier_logits

test_step = jit(test_step)

e_params = embedding_model.trainable_variables
e_nt_params = embedding_model.non_trainable_variables
s_params = scene_model.trainable_variables
s_nt_params = scene_model.non_trainable_variables
params = e_params, s_params
nt_params = e_nt_params, s_nt_params

# from jaxopt
from jax.nn import softplus
def binary_logistic_loss(label: int, logit: float) -> float:
  return softplus(jnp.where(label, -logit, logit))

losses = []
for step, ((obj_x, _obj_y), (scene_x, scene_y_true)) in enumerate(validate_ds):
    obj_x = jnp.array(obj_x)
    scene_x = jnp.array(scene_x)
    scene_y_true = jnp.array(scene_y_true)

    y_pred_logit = test_step(params, nt_params, obj_x, scene_x).squeeze()
    y_pred_logit = y_pred_logit.flatten()
    y_true = scene_y_true.flatten()
    log_loss = jnp.mean(binary_logistic_loss(y_true, y_pred_logit))

    #print("yp", y_pred.shape, y_pred)
    #print("yt", y_true.shape, y_true)
    losses.append(log_loss(y_true, y_pred))

print("mean log loss", jnp.mean(losses))
