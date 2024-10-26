import os
os.environ['KERAS_BACKEND'] = 'jax'

import json, tqdm

import jax.numpy as jnp
from jax import vmap, jit, value_and_grad, nn

import optax

from data import ObjIdsHelper, ContrastiveExamples, SceneExamples
from models.models import construct_embedding_model
from models.models import construct_scene_model
from util import to_pil_img, smooth_highlight, highlight, collage, create_dir_if_required

import numpy as np
np.set_printoptions(precision=5, threshold=10000, suppress=True, linewidth=10000)

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
parser.add_argument('--contrastive-loss-weight', type=float, default=1.0)
parser.add_argument('--classifier-loss-weight', type=float, default=100.0)

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

# double check classifier output matches what the scene dataset has been
# configured to run
# TODO: make dataset just match model!!
classifier_shape = scene_model.output.shape
classifier_spatial_w = classifier_shape[1]
classifier_spatial_h = classifier_shape[2]
assert classifier_spatial_w == classifier_spatial_h

## datasets

def construct_datasets(root_dir):
    obj_ids_helper = ObjIdsHelper(
        root_dir=root_dir,
        obj_ids=obj_ids,
        seed=123
    )
    obj_egs = ContrastiveExamples(obj_ids_helper)
    obj_ds = obj_egs.dataset(num_batches=opts.num_batches,
                    num_obj_references=opts.num_obj_references,
                    num_contrastive_examples=opts.num_focus_objs)
    print(f"scene grid_size set to {classifier_spatial_h} from scene model")
    scene_egs = SceneExamples(
        obj_ids_helper=obj_ids_helper,
        grid_size=classifier_spatial_w,
        num_other_objs=4,
        instances_per_obj=3,
        seed=123)
    scene_ds = scene_egs.dataset(
        num_batches=opts.num_batches,
        num_focus_objects=opts.num_focus_objs)
    return zip(obj_ds, scene_ds)


train_ds = construct_datasets(opts.eg_train_root_dir)
validate_ds = construct_datasets(opts.eg_validate_root_dir)

## training

def mean_embeddings(params, nt_params, x, training):
    # x (N, H, W, 3)
    embeddings, nt_params = embedding_model.stateless_call(
        params, nt_params, x, training=training)  # (N, E)
    # average over N
    embeddings = jnp.mean(embeddings, axis=0)  # (E)
    # (re) L2 normalise
    embeddings /= jnp.linalg.norm(embeddings, axis=-1, keepdims=True)
    return embeddings, nt_params  # (E,)

def main_diagonal_softmax_cross_entropy(logits):
    # cross entropy assuming "labels" are just (0, 1, 2, ...) i.e. where
    # one_hot mask for log_softmax ends up just being the main diagonal
    return -jnp.sum(jnp.diag(nn.log_softmax(logits)))

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
    classifier_logits, s_nt_params = scene_model.stateless_call(
        s_params, s_nt_params, [scene_x, anchors], training=training)

    nt_params = e_nt_params, s_nt_params
    return anchors, positives, classifier_logits, nt_params

def calculate_individual_losses(params, nt_params, obj_x, scene_x, scene_y_true):
    # obj_x    (C, 2, N, oHW, oHW, 3)
    # scene_x  (C, sHW, sHW, 3)
    # scene_y  (C, G, G, 1)

    # run forward through two networks
    anchors, positives, classifier_logits, nt_params = forward(
        params, nt_params, obj_x, scene_x, training=True)

    # calculate contrastive loss from obj embeddings
    gram_ish_matrix = jnp.einsum('ae,be->ab', anchors, positives)
    metric_losses = main_diagonal_softmax_cross_entropy(logits=gram_ish_matrix)
    metric_loss = jnp.mean(metric_losses)

    # calculate classifier loss is binary cross entropy ( mean across all instances )
    scene_losses = optax.losses.sigmoid_binary_cross_entropy(
        logits=classifier_logits.flatten(),
        labels=scene_y_true.flatten())
    scene_loss = jnp.mean(scene_losses)

    # return losses ( with nt_params updated from forward call )
    return metric_loss, scene_loss, nt_params

def calculate_single_loss(params, nt_params, obj_x, scene_x, scene_y_true):
    metric_loss, scene_loss, nt_params = calculate_individual_losses(
        params, nt_params, obj_x, scene_x, scene_y_true)
    loss = metric_loss * opts.contrastive_loss_weight
    loss += scene_loss * opts.classifier_loss_weight
    return loss,  nt_params

def calculate_gradients(params, nt_params, obj_x, scene_x, scene_y_true):
    # obj_x    (C, 2, N, oHW, oHW, 3)
    # scene_x  (C, sHW, sHW, 3)
    # scene_y  (C, G, G, 1)
    grad_fn = value_and_grad(calculate_single_loss, has_aux=True)
    (loss, nt_params), grads = grad_fn(
        params, nt_params, obj_x, scene_x, scene_y_true)
    return (loss, nt_params), grads

opt = optax.adam(learning_rate=opts.learning_rate)

def train_step(
    params, nt_params, opt_state,
    obj_x, scene_x, scene_y_true):

    # calculate gradients
    (loss, nt_params), grads = calculate_gradients(
        params, nt_params, obj_x, scene_x, scene_y_true)

    # calculate updates from optimiser
    updates, opt_state = opt.update(grads, opt_state, params)

    # apply updates to get new params
    params = optax.apply_updates(params, updates)

    # return
    return params, nt_params, opt_state, loss

def test_step(
    params, nt_params,
    obj_x, scene_x):
    _anchors, _positives, classifier_logits, _nt_params = forward(
        params, nt_params, obj_x, scene_x, training=False)
    return classifier_logits

print("compiling")
train_step = jit(train_step)
test_step = jit(test_step)
calculate_individual_losses = jit(calculate_individual_losses)

# package up trainable and non trainables in tuples
e_params = embedding_model.trainable_variables
e_nt_params = embedding_model.non_trainable_variables
s_params = scene_model.trainable_variables
s_nt_params = scene_model.non_trainable_variables
params = e_params, s_params
nt_params = e_nt_params, s_nt_params

# init optimiser
opt_state = opt.init(params)


print("running", opts.run_dir)

def write_weights():
    # set values back in model
    e_params, s_params = params
    e_nt_params, s_nt_params = nt_params
    for variable, value in zip(embedding_model.trainable_variables, e_params):
        variable.assign(value)
    for variable, value in zip(embedding_model.non_trainable_variables, e_nt_params):
        variable.assign(value)
    for variable, value in zip(scene_model.trainable_variables, s_params):
        variable.assign(value)
    for variable, value in zip(scene_model.non_trainable_variables, s_nt_params):
        variable.assign(value)
    # write weights as pickle
    import pickle
    with open(os.path.join(opts.run_dir, 'models_weights.pkl'), 'wb') as f:
        weights = (embedding_model.get_weights(),
                    scene_model.get_weights())
        pickle.dump(weights, f)

def generate_debug_imgs(step, obj_x, scene_x, scene_y_true, split):
    obj_x = jnp.array(obj_x)
    scene_x = jnp.array(scene_x)
    scene_y_true = jnp.array(scene_y_true)
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

losses = []
with tqdm.tqdm(train_ds, total=opts.num_batches) as progress:
    for step, ((obj_x, _obj_y), (scene_x, scene_y_true)) in enumerate(progress):
        obj_x = jnp.array(obj_x)
        scene_x = jnp.array(scene_x)
        scene_y_true = jnp.array(scene_y_true)

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

        if step % 100 == 0:

            # generate debug imgs for training with most recent batch
            generate_debug_imgs(step, obj_x, scene_x, scene_y_true, split='train')

            # grab next batch from validation for validation debug imgs
            (obj_x, _obj_y), (scene_x, scene_y_true) = next(validate_ds)
            generate_debug_imgs(step, obj_x, scene_x, scene_y_true, split='validate')

            # write latest weights
            write_weights()

            # flush losses
            with open(os.path.join(opts.run_dir, 'losses.json'), 'w') as f:
                json.dump(losses, f)



