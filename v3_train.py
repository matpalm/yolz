import os
os.environ['KERAS_BACKEND'] = 'jax'

import jax.numpy as jnp
from jax import vmap, jit, value_and_grad, nn

import optax

from data import ObjIdsHelper, ContrastiveExamples, SceneExamples
from models.models import construct_embedding_model
from models.models import construct_scene_model
from util import to_pil_img, smooth_highlight, collage

class Opts:
    num_batches = 10000         # effective epoch length

    obj_height_width = 64
    num_obj_references = 8     # N number of reference examples given for each object
    num_focus_objs = 8         # C total number of classes used for contrasting & focus in scene
    obj_filter_sizes = [8, 16, 32, 64]
    obj_embedding_dim = 64     # E dim for obj reference embeddings

    scene_height_width = 640
    scene_filter_sizes = [4, 8, 16, 32, 64, 64]
    scene_feature_dim = 64     # F dim for scene features
    classifier_filter_sizes = [16, 16]

    learning_rate = 1e-4

opts = Opts()

def shapes(debug_str, list_of_variables):
    return f"{debug_str} ({len(list_of_variables)}) {[v.shape for v in list_of_variables]}"

## models

embedding_model = construct_embedding_model(
    opts.obj_height_width, opts.obj_filter_sizes, opts.obj_embedding_dim)
scene_model = construct_scene_model(
    scene_height_width=opts.scene_height_width,
    scene_filter_sizes=opts.scene_filter_sizes,
    scene_feature_dim=opts.scene_feature_dim,
    expected_obj_embedding_dim=opts.obj_embedding_dim,
    classifier_filter_sizes=[8, 16] #opts.classifier_filter_sizes
)

# double check classifier output matches what the scene dataset has been
# configured to run
# TODO: make dataset just match model!!
classifier_shape = scene_model.output.shape
classifier_spatial_w = classifier_shape[1]
classifier_spatial_h = classifier_shape[2]
assert classifier_spatial_w == classifier_spatial_h

## datasets

obj_ids_helper = ObjIdsHelper(
    root_dir='data/train/reference_patches/',
    obj_ids=["061", "135","182",  # x3 red
             "111", "153","198",  # x3 green
             "000", "017","019"], # x3 blue
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

## training

loss_weights = { 'constrastive': 1.0, 'scene': 100.0 }

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

    # first flatten obj_x to single 2C "batch" over N to get common batch norm stats
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
    #print('anchors', anchors.shape)

    # second; run scene branch runs ( with just anchors for obj references )

    # classifier_out (C, G, G, 1) ( logits )
    classifier_out, s_nt_params = scene_model.stateless_call(
        s_params, s_nt_params, [scene_x, anchors], training=training)
    #print('classifier_out', classifier_out.shape)
    #print(shapes('s_nt_params', s_nt_params))

    nt_params = e_nt_params, s_nt_params
    return anchors, positives, classifier_out, nt_params

def calculate_individual_losses(params, nt_params, obj_x, scene_x, scene_y_true):
    # obj_x    (C, 2, N, oHW, oHW, 3)
    # scene_x  (C, sHW, sHW, 3)
    # scene_y  (C, G, G, 1)

    # run forward through two networks
    anchors, positives, classifier_out, nt_params = forward(
        params, nt_params, obj_x, scene_x, training=True)

    # calculate contrastive loss from obj embeddings
    gram_ish_matrix = jnp.einsum('ae,be->ab', anchors, positives)
    metric_losses = main_diagonal_softmax_cross_entropy(logits=gram_ish_matrix)
    metric_loss = jnp.mean(metric_losses)

    # calculate classifier loss is binary cross entropy ( mean across all instances )
    scene_losses = optax.losses.sigmoid_binary_cross_entropy(
        logits=classifier_out.flatten(),
        labels=scene_y_true.flatten())
    scene_loss = jnp.mean(scene_losses)

    # return losses ( with nt_params updated from forward call )
    return metric_loss, scene_loss, nt_params

def calculate_single_loss(params, nt_params, obj_x, scene_x, scene_y_true):
    metric_loss, scene_loss, nt_params = calculate_individual_losses(
        params, nt_params, obj_x, scene_x, scene_y_true)
    loss = (loss_weights['constrastive']) * metric_loss + (loss_weights['scene'] * scene_loss)
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

    # # this is bit clumsy; because params was passed to grad call
    # # with _all_ params (including non trainables, we get back
    # # grads w.r.t to the non trainables ( which
    # # will be zero and can be ignored... )
    # e_params, _, s_params, _ = params
    # e_grads, _, s_grads, _ = grads

    # calculate updates from optimiser
    updates, opt_state = opt.update(grads, opt_state, params)

    # apply updates to get new params
    params = optax.apply_updates(params, updates)

    # return
    return params, nt_params, opt_state, loss

def test_step(
    params, nt_params,
    obj_x, scene_x):
    _anchors, _positives, classifier_out, _nt_params = forward(
        params, nt_params, obj_x, scene_x, training=False)
    return classifier_out

print("compiling")
train_step = jit(train_step)
test_step = jit(test_step)

e_params = embedding_model.trainable_variables
e_nt_params = embedding_model.non_trainable_variables
s_params = scene_model.trainable_variables
s_nt_params = scene_model.non_trainable_variables
# print(shapes('e_params', e_params))
# print(shapes('e_nt_params', e_nt_params))
# print(shapes('s_params', s_params))
# print(shapes('s_nt_params', s_nt_params))

# package up trainable and non trainables in tuples
params = e_params, s_params
nt_params = e_nt_params, s_nt_params

# optimser will run against both
opt_state = opt.init(params)

print("running")

for step, ((obj_x, _obj_y), (scene_x, scene_y_true)) in enumerate(zip(obj_ds, scene_ds)):
    obj_x = jnp.array(obj_x)
    scene_x = jnp.array(scene_x)
    scene_y_true = jnp.array(scene_y_true)

    params, nt_params, opt_state, loss = train_step(
        params, nt_params, opt_state,
        obj_x, scene_x, scene_y_true)

    if step % 100 == 0:
        metric_loss, scene_loss, _ = calculate_individual_losses(
            params, nt_params, obj_x, scene_x, scene_y_true)

        print('step', step, 'metric_loss', metric_loss, 'scene_loss', scene_loss)

        try:
            y_pred = nn.sigmoid(test_step(params, nt_params, obj_x, scene_x).squeeze())
            anchors0 = obj_x[0, 0]
            anchors0 = list(map(to_pil_img, anchors0))
            collage(anchors0, 1, len(anchors0)).save(f"test/s{step:04d}_anchors_eg.png")
            scene0 = to_pil_img(scene_x[0])
            scene0.save(f"test/s{step:04d}_scene_eg.png")
            smooth_highlight(scene0, y_pred[0]).save(f"test/s{step:04d}_highlight_eg.png")
        except Exception as e:
            print("FAILED TO GENERATE IMAGFES?", str(e))