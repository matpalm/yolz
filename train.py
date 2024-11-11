import os
os.environ['KERAS_BACKEND'] = 'jax'

import json, tqdm

import jax.numpy as jnp
from jax import jit, nn
import optax
#from sklearn.metrics import average_precision_score, recall_score, precision_score, f1_score

from data import ContrastiveExamples
from models.yolz import Yolz
from util import array_to_pil_img, mask_to_pil_img, alpha_from_mask, collage, create_dir_if_required

import numpy as np
np.set_printoptions(precision=5, threshold=10000, suppress=True, linewidth=10000)

# # from jaxopt
# from jax.nn import softplus
# def binary_logistic_loss(label: int, logit: float) -> float:
#   return softplus(jnp.where(label, -logit, logit))

import argparse
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--run-dir', type=str, required=True,
                    help='where to store weights, losses.json, examples etc')
parser.add_argument('--num-repeats', type=int, default=1)
# parser.add_argument('--num-obj-references', type=int, default=8,
#                     help='(N). number of samples for each instance sampled from'
#                          ' reference patches')
# parser.add_argument('--models-config-json', type=str, required=True,
#                     help='embedding model config json file')
parser.add_argument('--initial-weights-pkl', type=str, default=None,
                    help='starting weights')
parser.add_argument('--train-root-dir', type=str,
                    default='data/train/',
                    help='root dir for scene and reference_patches')
# parser.add_argument('--validate-root-dir', type=str,
#                     default='data/validate/',
                    # help='.')
# parser.add_argument('--eg-obj-ids-json', type=str, default=None,
#                     help='ids to use, json str. if None use all'
#                          ' entries from --eg-root-dir')
parser.add_argument('--optimiser', type=str, default='adam')
parser.add_argument('--learning-rate', type=float, default=1e-3,
                    help='adam learning rate')
# parser.add_argument('--stop-anchor-gradient', action='store_true')
parser.add_argument('--contrastive-loss-weight', type=float, default=1.0)
parser.add_argument('--classifier-loss-weight', type=float, default=100.0)
parser.add_argument('--focal-loss-alpha', type=float, default=0.5)
parser.add_argument('--focal-loss-gamma', type=float, default=2.0)
parser.add_argument('--use-wandb', action='store_true')
# parser.add_argument('--embedding-dim', type=int, default=None)
# parser.add_argument('--feature-dim', type=int, default=None)
# parser.add_argument('--init-classifier-bias', type=float, default=-5)
parser.add_argument('--seed', type=int, default=123)
# parser.add_argument('--test-threshold', type=float, default=0.1,
#                     help='threshold to use during validation for P/R/F1 calcs')

opts = parser.parse_args()
print("opts", opts)

models_config = {
    'embedding': {'height_width': 64,
                  'filter_sizes': [16, 32, 64, 256],
                  'embedding_dim': 512},
    'scene': {'height_width': 640,
              'filter_sizes': [16, 32, 64, 128, 256, 256],
              'feature_dim': 512,
              'classifier_filter_sizes': [256, 128],
              'init_classifier_bias': -5}
}
with open(os.path.join(opts.run_dir, 'model_config.json'), 'w') as f:
    json.dump(models_config, f)

if opts.use_wandb:
    import wandb
    wandb.init(project='yolz',
                name=opts.run_dir,
                reinit=True)
    config = vars(opts)
    # TODO: work out how to pack model configs here
#    config['embedding_dim'] = models_config['embedding']['embedding_dim']
#    config['feature_dim'] = models_config['scene']['feature_dim']
    for k, v in config.items():
        wandb.config[k] = v

# create model and extract initial params
yolz = Yolz(
    models_config,
    initial_weights_pkl=opts.initial_weights_pkl,
    contrastive_loss_weight=opts.contrastive_loss_weight,
    classifier_loss_weight=opts.classifier_loss_weight,
    focal_loss_alpha=opts.focal_loss_alpha,
    focal_loss_gamma=opts.focal_loss_gamma
    )
params, nt_params = yolz.get_params()
yolz.embedding_model.summary()
yolz.scene_model.summary()

train_ds = ContrastiveExamples(
    root_dir='/dev/shm/zero_shot_detection/data/train/',
    seed=123,
    random_background_colours=True,
    instances_per_obj=8,
    cache_dir='/dev/shm/zero_shot_detection/cache/train')

# validate_ds = ContrastiveExamples(
#     root_dir=opts.validate_root_dir,
#     root_dir=opts.train_root_dir,
#     seed=opts.seed,
#     random_background_colours=False,
#     instances_per_obj=opts.num_obj_references)

# set up training step & optimiser
if opts.optimiser == 'adam':
    opt = optax.adam(learning_rate=opts.learning_rate)
elif opts.optimiser == 'adamw':
    opt = optax.adamw(learning_rate=opts.learning_rate)
elif opts.optimiser == 'sgd':
    opt = optax.sgd(learning_rate=opts.learning_rate, momentum=0.9)
else:
    raise Exception(f"unsupported --optimiser {opts.optimiser}")

opt = optax.adam(learning_rate=opts.learning_rate)
opt_state = opt.init(params)

def train_step(params, nt_params,
               opt_state,
               anchors_a, positives_a, scene_img_a, masks_a):
    # calculate gradients
    (loss, nt_params), grads = yolz.calculate_gradients(
        params, nt_params,
        anchors_a, positives_a, scene_img_a, masks_a)
    # calculate updates from optimiser
    updates, opt_state = opt.update(grads, opt_state, params)
    # apply updates to get new params
    params = optax.apply_updates(params, updates)
    # return
    return params, nt_params, opt_state, loss

train_step = jit(train_step)
calculate_individual_losses = jit(yolz.calculate_individual_losses)

# TODO: reinstate tqdm status bars
step = 0
for step, batch in enumerate(train_ds.tf_dataset(num_repeats=opts.num_repeats)):
    anchors_a, positives_a, scene_img_a, masks_a  = train_ds.process_batch(*batch)

    params, nt_params, opt_state, loss = train_step(
        params, nt_params,
        opt_state,
        anchors_a, positives_a, scene_img_a, masks_a)

    if step % 1000 == 0:
        metric_loss, scene_loss, _ = calculate_individual_losses(
            params, nt_params,
            anchors_a, positives_a, scene_img_a, masks_a)
        print(f"step={step} metric_loss={metric_loss} scene_loss={scene_loss}")

    # TODO: figure out how to best do P/R/F1 etc. flood fills?
    if step > 1000:
        break

y_pred_logits = yolz.test_step(
    params, nt_params,
    anchors_a, positives_a, scene_img_a)
y_pred = nn.sigmoid(y_pred_logits.squeeze())

# debug some reference examples
for obj_idx in [4, 5, 6]:
    num_egs = anchors_a.shape[1]
    ref_images = [array_to_pil_img(anchors_a[obj_idx, a]) for a in range(num_egs)]
    c = collage(ref_images, rows=1, cols=num_egs)
    c.save(f"obj_{obj_idx}.anchors.png")
    y_true_m = mask_to_pil_img(masks_a[obj_idx])
    y_pred_m = mask_to_pil_img(y_pred[obj_idx])
    c = collage([y_true_m, y_pred_m], rows=1, cols=2)
    c.save(f"obj_{obj_idx}.y_true_pred.mask.png")
    scene_rgb = array_to_pil_img(scene_img_a[0])
    scene_with_y_pred = alpha_from_mask(scene_rgb, y_pred_m)
    scene_with_y_true = alpha_from_mask(scene_rgb, y_true_m)
    c = collage([scene_with_y_pred, scene_rgb, scene_with_y_true], rows=1, cols=3)
    c.save(f"obj_{obj_idx}.y_true_pred.alpha.png")

# write final weights
yolz.write_weights(
    params, nt_params,
    os.path.join(opts.run_dir, 'models_weights.pkl'))


# def generate_debug_imgs(step, obj_x, scene_x, scene_y_true, split):
#     obj_x = jnp.array(obj_x)
#     scene_x = jnp.array(scene_x)
#     scene_y_true = jnp.array(scene_y_true)
#     create_dir_if_required(os.path.join(opts.run_dir, 'debug_imgs'))
#     y_pred = nn.sigmoid(test_step(params, nt_params, obj_x, scene_x).squeeze())
#     anchors0 = list(map(to_pil_img, obj_x[0, 0]))
#     positives0 = list(map(to_pil_img, obj_x[0, 1]))
#     anchor_positives = anchors0 + positives0
#     collage(anchor_positives, 2, len(anchors0)).save(
#         os.path.join(opts.run_dir, 'debug_imgs', f"s{step:06d}_{split}_anchor_positives.png"))
#     scene0 = to_pil_img(scene_x[0])
#     highlight(scene0, scene_y_true[0]).save(
#         os.path.join(opts.run_dir, 'debug_imgs', f"s{step:06d}_{split}_y_true.png"))
#     scene0 = to_pil_img(scene_x[0])
#     smooth_highlight(scene0, y_pred[0]).save(
#         os.path.join(opts.run_dir, 'debug_imgs', f"s{step:06d}_{split}_y_pred.png"))

# def mean_log_loss(params, nt_params, root_dir, num_egs):
#     ds_for_log_loss = construct_datasets(
#         root_dir, num_egs,
#         obj_ids, yolz.classifier_spatial_size(), opts)
#     losses = []
#     for obj_x, scene_x, scene_y_true in jnp_arrayed(ds_for_log_loss):
#         y_true = scene_y_true.flatten()
#         y_pred_logits = test_step(params, nt_params, obj_x, scene_x).flatten()
#         log_loss = jnp.mean(binary_logistic_loss(y_true, y_pred_logits))
#         losses.append(float(log_loss))
#     return np.mean(losses)

    # ds_for_log_loss = construct_datasets(
    #     root_dir, num_egs,
    #     obj_ids, yolz.classifier_spatial_size(), opts,
    #     random_background_colours=random_background_colours)
    # y_true_all = []
    # y_pred_all = []
    # for obj_x, scene_x, scene_y_true in jnp_arrayed(ds_for_log_loss):
    #     y_true = scene_y_true.flatten()
    #     y_pred_logits = test_step(params, nt_params, obj_x, scene_x).flatten()
    #     y_pred = nn.sigmoid(y_pred_logits)
    #     y_pred = (y_pred > opts.test_threshold).astype(float)
    #     y_true_all.append(y_true)
    #     y_pred_all.append(y_pred)
    # y_true_all = np.concatenate(y_true_all)
    # y_pred_all = np.concatenate(y_pred_all)

    # return {
    #     'ap': average_precision_score(y_true_all, y_pred_all),
    #     'p': precision_score(y_true_all, y_pred_all),
    #     'r': recall_score(y_true_all, y_pred_all),
    #     'f1': f1_score(y_true_all, y_pred_all)
    #     }

# losses = []
# with tqdm.tqdm(train_ds.tf_dataset(opts.num_repeats)) as progress:
#     for step, batch in enumerate(progress):
#         anchors_a, positives_a, scene_img_a, masks_a  = train_ds.process_batch(*batch)

#         wandb_to_log = {}

#         params, nt_params, opt_state, loss = train_step(
#             params, nt_params, opt_state,
#             anchors_a, positives_a, scene_img_a, masks_a)
#         # print("step", step, "loss", loss)

#         if step % 100 == 0:

#             metric_loss, scene_loss, _ = calculate_individual_losses(
#                 params, nt_params,
#                 anchors_a, positives_a, scene_img_a, masks_a)
#             metric_loss, scene_loss = map(float, (metric_loss, scene_loss))

#             progress.set_description(
#                 f"step {step} losses (weighted)"
#                 f" metric {(metric_loss * opts.contrastive_loss_weight):0.5f}"
#                 f" scene  {(scene_loss * opts.classifier_loss_weight):0.5f}")

#             if opts.use_wandb:
#                 wandb_to_log['metric_loss'] = metric_loss
#                 wandb_to_log['scene_loss'] = scene_loss
#                 losses.append((step, metric_loss, scene_loss))

        # if step % 1000 == 0:

        #     train_with_rnd_background_stats = stats(
        #         params, nt_params, opts.eg_train_root_dir, num_egs=100,
        #         random_background_colours=True)
        #     train_without_rnd_background_stats = stats(
        #         params, nt_params, opts.eg_train_root_dir, num_egs=100,
        #         random_background_colours=False)
        #     validation_stats = stats(
        #         params, nt_params, opts.eg_validate_root_dir, num_egs=100,
        #         random_background_colours=False)
        #     print('STATS', step)
        #     print(' train_with_rnd_background_stats', train_with_rnd_background_stats)
        #     print(' train_without_rnd_background_stats', train_without_rnd_background_stats)
        #     print(' validation_stats', validation_stats)

        #     if opts.use_wandb:
        #         wandb_to_log['train_rng_background_stats'] = train_with_rnd_background_stats
        #         wandb_to_log['train_stats'] = train_without_rnd_background_stats
        #         wandb_to_log['validation_stats'] = validation_stats

        #     # generate debug imgs for training with most recent batch
        #     generate_debug_imgs(step, obj_x, scene_x, scene_y_true, split='train')

        #     # grab next batch from validation for validation debug imgs
        #     (obj_x, _obj_y), (scene_x, scene_y_true) = next(validate_ds)
        #     generate_debug_imgs(step, obj_x, scene_x, scene_y_true, split='validate')

        #     # write latest weights
        #     yolz.write_weights(params, nt_params,
        #                        os.path.join(opts.run_dir, 'models_weights.pkl'))

        #     # flush losses
        #     with open(os.path.join(opts.run_dir, 'losses.json'), 'w') as f:
        #         json.dump(losses, f)

        # if len(wandb_to_log) > 0:
        #     wandb.log(step=step, data=wandb_to_log)



