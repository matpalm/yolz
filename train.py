import os
os.environ['KERAS_BACKEND'] = 'jax'

import json, tqdm

from jax import jit, nn
import optax

from data import ContrastiveExamples
from models.yolz import Yolz
from util import create_dir_if_required, generate_debug_imgs

import numpy as np
np.set_printoptions(precision=5, threshold=10000, suppress=True, linewidth=10000)

from sklearn.metrics import PrecisionRecallDisplay, precision_recall_curve
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc

import wandb


# # from jaxopt
# from jax.nn import softplus
# def binary_logistic_loss(label: int, logit: float) -> float:
#   return softplus(jnp.where(label, -logit, logit))

import argparse
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--run-dir', type=str, required=True,
                    help='where to store weights, losses.json, examples etc')
parser.add_argument('--train-root-dir', type=str, default='data/train/')
parser.add_argument('--validate-root-dir', type=str, default='data/validation/')
parser.add_argument('--num-repeats', type=int, default=1)
parser.add_argument('--initial-weights-pkl', type=str, default=None,
                    help='starting weights')
parser.add_argument('--optimiser', type=str, default='adam')
parser.add_argument('--learning-rate', type=float, default=1e-3,
                    help='adam learning rate')
parser.add_argument('--contrastive-loss-weight', type=float, default=1.0)
parser.add_argument('--classifier-loss-weight', type=float, default=100.0)
parser.add_argument('--focal-loss-alpha', type=float, default=0.5)
parser.add_argument('--focal-loss-gamma', type=float, default=2.0)
parser.add_argument('--use-wandb', action='store_true')
parser.add_argument('--seed', type=int, default=123)

opts = parser.parse_args()
print("opts", opts)

models_config = {
    'embedding': {'height_width': 64,
                  'filter_sizes': [16, 32, 64, 256],
                  'embedding_dim': 256},
    'scene': {'height_width': 640,
              'filter_sizes': [16, 32, 64, 128, 256, 256],
              'feature_dim': 256,
              'classifier_filter_sizes': [256, 128],
              'init_classifier_bias': -6,
              'mixing_strategy': 'elementwise_add'}
}
with open(os.path.join(opts.run_dir, 'model_config.json'), 'w') as f:
    json.dump(models_config, f)

if opts.use_wandb:
    wandb.init(project='yolz',
                name=opts.run_dir,
                reinit=True)
    wandb.config['model_version'] = 'v4'
    for k, v in vars(opts).items():
        wandb.config[k] = v
    for k, v in models_config.items():  # wandb will unpack to embedding.embedding_dim etc
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
# yolz.embedding_model.summary()
# yolz.scene_model.summary()

train_ds = ContrastiveExamples(
    root_dir=opts.train_root_dir,
    seed=123,
    random_background_colours=True,
    instances_per_obj=8,
    cache_dir=None)

validate_ds = ContrastiveExamples(
    root_dir=opts.validate_root_dir,
    seed=123,
    random_background_colours=False,
    instances_per_obj=8,
    cache_dir=None)

# set up training step & optimiser
if opts.optimiser == 'adam':
    opt = optax.adam(learning_rate=opts.learning_rate)
elif opts.optimiser == 'adamw':
    opt = optax.adamw(learning_rate=opts.learning_rate)
elif opts.optimiser == 'sgd':
    opt = optax.sgd(learning_rate=opts.learning_rate, momentum=0.9)
else:
    raise Exception(f"unsupported --optimiser {opts.optimiser}")
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

total_steps = train_ds.num_scenes() * opts.num_repeats
print(f"|train scenes|={train_ds.num_scenes()}, repeats={opts.num_repeats}"
      f" => total_steps={total_steps}")

with tqdm.tqdm(train_ds.tf_dataset(opts.num_repeats), total=total_steps) as progress:
    for step, batch in enumerate(progress):

        wandb_to_log = {}

        anchors_a, positives_a, scene_img_a, masks_a = train_ds.process_batch(*batch)

        params, nt_params, opt_state, loss = train_step(
            params, nt_params,
            opt_state,
            anchors_a, positives_a, scene_img_a, masks_a)

        if step % 100 == 0:
            metric_loss, scene_loss, _ = calculate_individual_losses(
                params, nt_params,
                anchors_a, positives_a, scene_img_a, masks_a)

            progress.set_description(f"metric_loss={metric_loss} scene_loss={scene_loss}")
            wandb_to_log['metric_loss'] = metric_loss
            wandb_to_log['scene_loss'] = scene_loss

        if step % 2500 == 0:

            # write weights
            create_dir_if_required(os.path.join(opts.run_dir, 'models_weights'))
            yolz.write_weights(
                params, nt_params,
                os.path.join(opts.run_dir, 'models_weights', f"s_{step:08d}.pkl"))

            @jit
            def test_step(anchors_a, scene_img_a):
                _anchor_embeddings, y_pred_logits = yolz.test_step(
                    params, nt_params, anchors_a, scene_img_a)
                return nn.sigmoid(y_pred_logits).squeeze()

            # run most recent test example through debugging images
            y_pred = test_step(anchors_a, scene_img_a)

            # TODO: add back in
            # generate_debug_imgs(
            #     anchors_a, scene_img_a, masks_a, y_pred,
            #     step,
            #     img_output_dir=os.path.join(opts.run_dir, 'debug_imgs'))

            # collect all validation data
            y_true = np.array([])
            y_pred = np.array([])
            flattened = lambda a: np.array(a).flatten()
            for batch in validate_ds.tf_dataset():
                anchors_a, _positives_a, scene_img_a, masks_a = validate_ds.process_batch(*batch)
                y_true = np.append(y_true, flattened(masks_a))
                y_pred = np.append(y_pred, flattened(test_step(anchors_a, scene_img_a)))
            prec, recall, _ = precision_recall_curve(y_true, y_pred) #, pos_label=clf.classes_[1])
            display = PrecisionRecallDisplay(precision=prec, recall=recall)
            display.plot()
            create_dir_if_required(os.path.join(opts.run_dir, 'p_r_curves'))
            display.figure_.savefig(os.path.join(opts.run_dir, 'p_r_curves', f"s_{step:08d}.png"))

            # calculation some p/r/f1 stats. log the 0.5 case
            wandb_to_log['validation'] = {}
            for threshold in [0.25, 0.5, 0.75]:
                y_pred_t = (y_pred >= threshold).astype(float)
                precision = precision_score(y_true, y_pred_t, zero_division=0)
                recall = recall_score(y_true, y_pred_t)
                f1 = f1_score(y_true, y_pred_t)
                print(f"threshold {threshold:.2f} precision {precision:.2f} recall {recall:.2f} f1 {f1:.2f}")
                if threshold == 0.5:
                    wandb_to_log['validation']['precision@0.5'] = precision
                    wandb_to_log['validation']['recall@0.5'] = recall
                    wandb_to_log['validation']['f1@0.5'] = f1
            # also log overall AUC ( though this seems way too high??? )
            fpr, tpr, _thresholds = roc_curve(y_true, y_pred)
            validation_auc = auc(fpr, tpr)
            print('validation auc', validation_auc)
            wandb_to_log['validation']['auc'] = validation_auc

        if opts.use_wandb and len(wandb_to_log) > 0:
            wandb.log(step=step, data=wandb_to_log)

print(opts.run_dir)

    # if step > 3000:
    #     break

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



