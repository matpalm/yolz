
import os
os.environ['KERAS_BACKEND'] = 'jax'

import json
import os

from util import array_to_pil_img, mask_to_pil_img, alpha_from_mask, collage, create_dir_if_required
from sklearn.metrics import PrecisionRecallDisplay, precision_recall_curve

def pr_curve(y_true, y_pred):
    prec, recall, _ = precision_recall_curve(y_true, y_pred) #, pos_label=clf.classes_[1])
    return PrecisionRecallDisplay(precision=prec, recall=recall).plot()

if __name__ == '__main__':

    from data import ContrastiveExamples
    from models.yolz import Yolz
    from jax import jit, nn

    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--run-dir', type=str, required=True,
                        help='where to store weights, losses.json, examples etc')
    # parser.add_argument('--validate-root-dir', type=str,
    #                     default='data/validate/reference_patches',
    #                     help='.')
    parser.add_argument('--threshold', type=float, default=0.5)

    opts = parser.parse_args()
    print("opts", opts)

    with open(os.path.join(opts.run_dir, 'model_config.json'), 'r') as f:
        models_config = json.load(f)

    yolz = Yolz(
        models_config,
        initial_weights_pkl=os.path.join(opts.run_dir, 'models_weights.pkl'),
        contrastive_loss_weight=1,
        classifier_loss_weight=1,
        focal_loss_alpha=1,
        focal_loss_gamma=1
        )

    @jit
    def test_step(anchors_a, scene_img_a):
        params, nt_params = yolz.get_params()
        return yolz.test_step(params, nt_params, anchors_a, scene_img_a)

    validate_ds = ContrastiveExamples(
        root_dir='/dev/shm/zero_shot_detection/data/train/',
        seed=123,
        random_background_colours=False,
        instances_per_obj=8,
        cache_dir='/dev/shm/zero_shot_detection/cache/train')

    y_true_all = []
    y_pred_all = []

    for step, batch in enumerate(validate_ds.tf_dataset(num_repeats=1)):
        anchors_a, _positives_a, scene_img_a, y_true  = validate_ds.process_batch(*batch)
        #y_true = np.array(y_true)

        _embeddings, y_pred_logits = test_step(anchors_a, scene_img_a)
        y_pred = nn.sigmoid(y_pred_logits.squeeze())

        # debug some reference examples
        img_output_dir = os.path.join(opts.run_dir, 'test_imgs')
        create_dir_if_required(img_output_dir)
        for obj_idx in range(len(anchors_a)):
            num_egs = anchors_a.shape[1]
            ref_images = [array_to_pil_img(anchors_a[obj_idx, a]) for a in range(num_egs)]
            c = collage(ref_images, rows=2, cols=4)
            c.save(f"{img_output_dir}/step_{step:06d}.obj_{obj_idx}.anchors.png")
            y_true_m = mask_to_pil_img(y_true[obj_idx])
            y_pred_m = mask_to_pil_img(y_pred[obj_idx])
            c = collage([y_true_m, y_pred_m], rows=1, cols=2)
            c.save(f"{img_output_dir}/step_{step:06d}.obj_{obj_idx}.y_true_pred.mask.png")
            scene_rgb = array_to_pil_img(scene_img_a[0])
            scene_with_y_pred = alpha_from_mask(scene_rgb, y_pred_m)
            scene_with_y_true = alpha_from_mask(scene_rgb, y_true_m)
            c = collage([scene_with_y_pred, scene_rgb, scene_with_y_true], rows=1, cols=3)
            c.save(f"{img_output_dir}/step_{step:06d}.obj_{obj_idx}.y_true_pred.alpha.png")

        y_true_all.extend(y_true.flatten())
        y_pred_all.extend(y_pred.flatten())

        # y_pred = (y_pred > opts.threshold).astype(float)
        # print({
        #     'p': precision_score(y_true, y_pred),
        #     'r': recall_score(y_true, y_pred),
        #     'f1': f1_score(y_true, y_pred)
        #     })

        if step > 10: break