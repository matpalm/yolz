
import os
os.environ['KERAS_BACKEND'] = 'jax'

import json
import os
import tqdm

from util import generate_debug_imgs
#from sklearn.metrics import PrecisionRecallDisplay, precision_recall_curve

# def pr_curve(y_true, y_pred):
#     prec, recall, _ = precision_recall_curve(y_true, y_pred) #, pos_label=clf.classes_[1])
#     return PrecisionRecallDisplay(precision=prec, recall=recall).plot()

if __name__ == '__main__':

    from data import ContrastiveExamples
    from models.yolz import Yolz
    from jax import jit, nn

    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--run-dir', type=str, required=True,
                        help='where to load model config etc')
    parser.add_argument('--weights-pkl', type=str, required=True,
                        help='weights pickle to load')
    #parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--test-dataset-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)

    opts = parser.parse_args()
    print("opts", opts)

    with open(os.path.join(opts.run_dir, 'model_config.json'), 'r') as f:
        models_config = json.load(f)
        print('models_config', models_config)

    yolz = Yolz(
        models_config,
        initial_weights_pkl=opts.weights_pkl,
        contrastive_loss_weight=1,
        classifier_loss_weight=1,
        focal_loss_alpha=1,
        focal_loss_gamma=1
        )

    @jit
    def test_step(anchors_a, scene_img_a):
        params, nt_params = yolz.get_params()
        return yolz.test_step(params, nt_params, anchors_a, scene_img_a)

    test_ds = ContrastiveExamples(
        root_dir=opts.test_dataset_dir,
        seed=123,
        random_background_colours=False,
        instances_per_obj=8,
        cache_dir=None) # TODO: ? '/dev/shm/zero_shot_detection/cache/train')

    #y_true_all = []
    #y_pred_all = []

    total_steps = test_ds.num_scenes()
    with tqdm.tqdm(test_ds.tf_dataset(), total=total_steps) as progress:
        for step, batch in enumerate(progress):
            anchors_a, _positives_a, scene_img_a, y_true  = test_ds.process_batch(*batch)

            _embeddings, y_pred_logits = test_step(anchors_a, scene_img_a)
            y_pred = nn.sigmoid(y_pred_logits.squeeze())

            # dangerous! step=scene only when batch size 1!
            obj_id_to_urdf = test_ds.obj_id_to_urdf_mapping(step)

            generate_debug_imgs(
                anchors_a, scene_img_a, y_true, y_pred,
                step, obj_id_to_urdf,
                img_output_dir=os.path.join(opts.output_dir, 'test_imgs'))

            # TODO: readd these back in before full batch experiment

    #       y_true_all.extend(y_true.flatten())
    #       y_pred_all.extend(y_pred.flatten())

            # y_pred = (y_pred > opts.threshold).astype(float)
            # print({
            #     'p': precision_score(y_true, y_pred),
            #     'r': recall_score(y_true, y_pred),
            #     'f1': f1_score(y_true, y_pred)
            #     })

