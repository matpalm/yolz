import tensorflow as tf
from tensorflow_datasets import as_numpy
import os
import random
from PIL import Image
import numpy as np
import json

import util

def load_fname(fname, background_colour=None):
    try:
        img = Image.open(fname.strip())
        if background_colour is None:
            # ensure alpha, if present, is removed
            return img.convert('RGB')
        background = Image.new('RGB', img.size, background_colour)
        background.paste(img, (0, 0), img)
        return background
    except AssertionError as ae:
        print("load_fname AssertionError?", str(ae))
        print('fname', fname)
        exit(-1)

class ContrastiveExamples(object):

    def __init__(self,
                 scene_root_dir: str,
                 reference_root_dir: str,
                 seed: int,
                 random_background_colours: bool,
                 total_num_scenes: int,
                 instances_per_obj: int  # N
                 ):
        # ['061', '002', '000']  # red, green, blue eg
        self.scene_root_dir = scene_root_dir
        self.reference_root_dir = reference_root_dir
        self.scenes = os.listdir(scene_root_dir)
        self.rnd = random.Random(seed)
        self.random_background_colours = random_background_colours
        self.total_num_scenes = total_num_scenes
        self.instances_per_obj = instances_per_obj
        self.manifest = {}  # { obj_id: [fnames], ... }
        # mapping from str obj_ids to [0, 1, 2, ... ]
        #self.label_idx_to_str = dict(enumerate(self.obj_ids))  # { 0:'053', 1:'234', .... }
        #self.label_str_to_idx = {v: k for k, v in self.label_idx_to_str.items()}

    def _random_example_for(self, obj_id) -> str:
        obj_id = f"{obj_id:03d}"
        if obj_id not in self.manifest:
            fnames = os.listdir(os.path.join(self.reference_root_dir, obj_id))
            assert len(fnames) > 0, obj_id
            self.manifest[obj_id] = fnames
        fname = self.rnd.choice(self.manifest[obj_id])
        return os.path.join(self.reference_root_dir, obj_id, fname)

    def _load(self, fname):
        if self.random_background_colours:
            r = self.rnd.randint(0, 255)
            g = self.rnd.randint(0, 255)
            b = self.rnd.randint(0, 255)
            return load_fname(fname, background_colour=(r, g, b))
        else:
            return load_fname(fname, background_colour=(128, 128, 128))

    def _scene_and_anc_pos_generator(self):
        for _ in range(self.total_num_scenes):

            # pick a new random scene
            scene_id = self.rnd.choice(self.scenes)

            # load the scene RGB ( and convert to np array )
            # and, since images need to be batched, batch as single
            # element
            # ( 1, sW, SH, 3 )
            scene_img = load_fname(os.path.join(
                self.scene_root_dir, scene_id, 'rgb.png'))
            scene_img_a = np.array(scene_img)
            scene_img_a = np.expand_dims(scene_img, 0)

            # load the segmasks  ( C, mW, mH )
            masks_a = np.load(os.path.join(
                self.scene_root_dir, scene_id, 'masks.npy'))

            # load the obj_ids in this scene
            # ( these correspond to the rows in the mask )
            with open(os.path.join(
                self.scene_root_dir, scene_id, 'obj_ids.json'), 'r') as f:
                obj_ids = json.load(f)
            if len(masks_a) != len(obj_ids):
                raise Exception("mask vs obj_ids mismatch for scene", scene_id)

            # load N instances for each obj anchor
            # and another N instance for obj positive
            # ( all distinct )
            all_anc_pos_a = []
            for obj_id in obj_ids:
                anc_pos_ids = set()
                while len(anc_pos_ids) < 2 * self.instances_per_obj:
                    anc_pos_ids.add(self._random_example_for(obj_id))
                # map from fnames to np arrays  ( 2N, oH, oW, 3 )
                anc_pos_imgs_a = np.stack([self._load(f) for f in anc_pos_ids])
                # add to all_ sets
                all_anc_pos_a.append(anc_pos_imgs_a)
            # final stack all_ pairs  ( C, 2N, oH, oW, 3 )
            all_anc_pos_a = np.stack(all_anc_pos_a)
            # slice out anchors from positives
            # by first reshaping to ( C, 2, N, oH, oW, 3 )
            hwc_shape = all_anc_pos_a.shape[-3:]
            all_anc_pos_a = np.reshape(
                all_anc_pos_a,
                (len(obj_ids), 2, self.instances_per_obj, *hwc_shape))
            # then slicing the two out
            anchors_a = all_anc_pos_a[:, 0]
            positives_a = all_anc_pos_a[:, 1]

            yield scene_img_a, masks_a, anchors_a, positives_a

    def tf_dataset(self):
        N = self.instances_per_obj
        ds = tf.data.Dataset.from_generator(
            lambda: self._scene_and_anc_pos_generator(),
            output_signature=(
                tf.TensorSpec(shape=(1, None, None, 3), dtype=tf.uint8),        # scene     (1, sH, sW, 3)     (0, 255)
                tf.TensorSpec(shape=(None, None, None), dtype=tf.uint8),        # masks     (C, mH, mW)        {0, 1}
                tf.TensorSpec(shape=(None, N, None, None, 3), dtype=tf.uint8),  # anchors   (C, N, oH, oW, 3)  (0, 255)
                tf.TensorSpec(shape=(None, N, None, None, 3), dtype=tf.uint8),  # positives (C, N, oH, oW, 3)  (0, 255)
            )
        )
        return ds.prefetch(tf.data.AUTOTUNE)


if __name__ == '__main__':
    ce = ContrastiveExamples(
        scene_root_dir = 'data/train/scenes',
        reference_root_dir = 'data/train/reference_patches',
        seed = 123,
        random_background_colours = False,
        total_num_scenes = 3,
        instances_per_obj = 9
    )
    def info(a):
        return f"{a.shape} {a.dtype} ({np.min(a)}, {np.max(a)})"
    for scene_img_a, masks_a, anchors_a, positives_a in as_numpy(ce.tf_dataset()):
        print('scene_img_a', info(scene_img_a))   # (1, 640, 640, 3)
        print('masks_a', info(masks_a))           # (16, 80, 80)
        print('anchors_a', info(anchors_a))       # (16, 9, 64, 64, 3)
        print('positives_a', info(positives_a))   # (16, 9, 64, 64, 3)
        break

    C = masks_a.shape[0]
    assert len(anchors_a) == C
    assert len(positives_a) == C
    N = anchors_a.shape[1]
    assert positives_a.shape[1] == N

    util.create_dir_if_required('data_egs')

    NUM_EG_OBJS = 6

    scene_img = util.to_pil_img(scene_img_a[0])
    scene_img.save(f"data_egs/scene.png")

    for obj_idx in range(NUM_EG_OBJS):
        mask_img = Image.fromarray(np.array(masks_a[obj_idx]), 'L')
        mask_img = mask_img.resize(scene_img.size, Image.Resampling.NEAREST)
        mask_a = np.expand_dims(np.array(mask_img), -1)
        mask_a *= 255
        masked_img_a = np.concatenate([scene_img_a[0], mask_a], axis=-1)  # (640, 640, 4)
        masked_img = Image.fromarray(masked_img_a, 'RGBA')
        masked_img.save(f"data_egs/mask.obj_id{obj_idx}.png")

    for obj_idx in range(NUM_EG_OBJS):
        anchors = [util.to_pil_img(a) for a in anchors_a[obj_idx]]
        positives = [util.to_pil_img(p) for p in positives_a[obj_idx]]
        util.collage(anchors+positives, 2, N).save(f"data_egs/anc_pos.obj_id{obj_idx}.png")
