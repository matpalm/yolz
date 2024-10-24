import tensorflow as tf
import os
from typing import List
import random
from PIL import Image
import numpy as np
from functools import cache

def load_fname(fname):
    img = Image.open(fname.strip())
    img_a = np.array(img, dtype=float)
    img_a /= 255
    return img_a

class RandomObjIdGenerator(object):
    # provides consistent obj_ids for a ContrastiveExamples &
    # SceneExamples Dataset pair. we need this because we want
    # to have two different datasets that provide examples for the
    # _same_ obj ids each batch

    def __init__(self,
                 obj_ids: List[int],
                 num_objs: int,
                 seed: int):
        self.obj_ids = obj_ids
        self.num_objs = num_objs
        self.rnd = random.Random(seed)

    def next_ids(self):
        self.rnd.shuffle(self.obj_ids)
        return self.obj_ids[:self.num_objs]

class ContrastiveExamples(object):

    def __init__(self,
                 root_dir: str,
                 obj_ids: List[str]
                 ):

        self.root_dir = root_dir

        if obj_ids is None:
            self.obj_ids = os.listdir(root_dir)
            print(f"|obj_ids|={len(self.obj_ids)} read from {root_dir}")
        else:
            self.obj_ids = obj_ids

        self.manifest = {}  # { obj_id: [fnames], ... }

        # mapping from str obj_ids to [0, 1, 2, ... ]
        self.label_idx_to_str = dict(enumerate(self.obj_ids))  # { 0:'053', 1:'234', .... }
        self.label_str_to_idx = {v: k for k, v in self.label_idx_to_str.items()}

    def _random_example_for(self, obj_id) -> str:
        if obj_id not in self.manifest:
            fnames = os.listdir(os.path.join(self.root_dir, obj_id))
            assert len(fnames) > 0, obj_id
            self.manifest[obj_id] = fnames
        fname = random.choice(self.manifest[obj_id])
        return os.path.join(self.root_dir, obj_id, fname)

    @cache
    def _load_fname(self, fname):
        return load_fname(fname)

    def _anc_pos_generator(self, total_examples, num_obj_references):
        for _ in range(total_examples):
            obj_ids = self.rnd_obj_ids.next_ids()
            for _ in range(num_obj_references):    # N
                for obj_id in obj_ids:             # C
                    anc_fname = self._random_example_for(obj_id)
                    anc_img_a = self._load_fname(anc_fname)
                    yield anc_img_a, self.label_str_to_idx[obj_id]
                    pos_fname = anc_fname
                    while pos_fname == anc_fname:
                        pos_fname = self._random_example_for(obj_id)
                    pos_img_a = self._load_fname(pos_fname)
                    yield pos_img_a, self.label_str_to_idx[obj_id]

    def dataset(self, num_batches, batch_size, num_obj_references, num_contrastive_objs, seed):

        if num_contrastive_objs is None:
            num_contrastive_objs = len(self.obj_ids)
            print("derived num_contrastive_objs", num_contrastive_objs)
        elif num_contrastive_objs > len(self.obj_ids):
            raise Exception(f"not enough obj_ids ({len(self.obj_ids)}) to sample"
                            f" num_constrastive_objs ({num_contrastive_objs})")

        self.rnd_obj_ids = RandomObjIdGenerator(
            self.obj_ids, num_contrastive_objs, seed)

        total_examples = num_batches * batch_size
        ds = tf.data.Dataset.from_generator(
            lambda: self._anc_pos_generator(
                total_examples,
                num_obj_references),
            output_signature=(
                tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32),  # img
                tf.TensorSpec(shape=(), dtype=tf.int64),                 # label
            )
        )

        # inner batch for the contrastive pairs
        # (2C, HW, HW, 3)
        ds = ds.batch(2*num_contrastive_objs)

        # batch again for the N reference examples
        # (N, 2C, HW, HW, 3). but ONLY IF num_obj_references > 1
        if num_obj_references > 1:
            ds = ds.batch(num_obj_references)

        # final batch for B
        # (B, N, 2C, HW, HW, 3)
        ds = ds.batch(batch_size)

        return ds

class SceneExamples(object):

    def __init__(self,
                root_dir: str,
                obj_ids: List[str]
                ):

        self.root_dir = root_dir

        if obj_ids is None:
            self.obj_ids = os.listdir(root_dir)
            print(f"|obj_ids|={len(self.obj_ids)} read from {root_dir}")
        else:
            self.obj_ids = obj_ids

        self.manifest = {}  # { obj_id: [fnames], ... }

        # mapping from str obj_ids to [0, 1, 2, ... ]
        self.label_idx_to_str = dict(enumerate(self.obj_ids))  # { 0:'053', 1:'234', .... }
        self.label_str_to_idx = {v: k for k, v in self.label_idx_to_str.items()}

    def _random_example_for(self, obj_id) -> str:
        if obj_id not in self.manifest:
            fnames = os.listdir(os.path.join(self.root_dir, obj_id))
            assert len(fnames) > 0, obj_id
            self.manifest[obj_id] = fnames
        fname = random.choice(self.manifest[obj_id])
        return os.path.join(self.root_dir, obj_id, fname)

    @cache
    def _load_fname(self, fname):
        return load_fname(fname)

    def _scene_generator(self, total_examples):
        for _ in range(total_examples):
            obj_ids = self.rnd_obj_ids.next_ids()
            for obj_id in obj_ids:             # C
                # generate a scene with this obj_id, as well as other
                # distractor objects of any other obj_id
                yield np.ones((1,1,3)), self.label_str_to_idx[obj_id]

                # anc_fname = self._random_example_for(obj_id)
                # anc_img_a = self._load_fname(anc_fname)
                # yield anc_img_a, self.label_str_to_idx[obj_id]
                # pos_fname = anc_fname
                # while pos_fname == anc_fname:
                #     pos_fname = self._random_example_for(obj_id)
                # pos_img_a = self._load_fname(pos_fname)
                # yield pos_img_a, self.label_str_to_idx[obj_id]

    def dataset(self, num_batches, batch_size, num_objs, seed):

        if num_objs is None:
            num_objs = len(self.obj_ids)
            print("derived num_objs", num_objs)
        elif num_objs > len(self.obj_ids):
            raise Exception(f"not enough obj_ids ({len(self.obj_ids)}) to sample"
                            f" num_constrastive_objs ({num_objs})")

        self.rnd_obj_ids = RandomObjIdGenerator(
            self.obj_ids, num_objs, seed)

        total_examples = num_batches * batch_size
        ds = tf.data.Dataset.from_generator(
            lambda: self._scene_generator(
                total_examples),
            output_signature=(
                tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32),  # img
                tf.TensorSpec(shape=(), dtype=tf.int64),                 # label
            )
        )

        # first batch by class
        # (C, HW, HW, 3)
        ds = ds.batch(num_objs)

        # final batch for B
        # (B, C, HW, HW, 3)
        ds = ds.batch(batch_size)

        return ds

if __name__ == '__main__':
    c_egs = ContrastiveExamples(
        root_dir='data/train/reference_patches/',
        obj_ids=["061","135","182",  # x3 red
                 "111","153","198",  # x3 green
                 "000","017","019"], # x3 blue
    )
    ds = c_egs.dataset(num_batches=1,
                       batch_size=4,            # B
                       num_obj_references=1,    # N
                       num_contrastive_objs=3,  # C
                       seed=123)
    for x, y in ds:
        print(x.shape, y)

    s_egs = SceneExamples(
        root_dir='data/train/reference_patches/',
        obj_ids=["061","135","182",  # x3 red
                 "111","153","198",  # x3 green
                 "000","017","019"], # x3 blue
    )
    ds = s_egs.dataset(num_batches=1,
                       batch_size=4,  # B
                       num_objs=3,    # C
                       seed=123)
    for x, y in ds:
        print(x.shape, y)

    # (B=4, N=1, 64, 64, 3) -> (5, 3, [ap], 64, 64, 3)
    # tf.Tensor(
    #    [[6 6 3 3 2 2]
    #     [3 3 7 7 1 1]
    #     [5 5 6 6 2 2]
    #     [2 2 8 8 5 5]
    #     [1 1 2 2 7 7]], shape=(5, 6), dtype=int64)


