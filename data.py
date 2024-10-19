import tensorflow as tf
import os
from typing import List
import random
from PIL import Image
import numpy as np

class ConstrastiveExamples(object):

    def __init__(self,
                 root_dir: str,
                 obj_ids: List[str]
                 ):

        self.root_dir = root_dir
        self.obj_ids = obj_ids
        self.manifest = {}  # { obj_id: [fnames], ... }

        # mapping from str obj_ids to [0, 1, 2, ... ]
        self.label_idx_to_str = dict(enumerate(obj_ids))  # { 0:'053', 1:'234', .... }
        self.label_str_to_idx = {v: k for k, v in self.label_idx_to_str.items()}
        print("self.label_idx_to_str", self.label_idx_to_str)

    def _random_example_for(self, obj_id) -> str:
        if obj_id not in self.manifest:
            fnames = os.listdir(os.path.join(self.root_dir, obj_id))
            assert len(fnames) > 0, obj_id
            self.manifest[obj_id] = fnames
        fname = random.choice(self.manifest[obj_id])
        return os.path.join(self.root_dir, obj_id, fname)

    def _load_fname(self, fname):
        img = Image.open(fname)
        img_a = np.array(img)
        return img_a

    def _anc_pos_generator(self, batch_size, objs_per_batch):
        for _ in range(batch_size):
            random.shuffle(self.obj_ids)
            for obj_id in self.obj_ids[:objs_per_batch]:

                anc_fname = self._random_example_for(obj_id)
                anc_img_a = self._load_fname(anc_fname)
                yield anc_img_a, self.label_str_to_idx[obj_id]

                pos_fname = anc_fname
                while pos_fname == anc_fname:
                    pos_fname = self._random_example_for(obj_id)
                pos_img_a = self._load_fname(pos_fname)
                yield pos_img_a, self.label_str_to_idx[obj_id]

    def dataset(self, batch_size, objs_per_batch):

        if objs_per_batch > len(self.obj_ids):
            raise Exception(f"not enough obj_ids ({len(self.obj_ids)}) to sample"
                            f" objs_per_batch ({objs_per_batch})")

        ds = tf.data.Dataset.from_generator(
            lambda: self._anc_pos_generator(batch_size, objs_per_batch),
            output_signature=(
                tf.TensorSpec(shape=(None, None, 3), dtype=tf.uint8),  # img
                tf.TensorSpec(shape=(), dtype=tf.int64),               # label
            )
        )

        # each sub batch is anc & pos pair for each obj
        # select ith object pair as x.reshape(-1, 2)[i]
        # ((2C, HW, HW, 3), (2C))
        ds = ds.batch(2*objs_per_batch)

        # batch again for _actual_ batch
        # ((B, 2C, HW, HW, 3), (B, 2C))
        ds = ds.batch(batch_size)

        return ds


if __name__ == '__main__':
    c_egs = ConstrastiveExamples(
        root_dir='data/reference_egs',
        obj_ids=["061","135","182",  # x3 red
                 "111","153","198",  # x3 green
                 "000","017","019"], # x3 blue
    )
    ds = c_egs.dataset(batch_size=5,
                       objs_per_batch=3)
    for x, y in ds:
        print(x.shape, y)

    # (5, 6, 64, 64, 3) -> (5, 3, [ap], 64, 64, 3)
    # tf.Tensor(
    #    [[6 6 3 3 2 2]
    #     [3 3 7 7 1 1]
    #     [5 5 6 6 2 2]
    #     [2 2 8 8 5 5]
    #     [1 1 2 2 7 7]], shape=(5, 6), dtype=int64)


