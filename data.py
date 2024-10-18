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

    def _anc_pos_generator(self, n):
        for _ in range(n):
            for obj_id in self.obj_ids:
                anc_fname = self._random_example_for(obj_id)
                anc_img_a = self._load_fname(anc_fname)
                yield anc_img_a, self.label_str_to_idx[obj_id]

                pos_fname = anc_fname
                while pos_fname == anc_fname:
                    pos_fname = self._random_example_for(obj_id)
                pos_img_a = self._load_fname(pos_fname)
                yield pos_img_a, self.label_str_to_idx[obj_id]

    def dataset(self, batch_size):
        # for x, y in self._anc_pos_generator():
        #     print(x.shape, y)
        # return

        ds = tf.data.Dataset.from_generator(
            lambda: self._anc_pos_generator(batch_size),
            output_signature=(
                tf.TensorSpec(shape=(None, None, 3), dtype=tf.uint8),  # img
                tf.TensorSpec(shape=(), dtype=tf.int64),               # label
            )
        )

        # each sub batch is anc & pos for each obj
        # select ith object pair as x.reshape(-1, 2)[i]
        ds = ds.batch(2*len(self.obj_ids))

        # batch again for actual batch
        ds = ds.batch(batch_size)
        return ds


if __name__ == '__main__':
    c_egs = ConstrastiveExamples(
        root_dir='data/reference_egs',
        obj_ids=['061', '111', '000'],  # simple R, G, B examples
    )
    ds = c_egs.dataset(batch_size=5)
    for x, y in ds:
        print(x.shape, y)

