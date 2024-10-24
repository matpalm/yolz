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


class ObjIdsHelper(object):

    def __init__(self,
                 root_dir: str,
                 obj_ids: List[str],
                 seed: int
    ):
        self.root_dir = root_dir
        self.seed = seed
        self.rnd = random.Random(seed)

        if obj_ids is None:
            self.obj_ids = os.listdir(root_dir)
            print(f"|obj_ids|={len(self.obj_ids)} read from {root_dir}")
        else:
            self.obj_ids = obj_ids

        self.manifest = {}  # { obj_id: [fnames], ... }

        # mapping from str obj_ids to [0, 1, 2, ... ]
        self.label_idx_to_str = dict(enumerate(self.obj_ids))  # { 0:'053', 1:'234', .... }
        self.label_str_to_idx = {v: k for k, v in self.label_idx_to_str.items()}

    def random_example_for(self, obj_id) -> str:
        if obj_id not in self.manifest:
            fnames = os.listdir(os.path.join(self.root_dir, obj_id))
            assert len(fnames) > 0, obj_id
            self.manifest[obj_id] = fnames
        fname = self.rnd.choice(self.manifest[obj_id])
        return os.path.join(self.root_dir, obj_id, fname)

    @cache
    def load_fname(self, fname):
        return load_fname(fname)

    def check_or_derive(self, num_objs):
        if num_objs is None:
            num_objs = len(self.obj_ids_helper.obj_ids)
            print("derived num_objs", num_objs)
        elif num_objs > len(self.obj_ids):
            raise Exception(f"not enough obj_ids ({len(self.obj_ids)}) to sample"
                            f" num_objs. ({num_objs})")
        return num_objs

    def construct_random_obj_id_generator(self, num_objs):
        return RandomObjIdGenerator(self.obj_ids, num_objs, self.seed)


class ContrastiveExamples(object):

    def __init__(self, root_dir: str, obj_ids: List[str], seed):
        self.obj_ids_helper = ObjIdsHelper(root_dir, obj_ids, seed)

    def _anc_pos_generator(self, num_batches, num_obj_references):
        for b in range(num_batches):
            obj_ids = self.rnd_obj_ids.next_ids()
            for obj_id in obj_ids:  # C
                label_idx = self.obj_ids_helper.label_str_to_idx[obj_id]
                for _ in range(num_obj_references):    # N
                    anc_fname = self.obj_ids_helper.random_example_for(obj_id)
                    anc_img_a = self.obj_ids_helper.load_fname(anc_fname)
                    yield anc_img_a, label_idx
                # TODO: pos should be strictly different to anchors
                #       at the moment this is just 2x num_obj_references examples
                for _ in range(num_obj_references):    # N
                    pos_fname = self.obj_ids_helper.random_example_for(obj_id)
                    pos_img_a = self.obj_ids_helper.load_fname(pos_fname)
                    yield pos_img_a, label_idx

    def dataset(self, num_batches, num_obj_references, num_contrastive_examples):

        num_contrastive_examples = self.obj_ids_helper.check_or_derive(num_contrastive_examples)
        print("num_contrastive_examples", num_contrastive_examples)

        self.rnd_obj_ids = self.obj_ids_helper.construct_random_obj_id_generator(
            num_contrastive_examples)

        ds = tf.data.Dataset.from_generator(
            lambda: self._anc_pos_generator(
                num_batches,
                num_obj_references),
            output_signature=(
                tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32),  # img
                tf.TensorSpec(shape=(), dtype=tf.int64),                 # label
            )
        )

        # inner batch is the N reference examples
        # (N, HW, HW, 3)
        ds = ds.batch(num_obj_references)

        # outer batch for the contrastive pairs
        # (2C, N, HW, HW, 3)
        ds = ds.batch(2*num_contrastive_examples)

        # dataset will return num_batches instances of (2C, N, HW, HW, 3)
        return ds

class SceneExamples(object):

    def __init__(self,
                 root_dir: str,
                 obj_ids: List[str],
                 seed: int,
                 grid_size: int):

        self.obj_ids_helper = ObjIdsHelper(root_dir, obj_ids, seed)
        self.obj_ids = obj_ids
        self.seed = seed
        self.grid_size = grid_size

    def x_y_ids_for_example(
            self,
            obj_id: int,            # primary object id
            num_other_objs: int,    # how many other objects to include
            instances_per_obj: int,  # how many instance of each obj_id
    ):
        rnd = random.Random(self.seed)

        # decide set of obj ids; seeded by obj_id
        eg_obj_ids = set([obj_id])
        while len(eg_obj_ids) < num_other_objs+1:
            candidate_id = rnd.choice(self.obj_ids)
            eg_obj_ids.add(candidate_id)
        eg_obj_ids = list(eg_obj_ids)

        # create generator for distinct x, y coords
        def distinct_xy_generator():
            yielded = set()
            while True:
                xy = (rnd.randint(0, self.grid_size-1),
                      rnd.randint(0, self.grid_size-1))
                if xy not in yielded:
                    yield xy
                    yielded.add(xy)
        distinct_xy = distinct_xy_generator()

        # generate x, y, obj_id examples
        for obj_id in eg_obj_ids:
            for _ in range(instances_per_obj):
                yield *next(distinct_xy), obj_id





#     def _scene_generator(self, total_examples):
#         for _ in range(total_examples):
#             obj_ids = self.rnd_obj_ids.next_ids()
#             for obj_id in obj_ids:             # C
#                 # generate a scene with this obj_id, as well as other
#                 # distractor objects of any other obj_id
#                 yield np.ones((1,1,3)), self.obj_ids_helper.label_str_to_idx[obj_id]

#                 # anc_fname = self._random_example_for(obj_id)
#                 # anc_img_a = self._load_fname(anc_fname)
#                 # yield anc_img_a, self.label_str_to_idx[obj_id]
#                 # pos_fname = anc_fname
#                 # while pos_fname == anc_fname:
#                 #     pos_fname = self._random_example_for(obj_id)
#                 # pos_img_a = self._load_fname(pos_fname)
#                 # yield pos_img_a, self.label_str_to_idx[obj_id]

#     def dataset(self, num_batches, batch_size, num_objs):

#         num_objs = self.obj_ids_helper.check_or_derive(num_objs)

#         self.rnd_obj_ids = self.obj_ids_helper.construct_random_obj_id_generator(
#             num_objs)

#         total_examples = num_batches * batch_size
#         ds = tf.data.Dataset.from_generator(
#             lambda: self._scene_generator(
#                 total_examples),
#             output_signature=(
#                 tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32),  # img
#                 tf.TensorSpec(shape=(), dtype=tf.int64),                 # label
#             )
#         )

#         # first batch by class
#         # (C, HW, HW, 3)
#         ds = ds.batch(num_objs)

#         # final batch for B
#         # (B, C, HW, HW, 3)
#         ds = ds.batch(batch_size)

#         return ds

if __name__ == '__main__':

    c_egs = ContrastiveExamples(
        root_dir='data/train/reference_patches/',
        obj_ids=["061","135","182",  # x3 red
                 "111","153","198",  # x3 green
                 "000","017","019"], # x3 blue
        seed=123
    )
    ds = c_egs.dataset(num_batches=2,
                       num_obj_references=2,        # N
                       num_contrastive_examples=3,  # C
                       )
    from util import to_pil_img, collage

    for b, (x, y) in enumerate(ds):
        # (2xC=6, N=2, HW, HW, 3)
        print("batch!", b, x.shape )#y)
        imgs = list(map(to_pil_img, x[:,0])) # N=0
        collage(imgs, 3, 2).save(f"data.batch_{b}.n_0.png")
        imgs = list(map(to_pil_img, x[:,1])) # N=1
        collage(imgs, 3, 2).save(f"data.batch_{b}.n_1.png")


    s_egs = SceneExamples(
        root_dir='data/train/reference_patches/',
        obj_ids=["061","135","182",  # x3 red
                 "111","153","198",  # x3 green
                 "000","017","019"], # x3 blue
        seed=123,
        grid_size=10,
    )

    print(list(
        s_egs.x_y_ids_for_example(
            obj_id='061', num_other_objs=3, instances_per_obj=5)))

    # ds = s_egs.dataset(num_batches=1,
    #                    batch_size=4,  # B
    #                    num_objs=3,    # C
    #                    )
    # for x, y in ds:
    #     print(x.shape, y)

    # (B=4, N=1, 64, 64, 3) -> (5, 3, [ap], 64, 64, 3)
    # tf.Tensor(
    #    [[6 6 3 3 2 2]
    #     [3 3 7 7 1 1]
    #     [5 5 6 6 2 2]
    #     [2 2 8 8 5 5]
    #     [1 1 2 2 7 7]], shape=(5, 6), dtype=int64)


