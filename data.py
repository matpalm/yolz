import tensorflow as tf
import os
from typing import List, Tuple
import random
from PIL import Image
import numpy as np
from functools import cache
import copy

def convert_dtype(pil_img):
    return np.array(pil_img, dtype=float) / 255

def load_fname(fname):
    try:
        return Image.open(fname.strip())
    except AssertionError as ae:
        print("load_fname AssertionError?", str(ae))
        print('fname', fname)
        exit(-1)

class RandomObjIdGenerator(object):
    # provides consistent obj_ids for a ContrastiveExamples &
    # SceneExamples Dataset pair. we need this because we want
    # to have two different datasets that provide examples for the
    # _same_ obj ids each batch

    def __init__(self,
                 obj_ids: List[int],
                 num_objs: int,
                 seed: int
    ):
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

    # def check_or_derive(self, num_objs):
    #     if num_objs is None:
    #         num_objs = len(self.obj_ids_helper.obj_ids)
    #         print("derived num_objs", num_objs)
    #     elif num_objs > len(self.obj_ids):
    #         raise Exception(f"not enough obj_ids ({len(self.obj_ids)}) to sample"
    #                         f" num_objs. ({num_objs})")
    #     return num_objs

    def construct_random_obj_id_generator(self, num_objs):
        return RandomObjIdGenerator(self.obj_ids, num_objs, self.seed)


class ContrastiveExamples(object):

    def __init__(self, obj_ids_helper):
        self.obj_ids_helper = copy.deepcopy(obj_ids_helper)

    def _anc_pos_generator(self, num_batches, num_obj_references):
        for _ in range(num_batches):
            obj_ids = self.rnd_obj_ids.next_ids()
            for obj_id in obj_ids:  # C
                label_idx = self.obj_ids_helper.label_str_to_idx[obj_id]
                for _ in range(num_obj_references):    # N
                    anc_fname = self.obj_ids_helper.random_example_for(obj_id)
                    anc_pil_img = load_fname(anc_fname)
                    anc_img_a = convert_dtype(anc_pil_img)
                    yield anc_img_a, label_idx
                # TODO: pos should be strictly different to anchors
                #       at the moment this is just 2x num_obj_references examples
                for _ in range(num_obj_references):    # N
                    pos_fname = self.obj_ids_helper.random_example_for(obj_id)
                    pos_pil_img = load_fname(pos_fname)
                    pos_img_a = convert_dtype(pos_pil_img)
                    yield pos_img_a, label_idx

    def dataset(self, num_batches, num_obj_references, num_contrastive_examples):

        # num_contrastive_examples = self.obj_ids_helper.check_or_derive(num_contrastive_examples)
        # print("num_contrastive_examples", num_contrastive_examples)

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

        # first batch is the N reference examples
        # (N, HW, HW, 3)
        ds = ds.batch(num_obj_references)

        # second batch for the pairs
        # (2, N, HW, HW, 3)
        ds = ds.batch(2)

        # final batch for the num of examples
        # (C, 2, N, HW, HW, 3)
        ds = ds.batch(num_contrastive_examples)

        # dataset will return num_batches examples
        return ds

class SceneExamples(object):

    def __init__(self,
                 obj_ids_helper,
                 grid_size: int,
                 num_other_objs: int,
                 instances_per_obj: int,
                 seed: int):

        num_objects_to_be_added = instances_per_obj * (num_other_objs + 1)
        num_grid_cells = grid_size ** 2
        if num_objects_to_be_added > num_grid_cells:
            raise Exception(f"can't fit instances_per_obj={instances_per_obj} &"
                            f" num_other_objs={num_other_objs} into"
                            f" grid_size={grid_size}")

        self.obj_ids_helper = copy.deepcopy(obj_ids_helper)
        self.grid_size = grid_size
        self.num_other_objs = num_other_objs
        self.instances_per_obj = instances_per_obj
        self.rnd = random.Random(seed)


    def x_y_ids_for_example(
            self,
            focus_obj_id: int       # primary object id
    ):

        # decide set of obj ids; seeded by obj_id
        eg_obj_ids = set([focus_obj_id])
        while len(eg_obj_ids) < self.num_other_objs+1:
            candidate_id = self.rnd.choice(self.obj_ids_helper.obj_ids)
            eg_obj_ids.add(candidate_id)
        eg_obj_ids = list(eg_obj_ids)

        # create generator for distinct x, y coords
        def distinct_xy_generator():
            yielded = set()
            while True:
                xy = (self.rnd.randint(0, self.grid_size-1),
                      self.rnd.randint(0, self.grid_size-1))
                if xy not in yielded:
                    yield xy
                    yielded.add(xy)
        distinct_xy = distinct_xy_generator()

        # generate x, y, obj_id examples
        for obj_id in eg_obj_ids:
            for _ in range(self.instances_per_obj):
                yield *next(distinct_xy), obj_id


    def render_example(
            self,
            focus_obj_id: int,
            x_y_obj_ids: List[Tuple]
    ):

        collage = None  # laxy create once we have first image size
        labels = np.zeros((self.grid_size, self.grid_size), dtype=int)

        for x, y, obj_id in x_y_obj_ids:
            try:
                fname = self.obj_ids_helper.random_example_for(obj_id)
                pil_img = load_fname(fname)

                if collage is None:
                    img_w, img_h = pil_img.size
                    collage = Image.new(
                        'RGB',
                        (img_w * self.grid_size, img_h * self.grid_size),
                        'white')

                # recall PIL and numpy x/y are transposed
                paste_y = img_w * x
                paste_x = img_h * y
                collage.paste(pil_img, (paste_x, paste_y))

                if obj_id == focus_obj_id:
                    labels[x, y] = 1
            except AssertionError as ae:
                print("getting 'assert self.png is not None' (???)")
                print(str(ae))
                print('fname', fname)

        return collage, labels

    def _example_generator(self, num_batches):
        for b in range(num_batches):
            obj_ids = self.rnd_obj_ids.next_ids()
            for obj_id in obj_ids:  # C
                # print("scene._example_generator",
                #       f"batch={b} obj_id={obj_id}"
                #       f" ( label = {self.obj_ids_helper.label_str_to_idx[obj_id]} ) ")
                xy_ids = self.x_y_ids_for_example(obj_id)
                pil_collage, labels = self.render_example(obj_id, xy_ids)
                yield convert_dtype(pil_collage), labels


    def dataset(self, num_batches, num_focus_objects):

        self.rnd_obj_ids = self.obj_ids_helper.construct_random_obj_id_generator(
            num_focus_objects)

        ds = tf.data.Dataset.from_generator(
            lambda: self._example_generator(
                num_batches),
            output_signature=(
                tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32),  # img
                tf.TensorSpec(shape=(None, None),    dtype=tf.int64),    # label
            )
        )

        # single batch corresponds to number of focus objects
        ds = ds.batch(num_focus_objects)

        # dataset will return num_batches instances of (C, HW, HW, 3)
        return ds


if __name__ == '__main__':

    from util import to_pil_img, collage, create_dir_if_required, highlight
    import copy

    obj_ids_helper = ObjIdsHelper(
        root_dir='data/train/reference_patches/',
        obj_ids=["061","135","182",  # x3 red
                 "111","153","198",  # x3 green
                 "000","017","019"], # x3 blue
        seed=123
    )

    # things that must match across dataset generators
    B = 2  # number of batches
    C = 4  # number of contrastive & focus objects

    c_egs = ContrastiveExamples(obj_ids_helper)
    N = 3
    c_ds = c_egs.dataset(num_batches=B,
                       num_obj_references=N,
                       num_contrastive_examples=C
                       )
    # for b, (bx, by) in enumerate(ds):
    #     # (2xC=6, N=2, HW, HW, 3)
    #     print("c_egs batch!", b, bx.shape, by)
    #     for n in range(N):
    #         imgs = list(map(to_pil_img, bx[:,n]))  # nth anc/pos set
    #         create_dir_if_required(f"data_egs/b{b}")
    #         collage(imgs, C, 2).save(f"data_egs/b{b}/c_egs.n{n}.png")


    s_egs = SceneExamples(
        obj_ids_helper=obj_ids_helper,
        grid_size=10,
        num_other_objs=4,
        instances_per_obj=3,
        seed=123,
    )
    s_ds = s_egs.dataset(num_batches=B,
                       num_focus_objects=C  # C
                       )

    for b, ((obj_x, obj_y), (scene_x, scene_y)) in enumerate(zip(c_ds, s_ds)):
        obj_x = np.array(obj_x)
        obj_y = np.array(obj_y)
        scene_x = np.array(scene_x)
        scene_y = np.array(scene_y)
        print("b", b,
              "obj_", obj_x.shape, obj_y.shape,
              "scene_", scene_x.shape, scene_y.shape)

        create_dir_if_required(f"data_egs/b{b}")

        # (C, AP=2, N, HW, HW, 3)
        print("c_egs batch!", b, obj_x.shape, obj_y)
        for n in range(N):
            ancs_poss = obj_x[:,:,n]  # (C, 2, HW, HW, 3)
            ancs_poss = np.reshape(ancs_poss, (-1, 64, 64, 3))  # (2C, HW, HW, 3)
            imgs = list(map(to_pil_img, ancs_poss))
            collage(imgs, C, 2).save(f"data_egs/b{b}/c_egs.n{n}.png")

        print("s_egs batch!", b, scene_x.shape, scene_y)
        for c in range(C):
            scene_img = to_pil_img(scene_x[c])
            scene_img = highlight(scene_img, scene_y[c])
            scene_img.save(f"data_egs/b{b}/s_egs.eg{c}.png")
