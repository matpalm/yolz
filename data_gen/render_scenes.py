import random
import os
import math
import json
from tqdm import tqdm

import pybullet as p
import pybullet_data

from util import create_dir_if_required
import numpy as np
from .render_util import render_scene
import random

import argparse
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--urdf-id-from', type=int, default=0,
                    help='min urdf-id to choose from')
parser.add_argument('--urdf-id-to', type=int, default=10,
                    help='max urdf-id to choose from (exclusive)')
parser.add_argument('--num-scenes', type=int, default=10,
                    help='number of examples to generate.')
parser.add_argument('--render-width-height', type=int, default=640)
parser.add_argument('--segmask-width-height', type=int, default=80)
parser.add_argument('--num-examples', type=int, default=10,
                    help='total number of scene examples to generate')
parser.add_argument('--num-distinct-objects-per-example', type=int, default=16,
                    help='the number of distinct urdf objects in each scene')
parser.add_argument('--num-instances-per-object', type=int, default=4,
                    help='how many of each object is in a scene')
parser.add_argument('--output-dir', type=str, required=True)
parser.add_argument('--seed', type=int, default=1337)
opts = parser.parse_args()
print(opts)

random.seed(opts.seed)

urdf_ids = range(opts.urdf_id_from, opts.urdf_id_to)

#p.connect(p.GUI)
p.connect(p.DIRECT)
p.setGravity(0, 0, -9.81)
p.loadURDF(pybullet_data.getDataPath()+"/table/table.urdf", 0.5,0.,-0.82, 0,0,0,1)

def random_in(a, b):
    if a > b:
        a, b = b, a
    return a + (random.random() * (b - a))

for example_id in tqdm(range(opts.num_examples)):

    output_dir = os.path.join(opts.output_dir, f"{example_id:06d}")
    create_dir_if_required(output_dir)

    # decide which distinct objects will be in this scene
    if opts.num_distinct_objects_per_example > len(urdf_ids):
        raise Exception("not enough urdf_ids to satisfy num_distinct_objects_per_example")
    distinct_urdfs_to_add = set()
    while len(distinct_urdfs_to_add) != opts.num_distinct_objects_per_example:
        distinct_urdfs_to_add.add(random.choice(urdf_ids))
    # make list with some number per distinct urdf
    urdfs_to_add = []
    for urdf_id in distinct_urdfs_to_add:
        for _ in range(opts.num_instances_per_object):
            urdfs_to_add.append(urdf_id)
    # shuffle them ( for insert ordering )
    random.shuffle(urdfs_to_add)

    # add objects, record mapping to urdf for recreating bounding boxes
    obj_uids_to_urdf = {}
    for urdf_id in urdfs_to_add:

        # add object in random location
        urdf_filename = os.path.join(pybullet_data.getDataPath(), "random_urdfs",
                                     f"{urdf_id:03d}", f"{urdf_id:03d}.urdf")
        x = random_in(0, 1)
        y = random_in(-0.25, 0.25)
        z = 0.5
        obj_angle = [random.random() * math.pi for _ in range(3) ]  # x, y, z
        obj_orient = p.getQuaternionFromEuler(obj_angle)
        obj_uid = p.loadURDF(urdf_filename,
                             basePosition=(x, y, z),
                             baseOrientation=obj_orient,
                             globalScaling=1)
        obj_uids_to_urdf[obj_uid] = urdf_id
        # let the object fall
        for _ in range(10):
            p.stepSimulation()

    # final settling
    for _ in range(100):
        p.stepSimulation()

    # render the scene
    img, segmentation = render_scene(opts.render_width_height)

    # save RGB image
    img.save(os.path.join(output_dir, "rgb.png"))

    # convert segmentation map to bounding boxes
    # write to json
    segmentation = segmentation.T
    bboxes = []
    for uid in np.unique(segmentation):
        if uid < 1:
            # 0 => table
            # -1 => background
            continue
        mask = np.where(segmentation==uid)
        x0, x1 = np.min(mask[0]), np.max(mask[0])
        y0, y1 = np.min(mask[1]), np.max(mask[1])
        xyxy = list(map(int, (x0, y0, x1, y1)))
        bboxes.append({'bbox': xyxy, 'label': obj_uids_to_urdf[uid]})
    with open(os.path.join(output_dir, "bboxes.json"), 'w') as f:
        json.dump(bboxes, f)

    # write added urdfs ( in order matching seg mask rows )
    objs_ids = list(set(urdfs_to_add))
    with open(os.path.join(output_dir, "obj_ids.json"), 'w') as f:
        json.dump(objs_ids, f)

    # build segmentation mask array (|objs|, H, W,) with values {0, 1}
    # row order matches written obj_ids.json
    downscaling = opts.render_width_height / opts.segmask_width_height
    padding = 3
    def downscale_and_clip(v):
        v /= downscaling
        v = max(0, v)
        v = min(v, opts.segmask_width_height-1)
        return int(v)
    masks = []
    for i, obj_id in enumerate(objs_ids):
        mask = np.zeros((opts.segmask_width_height, opts.segmask_width_height), np.uint8)
        for bbox in bboxes:
            if bbox['label'] == obj_id:
                x0, y0, x1, y1 = bbox['bbox']
                x0, y0, x1, y1 = map(downscale_and_clip, (x0-padding, y0-padding, x1+padding, y1+padding))
                for x in range(x0, x1+1):
                    for y in range(y0, y1+1):
                        mask[x, y] = 1
        masks.append(mask.T)  # transpose to make it align with RGB
    masks = np.stack(masks)  # ( |obj_ids|, hw, hw )
    np.save(os.path.join(output_dir, "masks.npy"), masks)

    # remove objects in prep for next render
    for obj_uid in obj_uids_to_urdf.keys():
        p.removeBody(obj_uid)
