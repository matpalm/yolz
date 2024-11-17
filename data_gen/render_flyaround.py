import random
import time
import tqdm
import os
import math
import numpy as np
import json

import pybullet as p
import pybullet_data

from .render_util import render_scene_with_params
from .util import create_dir_if_required

import argparse
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--output-dir', type=str, required=True,
                    help='where to load model config etc')
parser.add_argument('--num-frames', type=int, default=1000)
parser.add_argument('--urdf-id-from', type=int, default=0)
parser.add_argument('--urdf-id-to', type=int, default=600)
parser.add_argument('--num-distinct-objects', type=int, default=8,
                    help='how many distinct obj_ids in scene')
parser.add_argument('--new-object-freq', type=int, default=5,
                    help='frequency in frames of adding a new random object')
parser.add_argument('--render-width-height', type=int, default=640)
parser.add_argument('--segmask-width-height', type=int, default=80)
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--gui', action='store_true')

opts = parser.parse_args()
print("opts", opts)

# class Opts(object):
#     output_dir = '/home/mat/dev/zero_shot_detection/data/fly_around'
#     num_frames = 1000
#     urdf_id_from = 0
#     urdf_id_to = 600
#     num_distinct_objects = 8   # how many distinct obj_ids in scene
#     new_object_freq = 5        # frequency in frames of adding a new random object
#     render_width_height = 640
#     segmask_width_height = 80
#     seed = 123
#opts = Opts()

random.seed(opts.seed)

def random_in(a, b):
    if a > b:
        a, b = b, a
    return a + (random.random() * (b - a))

# decide which distinct objects will be in this scene
urdf_ids = range(opts.urdf_id_from, opts.urdf_id_to)
if opts.num_distinct_objects > len(urdf_ids):
    raise Exception("not enough urdf_ids to satisfy num_distinct_objects")
distinct_urdfs = set()
while len(distinct_urdfs) != opts.num_distinct_objects:
    distinct_urdfs.add(random.choice(urdf_ids))
distinct_urdfs = list(distinct_urdfs)
print("distinct_urdfs", distinct_urdfs)

# setup bullet
if opts.gui:
    p.connect(p.GUI)
else:
    p.connect(p.DIRECT)
p.setGravity(0, 0, -9.81)
p.loadURDF(pybullet_data.getDataPath()+"/table/table.urdf", 0.5,0.,-0.82, 0,0,0,1)

# decide fixed params for rendering
fov = random_in(40, 50)
tx = random_in(0.4, 0.6)
ty = random_in(-0.1, 0.1)
tz = 0
target = tx, ty, tz
distance = random_in(1, 2)
yaw = 0
pitch = random_in(-60, -50)
roll = random_in(-10, 10)

def render(yaw):
    return render_scene_with_params(
        opts.render_width_height,
        fov, target, distance,
        yaw, pitch, roll)

# helper objects for managing loaded objects

class Objects(object):

    def __init__(self):
        self.obj_uids = []
        self.obj_uid_to_urdf = {}

    def add(self, urdf_id):
        # add object in random location
        urdf_filename = os.path.join(pybullet_data.getDataPath(), "random_urdfs",
                                     f"{urdf_id:03d}", f"{urdf_id:03d}.urdf")
        x = random_in(0, 1)
        y = random_in(-0.25, 0.25)
        z = 0.75
        obj_angle = [random.random() * math.pi for _ in range(3) ]  # x, y, z
        obj_orient = p.getQuaternionFromEuler(obj_angle)
        obj_uid = p.loadURDF(urdf_filename,
                             basePosition=(x, y, z),
                             baseOrientation=obj_orient,
                             globalScaling=1)
        self.obj_uids.append(obj_uid)
        self.obj_uid_to_urdf[obj_uid] = urdf_id

    def distinct_urdfs(self):
        return sorted(set(self.obj_uid_to_urdf.values()))

    def check_fallen_out_of_bounds(self):
        # remove any objects that have fallen below the table
        to_remove = []
        for uid in objects.obj_uids:
            (_x, _y, z), _orient = p.getBasePositionAndOrientation(uid)
            if z < -1:
                to_remove.append(uid)
        for uid in to_remove:
            objects.obj_uids.remove(uid)
            p.removeBody(uid)
            self.obj_uid_to_urdf.pop(uid)

    def remove_all(self):
        for uid in objects.obj_uids:
            p.removeBody(uid)
            self.obj_uid_to_urdf.pop(uid)


def save_scene_assets(output_dir, img, segmentation, objects):

    # TODO: refactor common code out from render_scenes.py

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
        bboxes.append({'bbox': xyxy, 'label': objects.obj_uid_to_urdf[uid]})
    with open(os.path.join(output_dir, "bboxes.json"), 'w') as f:
        json.dump(bboxes, f)

    # write added urdfs ( in order matching seg mask rows )
    with open(os.path.join(output_dir, "obj_ids.json"), 'w') as f:
        json.dump(objects.distinct_urdfs(), f)

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
    for i, urdf_id in enumerate(objects.distinct_urdfs()):
        mask = np.zeros((opts.segmask_width_height, opts.segmask_width_height), np.uint8)
        for bbox in bboxes:
            if bbox['label'] == urdf_id:
                x0, y0, x1, y1 = bbox['bbox']
                x0, y0, x1, y1 = map(downscale_and_clip, (x0-padding, y0-padding, x1+padding, y1+padding))
                for x in range(x0, x1+1):
                    for y in range(y0, y1+1):
                        mask[x, y] = 1
        masks.append(mask.T)  # transpose to make it align with RGB
    masks = np.stack(masks)  # ( |obj_ids|, hw, hw )
    np.save(os.path.join(output_dir, "masks.npy"), masks)


objects = Objects()

for frame_id in tqdm.tqdm(range(opts.num_frames)):
    output_dir = os.path.join(opts.output_dir, 'scenes', f"{frame_id:06d}")
    create_dir_if_required(output_dir)

    if frame_id % opts.new_object_freq == 0:
        objects.add(random.choice(distinct_urdfs))

    for _ in range(5):  # TODO: eye ball based on relationship to yaw +=
        p.stepSimulation()

    # render the scene
    img, segmentation = render(yaw)
    save_scene_assets(output_dir, img, segmentation, objects)
    yaw += 1

    objects.check_fallen_out_of_bounds()

    time.sleep(0.05)

objects.remove_all()

