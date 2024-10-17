import random
import os
import math
import json
from tqdm import tqdm

import pybullet as p
import pybullet_data

from util import create_dir_if_required

from .render_util import render

import argparse
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--urdf-ids', type=str,
                    help='(json) str with urdfs ids to iterate over')
parser.add_argument('--render-width-height', type=int, default=64)
parser.add_argument('--render-fov', type=int, default=70)
parser.add_argument('--num-examples', type=int, default=10,
                    help='num examples to generate per urdf')
parser.add_argument('--output-dir', type=str, required=True)
opts = parser.parse_args()
print(opts)

urdf_ids = json.loads(opts.urdf_ids)

#p.connect(p.GUI)
p.connect(p.DIRECT)

for urdf_id in tqdm(urdf_ids):
    urdf_filename = os.path.join(pybullet_data.getDataPath(), "random_urdfs",
                                    urdf_id, f"{urdf_id}.urdf")

    output_dir = os.path.join(opts.output_dir, urdf_id)
    create_dir_if_required(output_dir)

    for i in range(opts.num_examples):

        obj_pos = 0, 0, 0
        obj_angle = [random.random() * math.pi for _ in range(3) ]  # x, y, z
        obj_orient = p.getQuaternionFromEuler(obj_angle)
        obj_uid = p.loadURDF(urdf_filename,
                            basePosition=obj_pos, baseOrientation=obj_orient,
                            globalScaling=1)

        img = render(opts.render_width_height)
        img.save(os.path.join(output_dir, f"{i:04d}.png"))

        p.removeBody(obj_uid)


# uid_to_urdf = {}

# random_urdf_id = random.choice(urdf_ids)
# urdf_filename = os.path.join(pybullet_data.getDataPath(), "random_urdfs",
#                                     random_urdf_id, f"{random_urdf_id}.urdf")
# print("urdf_filename", urdf_filename)

# obj_pos = 0, 0, 0

# obj_angle = [random.random() * math.pi for _ in range(3) ]  # x, y, z
# obj_orient = p.getQuaternionFromEuler(obj_angle)

# obj_uid = p.loadURDF(urdf_filename,
#                         basePosition=obj_pos, baseOrientation=obj_orient,
#                         globalScaling=1)

# uid_to_urdf[obj_uid] = random_urdf_id

# render_rgba, render_segmentation = bullet_render(
#     width_height=opts.render_width_height,
#     fov=opts.render_fov)

# pil_img = render_rgba_to_pil_img(render_rgba)
# pil_img.save('test.png')

# detections = semgentation_to_bounding_boxes(render_segmentation, uid_to_urdf)
# print(detections)

