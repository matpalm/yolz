import numpy as np
from PIL import Image
import random
import os
import math
import json

import pybullet as p
import pybullet_data

def bullet_render(width_height, fov):
    width, height = width_height, width_height
    proj_matrix = p.computeProjectionMatrixFOV(fov=fov,
                                               aspect=float(width) / height,
                                               nearVal=0.1,
                                               farVal=100.0)

    camera_target = (0, 0, 0)
    distance = 0.5
    # yaw=0 => left hand side, =90 towards center, =180 from right hand side
    yaw = 0
    # pitch=0 => horizontal, -90 down
    pitch = -90
    roll = 0
    view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=camera_target,
                                                      distance=distance,
                                                      yaw=yaw, pitch=pitch, roll=roll,
                                                      upAxisIndex=2)

    light_colour = (1, 1, 1)
    light_direction = (0, 1, 1, 0)
    rendering = p.getCameraImage(width=width, height=height,
                                 viewMatrix=view_matrix,
                                 projectionMatrix=proj_matrix,
                                 lightColor=light_colour,
                                 lightDirection=light_direction,
                                 shadow=0,
                                 renderer=p.ER_BULLET_HARDWARE_OPENGL)  # shadows are noisy in OPENGL (?)

    render_rgba = rendering[2]
    render_segmentation = rendering[4]
    return render_rgba, render_segmentation


def render_rgba_to_pil_img(render_rgba):
    rgba_array = np.array(render_rgba, dtype=np.uint8)
    rgb_array = rgba_array[:, :, :3]
    return Image.fromarray(rgb_array)

def semgentation_to_bounding_boxes(segmentation, uid_to_urdf):
    detections = []

    for obj_id in np.unique(segmentation):
        if obj_id < 0:
            # ignore background
            continue
        mask = np.where(segmentation==obj_id)

        # note: transposed x/y
        y0, y1 = map(int, (np.min(mask[0]), np.max(mask[0])))
        x0, x1 = map(int, (np.min(mask[1]), np.max(mask[1])))

        detections.append({
            'label': f"o{uid_to_urdf[obj_id]}",
            'bbox': (x0, y0, x1, y1)
        })

    return detections


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--urdf-ids', type=str, default='["000", "061", "075"]',
                        help='(json) str with urdfs ids to choose from')
    parser.add_argument('--render-width-height', type=int, default=320)
    parser.add_argument('--render-fov', type=int, default=70)
    parser.add_argument('--num-examples', type=int, default=1000,
                        help='stop after this many generated examples.'
                            ' first --training-proportion is training, remaining is for test')
    parser.add_argument('--gui', action='store_true',
                        help='whether to run GUI or DIRECT')
    opts = parser.parse_args()
    print(opts)
    urdf_ids = json.loads(opts.urdf_ids)

    if opts.gui:
        p.connect(p.GUI)
    else:
        p.connect(p.DIRECT)

    uid_to_urdf = {}

    random_urdf_id = random.choice(urdf_ids)
    urdf_filename = os.path.join(pybullet_data.getDataPath(), "random_urdfs",
                                     random_urdf_id, f"{random_urdf_id}.urdf")
    print("urdf_filename", urdf_filename)

    obj_pos = 0, 0, 0

    obj_angle = [random.random() * math.pi for _ in range(3) ]  # x, y, z
    obj_orient = p.getQuaternionFromEuler(obj_angle)

    obj_uid = p.loadURDF(urdf_filename,
                            basePosition=obj_pos, baseOrientation=obj_orient,
                            globalScaling=1)

    uid_to_urdf[obj_uid] = random_urdf_id

    render_rgba, render_segmentation = bullet_render(
        width_height=opts.render_width_height,
        fov=opts.render_fov)

    pil_img = render_rgba_to_pil_img(render_rgba)
    pil_img.save('test.png')

    detections = semgentation_to_bounding_boxes(render_segmentation, uid_to_urdf)
    print(detections)
