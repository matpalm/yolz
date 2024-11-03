import pybullet as p
from PIL import Image
import numpy as np

def render(hw: int, include_alpha: bool=False):

    proj_matrix = p.computeProjectionMatrixFOV(fov=50,
                                                aspect=float(hw) / hw,
                                                nearVal=0.1,
                                                farVal=100.0)

    view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=(0, 0, 0),
                                                        distance=0.2,
                                                        yaw=0, pitch=0, roll=0,
                                                        upAxisIndex=2)

    light_colour = (1, 1, 1)
    light_direction = (0, 1, 1, 0)
    rendering = p.getCameraImage(width=hw, height=hw,
                                 viewMatrix=view_matrix,
                                 projectionMatrix=proj_matrix,
                                 lightColor=light_colour,
                                 lightDirection=light_direction,
                                 shadow=1,
                                 renderer=p.ER_BULLET_HARDWARE_OPENGL)  # shadows are noisy in OPENGL (?)

    # note: render is 4 channel, but doesn't include alpha :/
    render_rgba = rendering[2]
    rgba_array = np.array(render_rgba, dtype=np.uint8)
    rgb_array = rgba_array[:, :, :3]
    img = Image.fromarray(rgb_array, 'RGB')
    if not include_alpha:
        return img

    render_segmentation = rendering[4]
    alpha = np.where(render_segmentation==-1, 0, 255)
    alpha = Image.fromarray(alpha.astype(np.uint8), mode='L')
    img.putalpha(alpha)
    return img

# below is all the stuff related to bounding box we don't use yet...

# def bullet_render(width_height, fov):
#     width, height = width_height, width_height
#     proj_matrix = p.computeProjectionMatrixFOV(fov=fov,
#                                                aspect=float(width) / height,
#                                                nearVal=0.1,
#                                                farVal=100.0)

#     camera_target = (0, 0, 0)
#     distance = 0.5
#     # yaw=0 => left hand side, =90 towards center, =180 from right hand side
#     yaw = 0
#     # pitch=0 => horizontal, -90 down
#     pitch = -90
#     roll = 0
#     view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=camera_target,
#                                                       distance=distance,
#                                                       yaw=yaw, pitch=pitch, roll=roll,
#                                                       upAxisIndex=2)

#     light_colour = (1, 1, 1)
#     light_direction = (0, 1, 1, 0)
#     rendering = p.getCameraImage(width=width, height=height,
#                                  viewMatrix=view_matrix,
#                                  projectionMatrix=proj_matrix,
#                                  lightColor=light_colour,
#                                  lightDirection=light_direction,
#                                  shadow=0,
#                                  renderer=p.ER_BULLET_HARDWARE_OPENGL)  # shadows are noisy in OPENGL (?)

#     render_rgba = rendering[2]
#     render_segmentation = rendering[4]
#     return render_rgba, render_segmentation


# def render_rgba_to_pil_img(render_rgba):
#     rgba_array = np.array(render_rgba, dtype=np.uint8)
#     rgb_array = rgba_array[:, :, :3]
#     return Image.fromarray(rgb_array)

# def semgentation_to_bounding_boxes(segmentation, uid_to_urdf):
#     detections = []

#     for obj_id in np.unique(segmentation):
#         if obj_id < 0:
#             # ignore background
#             continue
#         mask = np.where(segmentation==obj_id)

#         # note: transposed x/y
#         y0, y1 = map(int, (np.min(mask[0]), np.max(mask[0])))
#         x0, x1 = map(int, (np.min(mask[1]), np.max(mask[1])))

#         detections.append({
#             'label': f"o{uid_to_urdf[obj_id]}",
#             'bbox': (x0, y0, x1, y1)
#         })

#     return detections