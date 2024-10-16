import pybullet as p
import pybullet_data
import numpy as np
import os, random, math, tqdm
from PIL import Image

HW = 160
proj_matrix = p.computeProjectionMatrixFOV(fov=50,
                                            aspect=float(HW) / HW,
                                            nearVal=0.1,
                                            farVal=100.0)

view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=(0, 0, 0),
                                                    distance=0.2,
                                                    yaw=0, pitch=0, roll=0,
                                                    upAxisIndex=2)

def render():
    light_colour = (1, 1, 1)
    light_direction = (0, 1, 1, 0)
    rendering = p.getCameraImage(width=HW, height=HW,
                                 viewMatrix=view_matrix,
                                 projectionMatrix=proj_matrix,
                                 lightColor=light_colour,
                                 lightDirection=light_direction,
                                 shadow=1,
                                 renderer=p.ER_BULLET_HARDWARE_OPENGL)  # shadows are noisy in OPENGL (?)

    render_rgba = rendering[2]
    rgba_array = np.array(render_rgba, dtype=np.uint8)
    rgb_array = rgba_array[:, :, :3]
    return Image.fromarray(rgb_array)

#p.connect(p.GUI)
p.connect(p.DIRECT)

D = pybullet_data.getDataPath() + '/random_urdfs/'
for urdf in tqdm.tqdm(os.listdir(D)):
    urdf_filename = f"{D}/{urdf}/{urdf}.urdf"
    collage = Image.new('RGB', (HW*2, HW*2))
    for i in range(4):
        x, y = i // 2, i % 2
        block_angle = ( random.random() * math.pi, random.random() * math.pi, random.random() * math.pi)
        block_orient = p.getQuaternionFromEuler(block_angle)
        obj_uid = p.loadURDF(urdf_filename, (0,0,0), block_orient)
        img = render()
        collage.paste(img, (x*HW, y*HW))
        p.removeBody(obj_uid)
    collage.save(f"data/all_egs/{int(urdf):04d}.png")


