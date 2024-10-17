import pybullet as p
import pybullet_data
import os, random, math, tqdm
from PIL import Image

from .render_util import render

#p.connect(p.GUI)
p.connect(p.DIRECT)

HW = 160
D = pybullet_data.getDataPath() + '/random_urdfs/'
for urdf in tqdm.tqdm(os.listdir(D)):
    urdf_filename = f"{D}/{urdf}/{urdf}.urdf"
    collage = Image.new('RGB', (HW*2, HW*2))
    for i in range(4):
        x, y = i // 2, i % 2
        block_angle = ( random.random() * math.pi, random.random() * math.pi, random.random() * math.pi)
        block_orient = p.getQuaternionFromEuler(block_angle)
        obj_uid = p.loadURDF(urdf_filename, (0,0,0), block_orient)
        img = render(HW)
        collage.paste(img, (x*HW, y*HW))
        p.removeBody(obj_uid)
    collage.save(f"data/all_egs/{int(urdf):04d}.png")


