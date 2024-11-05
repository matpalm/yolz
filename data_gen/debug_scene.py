from PIL import Image, ImageDraw
from pathlib import Path
import json
import numpy as np

import argparse
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--scene-dir', type=str, required=True)
opts = parser.parse_args()
print(opts)

scene_dir = Path(opts.scene_dir)

img = Image.open(scene_dir/'rgb.png')
img_a = np.array(img)  # (640, 640, 3)
img.save("scene_egs/rgb.png")
print("imgs size", img.size)

with open(scene_dir/'obj_ids.json', 'r') as f:
    obj_ids = json.load(f)

seg_masks = np.load(scene_dir/'masks.npy')
print('seg_masks', seg_masks.shape)
assert len(seg_masks) == len(obj_ids)
h, w = seg_masks.shape[1], seg_masks.shape[2]
assert h == w
hw = h
scale = img.size[0] / hw
print('hw', hw, 'scale', scale)

for i, obj_id in enumerate(obj_ids):
    mask_img = Image.fromarray(seg_masks[i], 'L').resize(img.size, Image.Resampling.NEAREST)
    mask_a = np.expand_dims(np.array(mask_img), -1)  # (640, 640, 1)
    mask_a *= 255
    masked_img_a = np.concatenate([img_a, mask_a], axis=-1)  # (640, 640, 4)
    masked_img = Image.fromarray(masked_img_a, 'RGBA')
    draw = ImageDraw.Draw(masked_img)
    for xy in range(0, hw):
        draw.line((xy*scale, 0, xy*scale, hw*scale))
        draw.line((0, xy*scale, hw*scale, xy*scale))
    masked_img.save(f"scene_egs/mask_{i}_{obj_id}.png")
