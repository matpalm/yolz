import os
from PIL import Image, ImageDraw
import numpy as np

from jax.nn import softplus
import jax.numpy as jnp

def create_dir_if_required(d):
    if d in [None, '', ',']:
        return
    if not os.path.exists(d):
        try:
            os.makedirs(d)
        except FileExistsError:
            # possible race condition
            pass

def to_pil_img(a):
    return Image.fromarray(np.array(a))

#def convert_dtype(pil_img):
#    return np.array(pil_img, dtype=float) / 255

def collage(pil_imgs, rows, cols):
    n = len(pil_imgs)
    if n != rows * cols:
        raise Exception(f"received {len(pil_imgs)} images but a"
                        f" collage of ({rows}, {cols}) was requested")
    img_w, img_h = pil_imgs[0].size
    collage = Image.new('RGBA', (cols*img_w, rows*img_h))
    for i in range(n):
        pc, pr = i%cols, i//cols
        collage.paste(pil_imgs[i], (pc*img_w, pr*img_h))
    return collage

def array_to_pil_img(a):
    a *= 255
    a = np.array(a, np.uint8).squeeze()
    return Image.fromarray(a, 'RGB').resize((640,640), 0)

def mask_to_pil_img(mask):
    mask *= 255
    mask = np.array(mask, np.uint8).squeeze()
    return Image.fromarray(mask, 'L').resize((640,640), 0)

def alpha_from_mask(rgb_pil_img, mask_pil_img):
    img_a = np.array(rgb_pil_img)
    mask_a = np.expand_dims(np.array(mask_pil_img), -1)
    masked_img_a = np.concatenate([img_a, mask_a], axis=-1)
    return Image.fromarray(masked_img_a, 'RGBA')

