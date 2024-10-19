import os
from PIL import Image
import numpy as np

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

def collage(pil_imgs, rows, cols):
    n = len(pil_imgs)
    if n != rows * cols:
        raise Exception()
    img_h, img_w = pil_imgs[0].size
    collage = Image.new('RGB', (rows*img_h, cols*img_w))
    for i in range(n):
        pr, pc = i%rows, i//rows
        collage.paste(pil_imgs[i], (pr*img_h, pc*img_w))
    return collage

