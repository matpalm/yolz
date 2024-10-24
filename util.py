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
    return Image.fromarray(np.array(a*255, dtype=np.uint8))

def collage(pil_imgs, rows, cols):
    n = len(pil_imgs)
    if n != rows * cols:
        raise Exception(f"received {len(pil_imgs)} images but a"
                        f" collage of ({rows}, {cols}) was requested")
    img_w, img_h = pil_imgs[0].size
    collage = Image.new('RGB', (cols*img_w, rows*img_h))
    for i in range(n):
        pc, pr = i%cols, i//cols
        collage.paste(pil_imgs[i], (pc*img_w, pr*img_h))
    return collage

def highlight(pil_img, labels):
    return pil_img
