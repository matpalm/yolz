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
    # given a pil image and a 2D label array, 0s and 1s
    # draw red "bboxes" on pil_img where labels is 1
    iw, ih = pil_img.size
    gw, gh = labels.shape
    sw = iw / gw
    sh = ih / gh
    draw = ImageDraw.Draw(pil_img)
    for x, y in np.argwhere(labels==1):
        x0 = x * sw
        x1 = x0 + sw
        y0 = y * sh
        y1 = y0 + sh
        draw.rectangle((y0,x0,y1,x1), fill=None, outline='red')
    return pil_img

def smooth_highlight(pil_img, labels):
    # given a pil image and a 2D label array with values (0, 1)
    # draw red "bboxes" on pil_img where brightness is based on label value
    iw, ih = pil_img.size
    gw, gh = labels.shape
    sw = iw / gw
    sh = ih / gh
    draw = ImageDraw.Draw(pil_img)
    for i in range(gw):
        for j in range(gh):
            brightness = int(labels[i][j] * 255)
            x0 = i * sw
            x1 = x0 + sw
            y0 = j * sh
            y1 = y0 + sh
            draw.rectangle((y0+2,x0+2,y1-2,x1-2),
                           fill=None,
                           outline=(255,255-brightness,255-brightness))
    return pil_img

# from jaxopt
def binary_logistic_loss(label: int, logit: float) -> float:
    return softplus(jnp.where(label, -logit, logit))
