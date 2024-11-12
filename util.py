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

def generate_debug_imgs(anchors_a, scene_img_a, masks_a, y_pred, step, img_output_dir):
    # debug some reference examples
    #img_output_dir = os.path.join(opts.run_dir, 'debug_imgs')
    create_dir_if_required(img_output_dir)
    for obj_idx in [4, 5, 6]:
        num_egs = anchors_a.shape[1]
        ref_images = [array_to_pil_img(anchors_a[obj_idx, a]) for a in range(num_egs)]
        c = collage(ref_images, rows=1, cols=num_egs)
        c.save(f"{img_output_dir}/step_{step:06d}.obj_{obj_idx}.anchors.png")
        y_true_m = mask_to_pil_img(masks_a[obj_idx])
        y_pred_m = mask_to_pil_img(y_pred[obj_idx])
        c = collage([y_true_m, y_pred_m], rows=1, cols=2)
        c.save(f"{img_output_dir}/step_{step:06d}.obj_{obj_idx}.y_true_pred.mask.png")
        scene_rgb = array_to_pil_img(scene_img_a[0])
        scene_with_y_pred = alpha_from_mask(scene_rgb, y_pred_m)
        scene_with_y_true = alpha_from_mask(scene_rgb, y_true_m)
        c = collage([scene_with_y_pred, scene_rgb, scene_with_y_true], rows=1, cols=3)
        c.save(f"{img_output_dir}/step_{step:06d}.obj_{obj_idx}.y_true_pred.alpha.png")