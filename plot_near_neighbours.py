import numpy as np
import jax.numpy as jnp
from PIL import Image
import json

from jax import jit

from util import collage

import argparse
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--manifest', type=str, required=True)
parser.add_argument('--embeddings-npy', type=str, required=True)
parser.add_argument('--idxs', type=str, required=True, help='json list of idxs')
parser.add_argument('--plot-fname', type=str, required=True, help='fname for nn plot')
opts = parser.parse_args()
print("opts", opts)

embeddings = jnp.array(np.load(opts.embeddings_npy))
print('e', embeddings.shape)

idxs_i = json.loads(opts.idxs)
assert len(idxs_i) > 0
assert max(idxs_i) < len(embeddings)

with open(opts.manifest, 'r') as f:
    manifest = f.read().split()
print("|manifest|", len(manifest))

@jit
def sims_for_idx(i):
    return jnp.dot(embeddings[i], embeddings.T)

def top_10_collage_for(idx_i):
    sims = sims_for_idx(idx_i)
    top_10_near_neighbours = reversed(np.argsort(sims)[-10:])
    pil_imgs = [Image.open(manifest[idx_i])]
    for i, idx_j in enumerate(top_10_near_neighbours):
        #print(i, idx_i, idx_j, sims[idx_i][idx_j], manifest[idx_j])
        pil_imgs.append(Image.open(manifest[idx_j]))
    return pil_imgs

pil_imgs = []
for idx_i in idxs_i:
    pil_imgs += top_10_collage_for(idx_i)
collage(pil_imgs, rows=len(idxs_i), cols=11).save(opts.plot_fname)
