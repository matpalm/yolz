import os
os.environ['KERAS_BACKEND'] = 'jax'

from data import load_fname
from models.embeddings import construct_embedding_model

import json, tqdm
import pickle


import numpy as np
np.set_printoptions(precision=5, threshold=10000, suppress=True, linewidth=10000)

import argparse
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--height-width', type=int, default=64)
parser.add_argument('--filter-sizes', type=str, default='[8,16,32]')
parser.add_argument('--embedding-dim', type=int, default=128)
parser.add_argument('--manifest', type=str, required=True)
parser.add_argument('--weights-pkl', type=str, required=True)
parser.add_argument('--embeddings-npy', type=str, required=True)
parser.add_argument('--batch-size', type=int, default=128)
opts = parser.parse_args()
print("opts", opts)

filter_sizes = json.loads(opts.filter_sizes)
for f in filter_sizes:
    assert type(f) == int and f > 0

manifest = open(opts.manifest, 'r').readlines()

embedding_model = construct_embedding_model(
    height=opts.height_width,
    width=opts.height_width,
    filter_sizes=filter_sizes,
    embedding_dim=opts.embedding_dim
)

with open(opts.weights_pkl, 'rb') as f:
    reloaded_weights = pickle.load(f)
embedding_model.set_weights(reloaded_weights)

batch = []
embeddings = []
for fname in manifest:
    batch.append(load_fname(fname))
    if len(batch) == opts.batch_size:
        batch = np.stack(batch)
        embeddings.append(embedding_model(batch))
        batch = []
if len(batch) > 0:
    # embed final batch
    batch = np.stack(batch)
    embeddings.append(embedding_model(batch))

embeddings = np.concatenate(embeddings)
print('embeddings', embeddings.shape)
np.save(opts.embeddings_npy, embeddings)
