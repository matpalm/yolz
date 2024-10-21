import os
os.environ['KERAS_BACKEND'] = 'jax'

from data import load_fname
from models.embeddings import construct_embedding_model

import json
from tqdm import tqdm
import pickle
from jax import jit

import numpy as np
np.set_printoptions(precision=5, threshold=10000, suppress=True, linewidth=10000)

import argparse
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model-config-json', type=str, required=True,
                    help='embedding model config json file')
parser.add_argument('--weights-pkl', type=str, required=True)
parser.add_argument('--manifest', type=str, required=True)
parser.add_argument('--embeddings-npy', type=str, required=True)
parser.add_argument('--batch-size', type=int, default=256)
opts = parser.parse_args()
print("opts", opts)

# create model, load weights and jit embedding function
with open(opts.model_config_json, 'r') as f:
    model_config = json.load(f)
embedding_model = construct_embedding_model(**model_config)

with open(opts.weights_pkl, 'rb') as f:
    reloaded_weights = pickle.load(f)
embedding_model.set_weights(reloaded_weights)

@jit
def embed(b):
    return embedding_model(b)

# iterate manifest collecting into batches
# TODO: is there a functools helper for this?
manifest = open(opts.manifest, 'r').readlines()
fname_batchs = []
batch = []
for fname in manifest:
    batch.append(fname)
    if len(batch) == opts.batch_size:
        fname_batchs.append(batch)
        batch = []
if len(batch) > 0:
    fname_batchs.append(batch)

# run all embeddings, concat and save
embeddings = []
for fnames in tqdm(fname_batchs):
    imgs = [load_fname(f) for f in fnames]
    batch = np.stack(imgs)
    embeddings.append(embed(batch))
embeddings = np.concatenate(embeddings)
print('embeddings', embeddings.shape)
np.save(opts.embeddings_npy, embeddings)
