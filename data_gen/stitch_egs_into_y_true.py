import os
import json
import random
from functools import lru_cache
from util import create_dir_if_required
from PIL import Image

import argparse
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--reference-egs-dir', type=str)
parser.add_argument('--num-egs', type=int)
parser.add_argument('--output-dir', type=str)
parser.add_argument('--output-labels-json', type=str)
parser.add_argument('--grid-size', type=int, default=10)
parser.add_argument('--grid-spacing', type=int, default=64)
parser.add_argument('--num-objs', type=int, default=10)
opts = parser.parse_args()
print(opts)

assert opts.grid_size ** 2  > opts.num_objs, 'grid not large enough'

create_dir_if_required(opts.output_dir)

obj_ids = os.listdir(opts.reference_egs_dir)

# { 'output_eg_fname.png': [(x, y, obj_id), ...], ... }
all_labels = {}

@lru_cache
def fnames_for(obj_id):
    return os.listdir(os.path.join(opts.reference_egs_dir, obj_id))

for i in range(opts.num_egs):

    # kinda clumsy :/
    grid_entries = []
    for x in range(opts.grid_size):
        for y in range(opts.grid_size):
            grid_entries.append((x, y))
    random.shuffle(grid_entries)

    HW = opts.grid_size * opts.grid_spacing

    collage = Image.new('RGB', (HW, HW), 'white')
    instance_labels = []

    for _ in range(opts.num_objs):
        obj_id = random.choice(obj_ids)
        eg_fname = random.choice(fnames_for(obj_id))
        eg_path = os.path.join(opts.reference_egs_dir, obj_id, eg_fname)
        img = Image.open(eg_path)
        x, y = grid_entries.pop(0)
        collage.paste(img, (x*opts.grid_spacing, y*opts.grid_spacing))
        instance_labels.append((x, y, obj_id))

    fname = os.path.join(opts.output_dir, f"{i:04d}.png")
    collage.save(fname)
    all_labels[fname] = instance_labels

with open(opts.output_labels_json, 'w') as f:
    json.dump(all_labels, f)
