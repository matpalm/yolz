
import argparse
import time, sys, tqdm

from data import ContrastiveExamples

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--seed', type=int, required=True)
parser.add_argument('--root-dir', type=str, required=True)
parser.add_argument('--cache-dir', type=str, required=True)
opts = parser.parse_args()
print("opts", opts)

ce = ContrastiveExamples(
    root_dir=opts.root_dir,
    seed=opts.seed,
    random_background_colours=True,
    instances_per_obj=8,
    cache_dir=opts.cache_dir
)

for _ in tqdm.tqdm(ce.tf_dataset(), total=len(ce.scenes)):
    pass