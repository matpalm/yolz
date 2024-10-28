set -ex

export R=`dts`
mkdir runs/$R
cp models_config_eg.json runs/$R/models_config.json
time python3 v15_train.py \
 --run-dir runs/$R \
 --models-config-json runs/$R/models_config.json \
 --num-batches 1000 \
 --learning-rate 1e-4 \
 --num-obj-references 8 \
 --num-focus-objs 8 \
 --losses-json runs/$R/losses.json
 # ~2min for 5000 batches

#python3 plot_losses.py --losses-json runs/$R/losses.json

find data/test/reference_patches/ -type f > test.reference_patches.manifest

time python3 embed.py \
 --models-config-json runs/$R/models_config.json \
 --weights-pkl runs/$R/models_weights.pkl \
 --manifest test.reference_patches.manifest \
 --embeddings-npy runs/$R/test.reference_patches.npy

time python3 plot_near_neighbours.py \
 --manifest test.reference_patches.manifest \
 --embeddings-npy runs/$R/test.reference_patches.npy \
 --idxs "[10000,20000,30000,40000,50000,60000,70000]" \
 --plot-fname runs/$R/nns.png

