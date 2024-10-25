set -ex

export R=`dts`
mkdir runs/$R
echo '{"height_width":64, "filter_sizes":[16,32,64,256], "embedding_dim":128}' \
 > runs/$R/embedding_config.json

time python3 v15_train.py \
 --model-config runs/$R/embedding_config.json \
 --num-batches 2000 \
 --learning-rate 1e-4 \
 --num-obj-references 8 \
 --num-contrastive-examples 16 \
 --weights-pkl runs/$R/weights.pkl \
 --losses-json runs/$R/losses.json
 # ~2min for 5000 batches

python3 plot_losses.py --losses-json runs/$R/losses.json

time python3 embed.py \
 --model-config-json runs/$R/embedding_config.json \
 --weights-pkl runs/$R/weights.pkl \
 --manifest test.reference_patches.manifest \
 --embeddings-npy runs/$R/test.reference_patches.npy

time python3 plot_near_neighbours.py \
 --manifest test.reference_patches.manifest \
 --embeddings-npy runs/$R/test.reference_patches.npy \
 --idxs "[10000,20000,30000,40000,50000,60000,70000]" \
 --plot-fname runs/$R/nns.png

