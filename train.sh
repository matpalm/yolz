set -ex
export R=`dts`
mkdir runs/$R
cp models_config_eg.json runs/$R/models_config.json
time python3 train.py \
 --run-dir runs/$R \
 --models-config-json runs/$R/models_config.json \
 --num-batches 20000 \
 --learning-rate 1e-4 \
 --num-obj-references 8 \
 --num-focus-objs 16 \
 --contrastive-loss-weight 1 \
 --classifier-loss-weight 10 \
 --focal-loss-alpha 0.25 \
 --focal-loss-gamma 2.0 \
 --use-wandb






# ~2min for 5000 batches

#--stop-anchor-gradient \

# python3 plot_losses.py --losses-json runs/$R/losses.json

# time python3 embed.py \
#  --model-config-json runs/$R/embedding_config.json \
#  --weights-pkl runs/$R/weights.pkl \
#  --manifest test.reference_patches.manifest \
#  --embeddings-npy runs/$R/test.reference_patches.npy

# time python3 plot_near_neighbours.py \
#  --manifest test.reference_patches.manifest \
#  --embeddings-npy runs/$R/test.reference_patches.npy \
#  --idxs "[10000,20000,30000,40000,50000,60000,70000]" \
#  --plot-fname runs/$R/nns.png

