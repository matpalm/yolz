set -ex

export B=10000

export R=`dts`
mkdir runs/$R
cp models_config_eg.json runs/$R/models_config.json
time python3 v3_train.py \
 --run-dir runs/$R \
 --models-config-json runs/$R/models_config.json \
 --num-batches $B \
 --learning-rate 1e-4 \
 --num-obj-references 8 \
 --num-focus-objs 8 \
 --contrastive-loss-weight 1 \
 --classifier-loss-weight 10 \
 --embedding-dim 64 \
 --feature-dim 64 \
 --stop-anchor-gradient \
 --use-wandb
