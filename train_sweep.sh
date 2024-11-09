set -ex

for W in 1 100; do
    export R=`dts`
    mkdir runs/$R
    cp models_config_eg.json runs/$R/models_config.json
    time python3 train.py \
        --run-dir runs/$R \
        --models-config-json runs/$R/models_config.json \
        --num-batches 10000 \
        --learning-rate 1e-4 \
        --num-obj-references 8 \
        --num-focus-objs 8 \
        --embedding-dim 128 \
        --feature-dim 128 \
        --contrastive-loss-weight 1 \
        --classifier-loss-weight $W \
        --focal-loss-alpha 0.25 \
        --focal-loss-gamma 2.0 \
        --use-wandb
done
