set -ex

for A in 0.25 0.5 0.75; do
  for G in 0.5 1.0 2.0; do
    export R=`dts`
    mkdir runs/$R
    cp models_config_eg.json runs/$R/models_config.json
    time python3 v3_train.py \
     --run-dir runs/$R \
     --models-config-json runs/$R/models_config.json \
     --num-batches 5000 \
     --learning-rate 1e-4 \
     --num-obj-references 8 \
     --num-focus-objs 8 \
     --contrastive-loss-weight 1 \
     --classifier-loss-weight 10 \
     --focal-loss-alpha $A \
     --focal-loss-gamma $G \
     --use-wandb
    done
done

