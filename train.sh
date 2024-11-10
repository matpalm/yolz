set -ex
export R=`dts`
mkdir runs/$R
cp models_config_eg.json runs/$R/models_config.json
time python3 train.py \
 --run-dir runs/$R \
 --num-batches 20000 \
 --num-obj-references 8 \
 --models-config-json runs/$R/models_config.json \
 --learning-rate 1e-4 \
 --contrastive-loss-weight 1 \
 --classifier-loss-weight 10 \
 --focal-loss-alpha 0.25 \
 --focal-loss-gamma 2.0

 #--use-wandb