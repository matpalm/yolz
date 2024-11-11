set -ex
export R=`dts`
mkdir runs/$R
time python3 train.py \
 --run-dir runs/$R \
 --num-repeats 1 \
 --learning-rate 1e-4
#  --contrastive-loss-weight 1 \
#  --classifier-loss-weight 10 \
#  --focal-loss-alpha 0.25 \
#  --focal-loss-gamma 2.0

 #--use-wandb