
set -ex

export R=`dts`
mkdir runs/$R
time python3 train.py \
 --run-dir runs/$R \
 --train-root-dir data/train_sm \
 --num-repeats 100 \
 --optimiser adamw \
 --learning-rate 1e-4 \
 --use-wandb

 #