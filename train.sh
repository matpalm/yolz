
set -ex

if [ ! -d "/dev/shm/zero_shot_detection/data/train" ]; then
    echo "copying data/train to /dev/shm"
    mkdir -p /dev/shm/zero_shot_detection/data/
    cp -r data/* /dev/shm/zero_shot_detection/data/
fi

export R=`dts`
mkdir runs/$R
time python3 train.py \
 --run-dir runs/$R \
 --num-repeats 10 \
 --learning-rate 1e-4 \
 --contrastive-loss-weight 0 \
 --use-wandb
#  --focal-loss-alpha 0.25 \
#  --focal-loss-gamma 2.0

 #