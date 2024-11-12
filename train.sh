
set -ex

if [ ! -d "/dev/shm/zero_shot_detection/data/train" ]; then
    echo "copying data/train to /dev/shm"
    mkdir -p /dev/shm/zero_shot_detection/data/
    cp -r data/train/ /dev/shm/zero_shot_detection/data/
fi

export R=`dts`
mkdir runs/$R
time python3 train.py \
 --run-dir runs/$R \
 --num-repeats 100 \
 --learning-rate 1e-4
#  --contrastive-loss-weight 1 \
#  --classifier-loss-weight 10 \
#  --focal-loss-alpha 0.25 \
#  --focal-loss-gamma 2.0

 #--use-wandb