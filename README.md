# zero shot detection

## generate data

1. render examples of all urdfs for visual debugging

```
mkdir data/all_egs
python3 -m data_gen.render_all_egs
```

( see also random urdfs https://github.com/matpalm/procedural_objects )

2. render `data/{train,validate,test}/{reference,scene}_patches`

```
sh generate_split_chunk_data.sh
```

80/10/10 for train/validate/test
80/20 for reference/scene patches

## train / test v1 model

train v1 embedding model

```
python3 v1_train.py

opts Namespace(num_batches=1000, batch_size=128,
objs_per_batch=32, height_width=64,
eg_root_dir='data/train/reference_patches', eg_obj_ids_json=None,
embedding_dim=128, learning_rate=0.001, weights_pkl='weights/v1.pkl')
```

embed all validation examples

```
find data/test/reference_patches/ -type f > test_reference_patches.manifest
python3 embed.py \
  --manifest test_reference_patches.manifest \
  --weights-pkl weights/v1.pkl \
  --embeddings-npy embeddings/v1.npy
```



# REWRITE BELOW

POC training

red objects; 061, 135, 182
green objects; 111, 153, 198
blue objects; 000, 017, 019

full training

all objects up to 500

test

red objects; 675, 681, 703
green objects; 631, 659, 677
blue objects; 619, 633, 634

## generate reference image data

this is the data fed to the reference embeddings branch

```
python3 -m data_gen.render_reference_egs \
 --urdf-ids '["061","135","182","111","153","198","000","017","019"]' \
 --num-examples 1000 \
 --output-dir data/reference_egs
```

next we get additional data, of same objects, that's used for the image branch

```
python3 -m data_gen.render_reference_egs \
 --urdf-ids '["061","135","182", "111","153","198","000","017","019"]' \
 --num-examples 100 \
 --output-dir data/scene_egs
```

and stitch these info reference images.
( each y_true is NxN image with NxN label of urdf or -1 )

```
python3 -m data_gen.stitch_egs_into_y_true.py \
 --reference-egs-dir data/scene_egs \
 --output-dir data/y_true_imgs \
 --num-egs 20 \
 --output-labels-json data/y_true.json \
 --grid-size 10 \
 --grid-spacing 64 \
 --num-objs 10
```

model takes samples from the `reference_egs` for the reference obj embeddings
and takes samples from y_true_imgs ( with y_true.json ) for the image input