# zero shot detection

## generate data

render examples of all urdfs

```
mkdir data/all_egs
python3 -m data_gen.render_all_egs
```

( see also random urdfs https://github.com/matpalm/procedural_objects )

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
 --urdf-ids '["061","135","182","111","153","198","000","017","019"]' \
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