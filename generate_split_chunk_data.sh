
set -ex

export NUM_OBJ_EXAMPLES=100  # bump to 1000
export NUM_SCENES=10

python3 -m data_gen.render_reference_egs \
 --urdf-id-from 0 --urdf-id-to 800 \
 --num-examples $NUM_OBJ_EXAMPLES \
 --output-dir data/train/reference_patches \
 --include-alpha &

python3 -m data_gen.render_scenes \
  --urdf-id-from 0 --urdf-id-to 800 \
  --num-scenes $NUM_SCENES \
  --output-dir data/train/scenes &

python3 -m data_gen.render_reference_egs \
 --urdf-id-from 801 --urdf-id-to 900 \
 --num-examples $NUM_OBJ_EXAMPLES \
 --output-dir data/validation/reference_patches \
 --include-alpha &

python3 -m data_gen.render_scenes \
  --urdf-id-from 801 --urdf-id-to 900 \
  --num-scenes $NUM_SCENES \
  --output-dir data/validation/scenes &

python3 -m data_gen.render_reference_egs \
 --urdf-id-from 901 --urdf-id-to 1000 \
 --num-examples $NUM_OBJ_EXAMPLES \
 --output-dir data/test/reference_patches \
 --include-alpha &

python3 -m data_gen.render_scenes \
  --urdf-id-from 901 --urdf-id-to 1000 \
  --num-scenes $NUM_SCENES \
  --output-dir data/test/scenes &

wait
