
set -ex

export NUM_OBJ_EXAMPLES=1000
export NUM_SCENES=1000

# python3 -m data_gen.render_reference_egs \
#  --urdf-id-from 0 --urdf-id-to 800 \
#  --num-examples $NUM_OBJ_EXAMPLES \
#  --output-dir data/train/reference_patches \
#  --include-alpha &

python3 -m data_gen.render_scenes \
  --urdf-id-from 0 --urdf-id-to 800 \
  --num-scenes $NUM_SCENES \
  --base-scene-id 6000 --seed 6000 \
  --output-dir data/train/scenes &

python3 -m data_gen.render_scenes \
  --urdf-id-from 0 --urdf-id-to 800 \
  --num-scenes $NUM_SCENES \
  --base-scene-id 7000 --seed 7000 \
  --output-dir data/train/scenes &

python3 -m data_gen.render_scenes \
  --urdf-id-from 0 --urdf-id-to 800 \
  --num-scenes $NUM_SCENES \
  --base-scene-id 8000 --seed 8000 \
  --output-dir data/train/scenes &

# python3 -m data_gen.render_reference_egs \
#  --urdf-id-from 801 --urdf-id-to 900 \
#  --num-examples $NUM_OBJ_EXAMPLES \
#  --output-dir data/validation/reference_patches \
#  --include-alpha &

# python3 -m data_gen.render_scenes \
#   --urdf-id-from 801 --urdf-id-to 900 \
#   --num-scenes $NUM_SCENES \
#   --output-dir data/validation/scenes &

# python3 -m data_gen.render_reference_egs \
#  --urdf-id-from 901 --urdf-id-to 1000 \
#  --num-examples $NUM_OBJ_EXAMPLES \
#  --output-dir data/test/reference_patches \
#  --include-alpha &

# python3 -m data_gen.render_scenes \
#   --urdf-id-from 901 --urdf-id-to 1000 \
#   --num-scenes $NUM_SCENES \
#   --output-dir data/test/scenes &

time wait
