set -ex
export NE=1000

python3 -m data_gen.render_reference_egs \
 --num-examples $NE \
 --output-dir data/train/reference_patches \
 --urdf-id-from 0 --urdf-id-to 640
python3 -m data_gen.render_reference_egs \
 --num-examples $NE \
 --output-dir data/train/scene_patches \
 --urdf-id-from 640 --urdf-id-to 800

 python3 -m data_gen.render_reference_egs \
 --num-examples $NE \
 --output-dir data/validate/reference_patches \
 --urdf-id-from 800 --urdf-id-to 880
python3 -m data_gen.render_reference_egs \
 --num-examples $NE \
 --output-dir data/validate/scene_patches \
 --urdf-id-from 880 --urdf-id-to 900

 python3 -m data_gen.render_reference_egs \
 --num-examples $NE \
 --output-dir data/test/reference_patches \
 --urdf-id-from 900 --urdf-id-to 980
python3 -m data_gen.render_reference_egs \
 --num-examples $NE \
 --output-dir data/test/scene_patches \
 --urdf-id-from 980 --urdf-id-to 1000