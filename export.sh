export CUDA_VISIBLE_DEVICES=7

python run.py
python export.py 

python export_vision.py
python merge_vision.py