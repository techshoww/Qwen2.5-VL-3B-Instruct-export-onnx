export CUDA_VISIBLE_DEVICES=7

set -e 

python run.py
python export.py 
python test_onnx.py

# python export_two_parts.py
# python test_onnx.py two_parts