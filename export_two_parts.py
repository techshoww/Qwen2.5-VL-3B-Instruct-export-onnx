import torch
import onnx
from onnx.shape_inference import infer_shapes
import onnxsim
import onnx
from onnx import helper
from transformers import  AutoTokenizer, AutoProcessor
from modeling_qwen2_5_vl_export import Qwen2_5_VLForConditionalGenerationExport
from qwen_vl_utils import process_vision_info
import numpy as np 
import os
import sys 

def export_onnx(model, input, input_names, output_names, onnx_output):

    torch.onnx.export(
        model,
        input,
        onnx_output,
        input_names=input_names,
        output_names=output_names,
        opset_version=16,
    )

    onnx_model = onnx.load(onnx_output)
    print("IR 版本:", onnx_model.ir_version)
    print("操作集:", onnx_model.opset_import)
    onnx_model = infer_shapes(onnx_model)
    # convert model
    model_simp, check = onnxsim.simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, onnx_output)
    print("onnx simpilfy successed, and model saved in {}".format(onnx_output))

def generate_attnmask(seq_length, cu_seqlens, device):
    attention_mask = torch.zeros([1, seq_length, seq_length], device=device, dtype=torch.bool)
    for i in range(1, len(cu_seqlens)):
        attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = True

    return attention_mask

checkpoint_dir = sys.argv[1]
# default: Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGenerationExport.from_pretrained(
    checkpoint_dir, torch_dtype=torch.float32, device_map="cpu"
)


device = torch.device("cpu")

hidden_states = torch.load("hidden_states.pth",weights_only=True).to(torch.float32).to(device)

input = ( hidden_states)
input_names = ["hidden_states"]

export_model = model.visual

export_model.forward = export_model.forward_export_part1
onnx_output_part1 = f"Qwen2.5-VL-3B-Instruct_vision_part1.onnx"
output_names = [f"hidden_states_out"]
export_onnx(export_model, input, input_names, output_names, onnx_output_part1)    

hidden_states = torch.zeros((1024,1280), dtype=torch.float32).to(device)
input = ( hidden_states)

export_model.forward = export_model.forward_export_part2
onnx_output_part2 = f"Qwen2.5-VL-3B-Instruct_vision_part2.onnx"
output_names = [f"hidden_states_out"]
export_onnx(export_model, input, input_names, output_names, onnx_output_part2)

