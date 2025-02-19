import torch
import onnx
from onnx.shape_inference import infer_shapes
import onnxsim
import onnx
from onnx import helper
from transformers import Qwen2_5_VLForConditionalGenerationExport, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info


class Merger(torch.nn.Module):
    def __init__(self, merger):
        super().__init__()
        self.merger = merger
    def forward(self, hidden_states, window_index):


        hidden_states = self.merger(hidden_states)
        reverse_indices = torch.argsort(window_index)
        hidden_states = hidden_states[reverse_indices, :]

        return hidden_states


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


# default: Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGenerationExport.from_pretrained(
    "./", torch_dtype="auto", device_map="cpu"
)

# export_model = model.visual
# export_model.forward = export_model.forward_export
device = torch.device("cpu")

model = model.to(device)
# h = torch.load("hidden_states.pth").to(device)
# thw = torch.load("grid_thw.pth").to(device)
# print("h",h.shape)
# print("thw",thw.shape)

hidden_states = torch.load("hidden_states.pth").to(device)
rotary_pos_emb = torch.load("rotary_pos_emb.pth").to(device)
cu_seqlens = torch.load("cu_seqlens.pth").to(device)
cu_window_seqlens = torch.load("cu_window_seqlens.pth").to(device)

seq_length = hidden_states.shape[0]

attention_mask = generate_attnmask(seq_length, cu_seqlens, device)
attention_mask_window = generate_attnmask(seq_length, cu_window_seqlens, device)

window_index = torch.load("window_index.pth").to(device)
# input = ( hidden_states, rotary_pos_emb, attention_mask, attention_mask_window, window_index)



for layer_num, blk in enumerate(model.visual.blocks):

    name_h =  f"hidden_states_in"

    if layer_num in model.visual.fullatt_block_indexes:
        attention_mask_now = attention_mask
        input_names=[name_h, "attn_mask", "rotary_pos_emb" ]
    else:
        attention_mask_now = attention_mask_window
        input_names=[name_h, "window_attn_mask", "rotary_pos_emb" ]

    
    input = (hidden_states, attention_mask_now, rotary_pos_emb)

    onnx_output = f"Qwen2.5-VL-3B-Instruct_vision_block{layer_num}.onnx"

    output_names = [f"hidden_states_out"]

    blk = blk.to(device)
    export_onnx(blk, input, input_names, output_names, onnx_output)    

    hidden_states = blk(
        hidden_states,
        attention_mask=attention_mask_now,
        rotary_pos_emb=rotary_pos_emb,
    )



merger = Merger(model.visual.merger)
input = (hidden_states, window_index)
input_names = ("hidden_states_in", "window_index")
output_names = ["image_embed"]
onnx_output = f"Qwen2.5-VL-3B-Instruct_vision_merger.onnx"

export_onnx(merger, input, input_names, output_names, onnx_output)    



