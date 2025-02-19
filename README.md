## [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) Vision Encoder 导出 onnx

对于小尺寸的图片，Vision Encoder 可以直接导出。基本上能够满足端侧部署的需求。
如果图片尺寸比较大，直接导出会在onnxsim.simplify时报错。这时可以考虑分层导出，然后再合并。目前分层导出没有问题，但是合并时也会莫名其妙的报错。这条路线暂时搁置。

有些操作不适合在模型中做，我们需要推理过程做一些修改。比如：
* 模型前部对 `hidden_states`和`rotary_pos_emb`的顺序编排可以放在模型外面
* 由 `cu_seqlens`生成`attention_mask`的过程  
* torch 自带的 `torch.nn.functional.scaled_dot_product_attention`导出报错，需要自定义  

本代码依赖[Transformer](https://github.com/huggingface/transformers.git)库
```
transformers                      4.49.0.dev0
```
需要修改其中的 `modeling_qwen2_5_vl.py`和`modeling_qwen2_vl.py`，由于`modeling_qwen2_5_vl.py`是由`modular_qwen2_5_vl.py`生成的，所以同步修改了`modular_qwen2_5_vl.py`。

### 导出过程

#### 对于小尺寸图片输入
1. 生成导出onnx需要的输入
```
python run.py
```
2. 导出onnx
```
python export.py
```

#### 对于大尺寸图片输入
1. 生成导出onnx需要的输入
```
python run.py
```
2. 分层导出onnx
```
python export_vision.py
```
3. 融合onnx  
**目前这一步还没有调通**，融合后的模型load时会报错
```
python merge_vision.py
```