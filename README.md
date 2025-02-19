## [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) Vision Encoder 导出 onnx

对于小尺寸的图片，Vision Encoder 可以直接导出。基本上能够满足端侧部署的需求。
如果图片尺寸比较大，直接导出会在onnxsim.simplify时报错。这时可以考虑分层导出，然后再合并。目前分层导出没有问题，但是合并时也会莫名其妙的报错。这条路线暂时搁置。

### torch_dtype设置  
模型导出时需要设置`torch_dtype=torch.float16`。
1. 设置为 `torch_dtype=torch.float32`，会在 onnxsim.simplify时报错
```
packages/onnxsim/onnx_simplifier.py", line 199, in simplify
    model_opt_bytes = C.simplify(
                      ^^^^^^^^^^^
RuntimeError: The model does not have an ir_version set properly.
```
2. 设置为 `torch_dtype=torch.bfloat16`(auto) 会在onnxruntime阶段报错算子不支持
```
Type Error: Type 'tensor(bfloat16)' of input parameter (hidden_states) of operator (Cast) in node (/blocks.0/norm1/Cast) is invalid.
```
### 代码修改  
有些操作不适合在模型中做，我们需要推理过程做一些修改。比如：
* 模型前部对 `hidden_states`和`rotary_pos_emb`的顺序编排可以放在模型外面
* 由 `cu_seqlens`生成`attention_mask`的过程  
* torch 自带的 `torch.nn.functional.scaled_dot_product_attention`导出报错，需要自定义  
修改逻辑见 modeling_qwen2_5_vl_export.py  

本代码依赖[Transformer](https://github.com/huggingface/transformers.git)库
```
transformers                      4.49.0.dev0
```
安装命令：
```
pip install git+https://github.com/huggingface/transformers.git@v4.49.0
```

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
3. 测试onnx
```
python test_onnx.py
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
**目前这一步还没有调通**，融合后的模型load时会报错。实在融合不了可以分层推理。  
```
python merge_vision.py
```
