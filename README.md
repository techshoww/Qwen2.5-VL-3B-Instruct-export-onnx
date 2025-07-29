## [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) Vision Encoder 导出 onnx

懒人下载链接 https://github.com/AXERA-TECH/Qwen2.5-VL-3B-Instruct.axera/releases/download/v1.0.0/models.tar.gz  

### 一、导出方案
该模型的Vision Encoder 比较大，超过了2G，有些特殊。
所以有三种导出方式：
#### 1. 导出为一个onnx  

后会导出两个文件，一个后缀名为".onnx"，存放计算图。一个后缀名为".onnx.data"，存放权重数据。

#### 2. 导出为多个onnx  
把模型forward逻辑切开，导出为多个onnx。这个模型最少导出为两个部分（图片尺寸448x448），如果图片尺寸比较大，需要切分为更多部分。
这种方式可以使用python API `onnxsim.simplify`。

需要注意这种方式导出的模型虽然可以合并成功，但是合并后再load时还会报错。
合并后onnx.load会报错：
```
google.protobuf.message.DecodeError: Error parsing message with type 'onnx.ModelProto'
```

#### 3. 低精度导出为一个onnx  
将 `torch_dtype` 设置为 `torch.float16` 或 `torch.bfloat16` 时可以导出为一个onnx。
注意若设置为 `torch.bfloat16`，导出的模型不能用python onnxruntime进行推理，模型编译可能也会遇到问题（工具链中如果使用了python）
**原因是 numpy 对 `bfloat16` 支持不完善。可以用C++ onnxruntime 进行 inference.**


**总结一下：**

| 精度 | 导出为一个onnx | 导出为多个onnx |
|------|------|--------|
| float32 | 需要使用二进制onnxsim程序进行simplify。计算图和权重数据分离。   | 可以按照常规onnx处理 |
| float16 | 可以导出为一个onnx。有精度问题，sigmoid和mul融合为SiLu算子时精度误差较大。   | 不需要 |
| bfloat16 | numpy不支持bfloat16，在后续的使用时也会遇到问题。   | 不需要 |


### 二、代码修改  
有些操作不适合在模型中做，我们需要推理过程做一些修改。比如：
* 模型前部对 `hidden_states`和`rotary_pos_emb`的顺序编排可以放在模型外面
* 由 `cu_seqlens`生成`attention_mask`的过程  
* torch 自带的 `torch.nn.functional.scaled_dot_product_attention`导出报错，需要自定义  
修改逻辑见 modeling_qwen2_5_vl_export.py  

本代码依赖[Transformer](https://github.com/huggingface/transformers.git)库
```
transformers                      4.49.0
```
安装命令：
```
pip install transformers==4.49.0
```

### 三、导出过程

#### 导出为一个onnx
1. 生成导出onnx需要的输入
```
python run.py {your checkpoint dir}
```
2. 导出onnx
```
python export.py {your checkpoint dir}
```
3. 测试onnx
```
python test_onnx.py {your checkpoint dir}
```

#### 导出为两个onnx
1. 生成导出onnx需要的输入
```
python run.py {your checkpoint dir}
```
2. 分层导出onnx
```
python export_two_parts.py {your checkpoint dir}
```
3. 测试onnx
```
python test_onnx.py {your checkpoint dir} two_parts
```

## 相关项目  
https://github.com/techshoww/Qwen2.5-VL-3B-Instruct.axera.git  
这里有更多大模型示例 https://github.com/AXERA-TECH
