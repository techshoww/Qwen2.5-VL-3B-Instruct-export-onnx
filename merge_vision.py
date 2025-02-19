import onnx
from onnx import helper, numpy_helper

def merge_onnx_models(model1_path, model2_path, output_path, prefix):
    # 加载原始模型
    model1 = onnx.load(model1_path)
    model2 = onnx.load(model2_path)

    # 创建新图组件
    merged_nodes = []
    merged_inputs = []
    merged_outputs = []
    merged_initializers = []

    # 处理第一个模型
    merged_inputs = list(model1.graph.input)
    merged_nodes.extend(model1.graph.node)
    merged_initializers.extend(model1.graph.initializer)
    

    # 处理第二个模型（添加前缀防止冲突）
    # prefix = "model2_"
    node_mapping = {}
    
    for init in model2.graph.initializer:
        # 创建带前缀的权重名称
        new_init_name = prefix + init.name
        # 复制权重数据并重命名
        # new_init = helper.make_tensor(
        #     name=new_init_name,
        #     data_type=init.data_type,
        #     dims=init.dims,
        #     vals=onnx.numpy_helper.to_array(init).flatten().tolist()
        # )
        new_init = onnx.TensorProto()
        new_init.CopyFrom(init)
        new_init.name = new_init_name

        merged_initializers.append(new_init)
        # 更新节点输入映射关系
        node_mapping[init.name] = new_init_name



    shared_inputs = set([i.name for i in model1.graph.input]) & set([i.name for i in model2.graph.input])
    shared_inputs.remove("hidden_states_in")

    not_shared_inputs = []
    # 处理输入重命名
    for inp in model2.graph.input:
        if inp.name=="hidden_states_in":
            new_name = prefix + inp.name
            node_mapping[inp.name] = new_name
            
            # merged_inputs.append(helper.make_tensor_value_info(
            #     new_name,
            #     inp.type.tensor_type.elem_type,
            #     [d.dim_value for d in inp.type.tensor_type.shape.dim]
            # ))
        elif inp.name not in shared_inputs:
            not_shared_inputs.append(inp.name)
            merged_inputs.append(inp)

    def is_weight_input(input_name, model):
        return any(init.name == input_name for init in model.graph.initializer)

    # 处理节点重命名
    for node in model2.graph.node:
        # new_inputs = [node_mapping.get(i, prefix+i) for i in node.input]
        new_inputs = []
        for i in node.input:
            if i in node_mapping:
                new_inputs.append(node_mapping[i])
            # if is_weight_input(i, model2):  # 判断是否为权重参数
            #     new_inputs.append(i)  
            elif i in not_shared_inputs:
                new_inputs.append(i)
            else:
                new_inputs.append(prefix+i)
        new_outputs = [prefix + o for o in node.output]
        new_node = helper.make_node(
            node.op_type,
            new_inputs,
            new_outputs,
            name=prefix + node.name,
            # **{attr.name: attr for attr in node.attribute}
        )
        merged_nodes.append(new_node)
        # 更新映射关系
        for orig_out, new_out in zip(node.output, new_outputs):
            node_mapping[orig_out] = new_out

    # 处理共享输入参数
    
    for shared_in in shared_inputs:
        # 创建共享输入连接节点
        connector = helper.make_node(
            'Identity',
            inputs=[shared_in],
            outputs=[prefix + shared_in],
            name=f'share_{shared_in}'
        )
        merged_nodes.append(connector)

    # 连接两个模型：model1输出 -> model2输入
    connector_node = helper.make_node(
        'Identity',
        inputs=[model1.graph.output[0].name],
        outputs=[node_mapping[model2.graph.input[0].name]],
        name='model_connector'
    )
    merged_nodes.append(connector_node)

    # 构建新计算图
    merged_graph = helper.make_graph(
        nodes=merged_nodes,
        name="merged_model",
        inputs=merged_inputs,
        outputs=[helper.make_tensor_value_info(
            node_mapping[model2.graph.output[0].name],
            model2.graph.output[0].type.tensor_type.elem_type,
            [d.dim_value for d in model2.graph.output[0].type.tensor_type.shape.dim]
        )],
        initializer=merged_initializers
    )
    
    # 创建并保存新模型
    merged_model = helper.make_model(merged_graph)
    # onnx.checker.check_model(merged_model)
    onnx.save(merged_model, output_path)

# 使用示例
for i in range(1,32):
    if i==1:
        merge_onnx_models(f"Qwen2.5-VL-3B-Instruct_vision_block{i-1}.onnx", f"Qwen2.5-VL-3B-Instruct_vision_block{i}.onnx", f"Qwen2.5-VL-3B-Instruct_vision_block0-{i}.onnx", f"block{i}_")
    else:
        merge_onnx_models(f"Qwen2.5-VL-3B-Instruct_vision_block0-{i-1}.onnx", f"Qwen2.5-VL-3B-Instruct_vision_block{i}.onnx", f"Qwen2.5-VL-3B-Instruct_vision_block0-{i}.onnx", f"block{i}_")
    print(i)
merge_onnx_models(f"Qwen2.5-VL-3B-Instruct_vision_block0-{i}.onnx", "Qwen2.5-VL-3B-Instruct_vision_merger.onnx", f"Qwen2.5-VL-3B-Instruct_vision.onnx", f"merger_")


