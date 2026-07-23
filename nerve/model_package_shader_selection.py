from nerve.model_package_common import *
from nerve.model_package_assets import stream_control_binding_for_node
from nerve.model_package_tensors import *

def feed_forward_intermediate_size(circuit: Json) -> int:
    for node in circuit.get("nodes", []):
        if node.get("id") == "ffn_gate_activation":
            width = int(node.get("attrs", {}).get("element_count", 0))
            if width > 0:
                return width
    raise ModelCompileError(
        f"circuit {circuit.get('id')!r} does not describe its feed-forward width"
    )


def shader_file_for_node(
    circuit: Json,
    node: Json,
    tensor_index: Json,
    dimensions: Json,
) -> str:
    hidden_size = int(dimensions["hidden_size"])
    op = node["op"]

    if op == "rms_norm":
        return rms_norm_shader_file(
            hidden_size,
            float(node["attrs"]["eps"]),
            float(node["attrs"]["weight_offset"]),
        )
    if op == "linear":
        parameter_shape = parameter_shape_for_node(circuit, node, tensor_index)
        out_features, in_features = parameter_shape
        parameter_dtype = parameter_dtype_for_node(circuit, node, tensor_index)
        if parameter_dtype == "I32":
            quantization_format = packed_linear_quantization_format_for_node(
                circuit, node, tensor_index
            )
            if quantization_format == "auto_gptq":
                group_size = packed_int4_linear_group_size_for_node(
                    circuit, node, tensor_index
                )
                format_token = "gptq"
                has_bias = len(node.get("params", [])) == 4
            elif quantization_format == "compressed_tensors_pack_quantized":
                group_size = compressed_tensors_int4_group_size_for_node(
                    circuit, node, tensor_index
                )
                format_token = "ct"
                has_bias = len(node.get("params", [])) == 3
            else:
                raise ModelCompileError(
                    f"linear node {node['id']!r} has unsupported packed format "
                    f"{quantization_format!r}"
                )
            scale_dtype = packed_int4_scale_dtype_for_node(
                circuit, node, tensor_index
            ).lower()
            prefix = "linear_bias" if has_bias else "linear"
            return (
                f"{prefix}_int4_{format_token}_s{scale_dtype}_g{group_size}_"
                f"{in_features}x{out_features}.comp"
            )
        if parameter_dtype == "F8_E4M3":
            block_rows, block_columns = fp8_block_shape_for_node(
                circuit, node, tensor_index
            )
            has_bias = len(node.get("params", [])) == 3
            prefix = "linear_bias" if has_bias else "linear"
            return (
                f"{prefix}_fp8_e4m3_b{block_rows}x{block_columns}_"
                f"{in_features}x{out_features}.comp"
            )
        if parameter_dtype == "Q8_0":
            out_features, in_features = q8_0_linear_shape_for_node(
                circuit, node, tensor_index
            )
            has_bias = len(node.get("params", [])) == 2
            prefix = "linear_bias" if has_bias else "linear"
            return f"{prefix}_q8_0_{in_features}x{out_features}.comp"
        if parameter_dtype != "BF16":
            raise ModelCompileError(
                f"linear node {node['id']!r} has unsupported weight dtype "
                f"{parameter_dtype}"
            )
        layout = parameter_layout_for_node(circuit, node, tensor_index)
        if layout != ROW_MAJOR_LAYOUT:
            raise ModelCompileError(
                f"linear node {node['id']!r} has unsupported layout {layout!r}"
            )
        has_bias = len(node.get("params", [])) == 2
        prefix = "linear"
        if has_bias:
            prefix += "_bias"
        prefix += "_bf16"
        return f"{prefix}_{in_features}x{out_features}.comp"
    if op in {"parallel_linear_2way", "parallel_linear_3way"}:
        expected_branch_count = 2 if op == "parallel_linear_2way" else 3
        branch_count = int(node["attrs"]["branch_count"])
        branch_parameter_counts = [
            int(count)
            for count in node["attrs"].get(
                "branch_parameter_counts", [1] * branch_count
            )
        ]
        if (
            branch_count != expected_branch_count
            or len(branch_parameter_counts) != branch_count
            or sum(branch_parameter_counts) != len(node["params"])
            or branch_count != len(node["outputs"])
        ):
            raise ModelCompileError(
                f"parallel-linear node {node['id']!r} has inconsistent branch metadata"
            )
        branch_params = []
        offset = 0
        for count in branch_parameter_counts:
            branch_params.append(node["params"][offset : offset + count])
            offset += count
        shapes = [
            parameter_shape_for_id(circuit, parameter_ids[0], tensor_index)
            for parameter_ids in branch_params
        ]
        input_widths = {int(shape[1]) for shape in shapes if len(shape) == 2}
        dtypes = {
            parameter_dtype_for_id(circuit, parameter_ids[0], tensor_index)
            for parameter_ids in branch_params
        }
        if (
            len(shapes) != branch_count
            or any(len(shape) != 2 for shape in shapes)
            or len(input_widths) != 1
            or dtypes not in ({"BF16"}, {"F8_E4M3"})
        ):
            raise ModelCompileError(
                f"parallel-linear node {node['id']!r} has incompatible shapes {shapes}"
            )
        output_widths = [int(shape[0]) for shape in shapes]
        layouts = {
            parameter_layout_for_id(circuit, parameter_ids[0], tensor_index)
            for parameter_ids in branch_params
        }
        if layouts != {ROW_MAJOR_LAYOUT}:
            raise ModelCompileError(
                f"parallel-linear node {node['id']!r} has unsupported layouts "
                f"{sorted(layouts)}"
            )
        if dtypes == {"F8_E4M3"}:
            block_shapes = {
                fp8_block_shape_for_node(
                    circuit,
                    {
                        "id": f"{node['id']}__branch_{index}",
                        "params": parameter_ids,
                    },
                    tensor_index,
                )
                for index, parameter_ids in enumerate(branch_params)
            }
            if len(block_shapes) != 1 or any(
                len(parameter_ids) != 2 for parameter_ids in branch_params
            ):
                raise ModelCompileError(
                    f"parallel-linear FP8 node {node['id']!r} has incompatible block scales"
                )
            block_rows, block_columns = block_shapes.pop()
            if len(set(output_widths)) != 1:
                raise ModelCompileError(
                    f"parallel-linear FP8 node {node['id']!r} requires equal output widths"
                )
            input_width = input_widths.pop()
            return (
                f"parallel_linear_{branch_count}way_fp8_e4m3_"
                f"b{block_rows}x{block_columns}_{input_width}x{output_widths[0]}.comp"
            )
        input_width = input_widths.pop()
        return (
            f"parallel_linear_{branch_count}way_bf16_{input_width}x"
            + "_".join(map(str, output_widths))
            + ".comp"
        )
    if op == "parallel_linear_silu_multiply":
        params = node.get("params", [])
        if (
            len(node.get("inputs", [])) != 1
            or len(node.get("outputs", [])) != 1
            or int(node.get("attrs", {}).get("branch_count", 0)) != 2
        ):
            raise ModelCompileError(
                f"fused FFN projection node {node['id']!r} has invalid bindings"
            )
        if len(params) == 2:
            weight_ids = params
            shapes = [
                parameter_shape_for_id(circuit, parameter_id, tensor_index)
                for parameter_id in weight_ids
            ]
            dtypes = {
                parameter_dtype_for_id(circuit, parameter_id, tensor_index)
                for parameter_id in weight_ids
            }
            layouts = {
                parameter_layout_for_id(circuit, parameter_id, tensor_index)
                for parameter_id in weight_ids
            }
            if (
                len(shapes) != 2
                or shapes[0] != shapes[1]
                or len(shapes[0]) != 2
                or dtypes != {"BF16"}
                or layouts != {ROW_MAJOR_LAYOUT}
            ):
                raise ModelCompileError(
                    f"fused FFN projection node {node['id']!r} has incompatible "
                    f"parameters {shapes}"
                )
            block_shape = None
        elif len(params) == 4:
            weight_ids = [params[0], params[2]]
            shapes = [
                parameter_shape_for_id(circuit, parameter_id, tensor_index)
                for parameter_id in weight_ids
            ]
            branch_params = [params[:2], params[2:]]
            block_shapes = {
                fp8_block_shape_for_node(
                    circuit,
                    {
                        "id": f"{node['id']}__branch_{index}",
                        "params": parameter_ids,
                    },
                    tensor_index,
                )
                for index, parameter_ids in enumerate(branch_params)
            }
            if (
                len(shapes) != 2
                or shapes[0] != shapes[1]
                or len(shapes[0]) != 2
                or len(block_shapes) != 1
                or any(
                    parameter_dtype_for_id(circuit, parameter_id, tensor_index)
                    != "F8_E4M3"
                    or parameter_layout_for_id(circuit, parameter_id, tensor_index)
                    != ROW_MAJOR_LAYOUT
                    for parameter_id in weight_ids
                )
            ):
                raise ModelCompileError(
                    f"fused FFN projection node {node['id']!r} has incompatible "
                    f"parameters {shapes}"
                )
            block_shape = block_shapes.pop()
        else:
            raise ModelCompileError(
                f"fused FFN projection node {node['id']!r} has invalid parameter count "
                f"{len(params)}"
            )
        output_width, input_width = map(int, shapes[0])
        if (
            input_width <= 0
            or input_width % 2
            or output_width <= 0
            or output_width % 2
            or int(node["attrs"].get("element_count", 0)) != output_width
            or node["attrs"].get("intermediate_rounding") != "BF16"
        ):
            raise ModelCompileError(
                f"fused FFN projection node {node['id']!r} has invalid geometry"
            )
        if block_shape is not None:
            block_rows, block_columns = block_shape
            return (
                "parallel_linear_silu_multiply_fp8_e4m3_"
                f"b{block_rows}x{block_columns}_{input_width}x{output_width}.comp"
            )
        return f"parallel_linear_silu_multiply_bf16_{input_width}x{output_width}.comp"
    if op == "linear_split_3way":
        parameter_shape = parameter_shape_for_node(circuit, node, tensor_index)
        out_features, in_features = map(int, parameter_shape)
        if parameter_dtype_for_node(circuit, node, tensor_index) != "BF16":
            raise ModelCompileError(
                f"linear-split node {node['id']!r} requires BF16 weights"
            )
        part_widths = [int(width) for width in node["attrs"]["part_widths"]]
        if (
            len(part_widths) != 3
            or any(width <= 0 or width % 2 for width in part_widths)
            or sum(part_widths) != out_features
        ):
            raise ModelCompileError(
                f"linear-split node {node['id']!r} cannot partition {out_features} "
                f"outputs into {part_widths}"
            )
        layout = parameter_layout_for_node(circuit, node, tensor_index)
        if layout != ROW_MAJOR_LAYOUT:
            raise ModelCompileError(
                f"linear-split node {node['id']!r} has unsupported layout {layout!r}"
            )
        return (
            f"linear_split_3way_bf16_{in_features}x"
            + "_".join(map(str, part_widths))
            + ".comp"
        )
    if op == "linear_residual":
        parameter_shape = parameter_shape_for_node(circuit, node, tensor_index)
        out_features, in_features = parameter_shape
        parameter_dtype = parameter_dtype_for_node(circuit, node, tensor_index)
        if parameter_dtype == "I32":
            quantization_format = packed_linear_quantization_format_for_node(
                circuit, node, tensor_index
            )
            if quantization_format == "auto_gptq":
                group_size = packed_int4_linear_group_size_for_node(
                    circuit, node, tensor_index
                )
                format_token = "gptq"
            elif quantization_format == "compressed_tensors_pack_quantized":
                group_size = compressed_tensors_int4_group_size_for_node(
                    circuit, node, tensor_index
                )
                format_token = "ct"
            else:
                raise ModelCompileError(
                    f"linear-residual node {node['id']!r} has unsupported packed "
                    f"format {quantization_format!r}"
                )
            scale_dtype = packed_int4_scale_dtype_for_node(
                circuit, node, tensor_index
            ).lower()
            return (
                f"linear_residual_int4_{format_token}_s{scale_dtype}_g{group_size}_"
                f"{in_features}x{out_features}.comp"
            )
        if parameter_dtype == "F8_E4M3":
            block_rows, block_columns = fp8_block_shape_for_node(
                circuit, node, tensor_index
            )
            return (
                f"linear_residual_fp8_e4m3_b{block_rows}x{block_columns}_"
                f"{in_features}x{out_features}.comp"
            )
        if parameter_dtype == "Q8_0":
            out_features, in_features = q8_0_linear_shape_for_node(
                circuit, node, tensor_index
            )
            return f"linear_residual_q8_0_{in_features}x{out_features}.comp"
        if parameter_dtype != "BF16":
            raise ModelCompileError(
                f"linear-residual node {node['id']!r} has unsupported weight dtype "
                f"{parameter_dtype}"
            )
        layout = parameter_layout_for_node(circuit, node, tensor_index)
        if layout != ROW_MAJOR_LAYOUT:
            raise ModelCompileError(
                f"linear-residual node {node['id']!r} has unsupported layout {layout!r}"
            )
        return f"linear_residual_bf16_{in_features}x{out_features}.comp"
    if op == "split":
        if node["attrs"].get("layout") == "per_head_interleaved":
            return (
                f"split_bf16_2x{node['attrs']['blocks']}x{node['attrs']['block_part_width']}"
                "_head_interleaved.comp"
            )
        if node["attrs"].get("part_widths") is not None:
            part_widths = [int(width) for width in node["attrs"]["part_widths"]]
            if len(part_widths) != 3:
                raise ModelCompileError(
                    f"split node {node['id']!r} has unsupported unequal part widths {part_widths}"
                )
            return "split_bf16_3x" + "_".join(map(str, part_widths)) + ".comp"
        part_width = int(node["attrs"]["part_width"])
        return f"split_bf16_{len(node['outputs'])}x{part_width}.comp"
    if op == "concatenate":
        part_widths = [int(width) for width in node["attrs"]["part_widths"]]
        if (
            node["attrs"].get("axis") != "channel"
            or len(node.get("inputs", [])) != len(part_widths)
            or len(node.get("outputs", [])) != 1
            or any(width <= 0 or width % 2 for width in part_widths)
        ):
            raise ModelCompileError(
                f"concatenate node {node['id']!r} has unsupported geometry"
            )
        return "concatenate_bf16_" + "_".join(map(str, part_widths)) + ".comp"
    if op == "multiply":
        element_count = int(
            node.get("attrs", {}).get(
                "element_count",
                (
                    feed_forward_intermediate_size(circuit)
                    if node["id"] == "ffn_gate_multiply"
                    else hidden_size
                ),
            )
        )
        return f"multiply_bf16_{element_count}.comp"
    if op == "scalar_multiply":
        return f"scalar_multiply_bf16_{int(node['attrs']['element_count'])}.comp"
    if op == "rolling_state_update":
        temporal_memory = state_port(circuit, "temporal_memory")
        frames, state_hidden = temporal_memory["shape"]
        return f"rolling_state_update_bf16_{frames}x{state_hidden}.comp"
    if op == "depthwise_conv1d":
        temporal_memory = state_port(circuit, "temporal_memory")
        frames, state_hidden = temporal_memory["shape"]
        return f"depthwise_conv1d_bf16_{frames}x{state_hidden}.comp"
    if op in {"multiply_rolling_depthwise", "multiply_rolling_depthwise_gate"}:
        expected_input_count = 4 if op.endswith("_gate") else 3
        if (
            len(node.get("inputs", [])) != expected_input_count
            or len(node.get("outputs", [])) != 1
            or len(node.get("params", [])) != 1
            or len(node.get("state_reads", [])) != 1
            or node.get("state_reads") != node.get("state_writes")
        ):
            raise ModelCompileError(
                f"fused recurrent convolution node {node['id']!r} has invalid bindings"
            )
        temporal_memory = state_port(circuit, node["state_reads"][0])
        frames, state_hidden = map(int, temporal_memory["shape"])
        kernel_shape = parameter_shape_for_node(circuit, node, tensor_index)
        supported_kernel_shapes = ([state_hidden, frames], [state_hidden, 1, frames])
        if (
            temporal_memory.get("dtype") != "BF16"
            or frames < 2
            or state_hidden <= 0
            or state_hidden % 2
            or kernel_shape not in supported_kernel_shapes
            or parameter_dtype_for_node(circuit, node, tensor_index) != "BF16"
            or parameter_layout_for_node(circuit, node, tensor_index)
            != ROW_MAJOR_LAYOUT
        ):
            raise ModelCompileError(
                f"fused recurrent convolution node {node['id']!r} has incompatible "
                f"state {temporal_memory.get('shape')} or kernel {kernel_shape}"
            )
        shader_prefix = (
            "multiply_rolling_depthwise_gate"
            if op.endswith("_gate")
            else "multiply_rolling_depthwise"
        )
        return f"{shader_prefix}_bf16_{frames}x{state_hidden}.comp"
    if op == "linear_split_recurrent_depthwise_gate":
        if (
            len(node.get("inputs", [])) != 2
            or len(node.get("outputs", [])) != 1
            or len(node.get("params", [])) != 2
            or len(node.get("state_reads", [])) != 1
            or node.get("state_reads") != node.get("state_writes")
        ):
            raise ModelCompileError(
                f"projected recurrent convolution node {node['id']!r} has invalid bindings"
            )
        temporal_memory = state_port(circuit, node["state_reads"][0])
        frames, hidden_size = map(int, temporal_memory["shape"])
        projection_shape = parameter_shape_for_id(
            circuit, node["params"][0], tensor_index
        )
        kernel_shape = parameter_shape_for_id(circuit, node["params"][1], tensor_index)
        part_widths = [
            int(width) for width in node["attrs"]["projection"]["part_widths"]
        ]
        input_gate_indices = [
            int(index) for index in node["attrs"]["input_gate_branch_indices"]
        ]
        output_gate_index = int(node["attrs"]["output_gate_branch_index"])
        projection_layout = parameter_layout_for_id(
            circuit, node["params"][0], tensor_index
        )
        if (
            temporal_memory.get("dtype") != "BF16"
            or frames < 2
            or hidden_size <= 0
            or hidden_size % 2
            or len(projection_shape) != 2
            or projection_shape[0] != 3 * hidden_size
            or projection_shape[1] <= 0
            or projection_shape[1] % 2
            or part_widths != [hidden_size] * 3
            or sorted([*input_gate_indices, output_gate_index]) != [0, 1, 2]
            or kernel_shape not in ([hidden_size, frames], [hidden_size, 1, frames])
            or any(
                parameter_dtype_for_id(circuit, parameter_id, tensor_index) != "BF16"
                for parameter_id in node["params"]
            )
            or projection_layout != ROW_MAJOR_LAYOUT
            or parameter_layout_for_id(circuit, node["params"][1], tensor_index)
            != ROW_MAJOR_LAYOUT
        ):
            raise ModelCompileError(
                f"projected recurrent convolution node {node['id']!r} has "
                f"incompatible projection {projection_shape}, state "
                f"{temporal_memory.get('shape')}, or kernel {kernel_shape}"
            )
        return (
            "linear_split_recurrent_depthwise_gate_bf16_"
            f"{projection_shape[1]}x{hidden_size}_k{frames}"
            f"_ig{input_gate_indices[0]}_{input_gate_indices[1]}"
            f"_og{output_gate_index}.comp"
        )
    if op == "residual_add":
        return f"add_bf16_{hidden_size}.comp"
    if op == "scaled_residual_add":
        return (
            f"scaled_add_bf16_{hidden_size}"
            f"_scale{shader_float_token(float(node['attrs']['scale']))}.comp"
        )
    if op == "silu":
        return f"silu_bf16_{int(node['attrs']['element_count'])}.comp"
    if op == "gelu_tanh":
        return f"gelu_tanh_bf16_{int(node['attrs']['element_count'])}.comp"
    if op == "silu_multiply":
        return f"silu_multiply_bf16_{int(node['attrs']['element_count'])}.comp"
    if op == "sigmoid_multiply":
        return "sigmoid_multiply_bf16.comp"
    if op == "softplus_multiply":
        attrs = node.get("attrs", {})
        query_heads = int(attrs.get("query_heads", 0))
        head_width = int(attrs.get("head_width", 0))
        if query_heads <= 0 or head_width <= 0 or head_width % 2:
            raise ModelCompileError(
                f"softplus gate node {node['id']!r} has invalid attention geometry"
            )
        mode = "per_head" if attrs.get("per_head") else "per_element"
        return f"softplus_multiply_bf16_q{query_heads}_d{head_width}_{mode}.comp"
    if op == "sigmoid_scalar_multiply":
        return f"sigmoid_scalar_multiply_bf16_{hidden_size}.comp"
    if op == "rms_norm_per_head":
        return (
            f"rms_norm_per_head_bf16_{node['attrs']['head_count']}x"
            f"{node['attrs']['head_width']}"
            f"_eps{shader_float_token(float(node['attrs']['eps']))}"
            f"_offset{shader_float_token(float(node['attrs']['weight_offset']))}.comp"
        )
    if op == "rms_norm_per_head_unscaled":
        return (
            f"rms_norm_per_head_unscaled_bf16_{node['attrs']['head_count']}"
            f"x{node['attrs']['head_width']}"
            f"_eps{shader_float_token(float(node['attrs']['eps']))}.comp"
        )
    if op == "parallel_head_norm_rope_2way":
        branches = node.get("attrs", {}).get("branches", [])
        if (
            len(branches) != 2
            or len(node.get("inputs", [])) != 2
            or len(node.get("outputs", [])) != 2
            or len(node.get("params", [])) != 2
        ):
            raise ModelCompileError(
                f"parallel head-norm/rope node {node['id']!r} has invalid branch metadata"
            )
        norms = [branch.get("norm", {}) for branch in branches]
        ropes = [branch.get("rope", {}) for branch in branches]
        head_counts = [int(norm["head_count"]) for norm in norms]
        common_fields = {
            "head_width": {int(norm["head_width"]) for norm in norms}
            | {int(rope["head_width"]) for rope in ropes},
            "eps": {float(norm["eps"]) for norm in norms},
            "weight_offset": {float(norm["weight_offset"]) for norm in norms},
            "rotary_width": {int(rope["rotary_width"]) for rope in ropes},
            "theta": {float(rope["theta"]) for rope in ropes},
            "rope_type": {str(rope.get("rope_type", "default")) for rope in ropes},
            "interleaved": {bool(rope["interleaved"]) for rope in ropes},
        }
        if any(len(values) != 1 for values in common_fields.values()) or any(
            int(norm["head_count"]) != int(rope["head_count"])
            for norm, rope in zip(norms, ropes, strict=True)
        ) or ropes[0].get("scaling") != ropes[1].get("scaling"):
            raise ModelCompileError(
                f"parallel head-norm/rope node {node['id']!r} mixes incompatible branch geometry"
            )
        parameter_dtypes = {
            parameter_dtype_for_id(circuit, parameter_id, tensor_index)
            for parameter_id in node["params"]
        }
        parameter_shapes = [
            parameter_shape_for_id(circuit, parameter_id, tensor_index)
            for parameter_id in node["params"]
        ]
        head_width = common_fields["head_width"].pop()
        if parameter_dtypes != {"BF16"} or any(
            list(map(int, shape)) != [head_width] for shape in parameter_shapes
        ):
            raise ModelCompileError(
                f"parallel head-norm/rope node {node['id']!r} has incompatible "
                f"normalization parameters {parameter_shapes}"
            )
        rope_attrs = {
            "theta": common_fields["theta"].pop(),
            "rope_type": common_fields["rope_type"].pop(),
            "interleaved": common_fields["interleaved"].pop(),
            "scaling": ropes[0].get("scaling"),
        }
        binding = stream_control_binding_for_node(circuit, node)
        return (
            f"parallel_head_norm_rope_2way_bf16_h{head_counts[0]}_{head_counts[1]}"
            f"_d{head_width}_r{common_fields['rotary_width'].pop()}"
            f"_eps{shader_float_token(common_fields['eps'].pop())}"
            f"_offset{shader_float_token(common_fields['weight_offset'].pop())}"
            f"_{rope_shader_suffix(rope_attrs)}"
            f"__sc{binding}.comp"
        )
    if op == "per_layer_embedding":
        attrs = node["attrs"]
        token_shape = parameter_shape_for_id(circuit, "token_embedding", tensor_index)
        vocab_size = int(token_shape[0])
        binding = stream_control_binding_for_node(circuit, node)
        return (
            f"per_layer_embedding_bf16_v{vocab_size}_h{attrs['hidden_size']}"
            f"_p{attrs['per_layer_width']}_l{attrs['layer_index']}of{attrs['layer_count']}"
            f"_c{attrs['embedding_chunk_count']}r{attrs['embedding_chunk_rows']}"
            f"_eps{shader_float_token(float(attrs['norm_eps']))}"
            f"_tes{shader_float_token(float(attrs['token_embedding_scale']))}"
            f"_pes{shader_float_token(float(attrs['per_layer_embedding_scale']))}"
            f"_mps{shader_float_token(float(attrs['model_projection_scale']))}"
            f"_cs{shader_float_token(float(attrs['combination_scale']))}__sc{binding}.comp"
        )
    if op == "rotary_position_embedding":
        binding = stream_control_binding_for_node(circuit, node)
        return (
            f"rotary_bf16_{node['attrs']['head_count']}x"
            f"{node['attrs']['head_width']}"
            f"_r{node['attrs']['rotary_width']}"
            f"_{rope_shader_suffix(node['attrs'])}"
            f"__sc{binding}.comp"
        )
    if op == "append_state_update":
        binding = stream_control_binding_for_node(circuit, node)
        return (
            f"append_kv_state_bf16_{node['attrs']['key_value_heads']}"
            f"x{node['attrs']['head_width']}__sc{binding}.comp"
        )
    if op == "scaled_dot_product_attention":
        attrs = node["attrs"]
        binding = stream_control_binding_for_node(circuit, node)
        name = (
            "gqa_attention_bf16_"
            f"q{attrs['query_heads']}_kv{attrs['key_value_heads']}_d{attrs['head_width']}"
            f"_scale{shader_float_token(float(attrs['scale']))}"
        )
        if attrs.get("window_size") is not None:
            name += f"_w{int(attrs['window_size'])}"
        if attrs.get("attention_sinks"):
            name += "_sinks"
        return f"{name}__sc{binding}.comp"
    if op == "append_scaled_dot_product_attention":
        attrs = node["attrs"]["attention"]
        binding = stream_control_binding_for_node(circuit, node)
        name = (
            "append_gqa_attention_bf16_"
            f"q{attrs['query_heads']}_kv{attrs['key_value_heads']}_d{attrs['head_width']}"
            f"_scale{shader_float_token(float(attrs['scale']))}"
        )
        if attrs.get("window_size") is not None:
            name += f"_w{int(attrs['window_size'])}"
        if attrs.get("attention_sinks"):
            name += "_sinks"
        return f"{name}__sc{binding}.comp"
    if op == "causal_conv1d_silu":
        return (
            f"causal_conv1d_silu_bf16_c{node['attrs']['channels']}"
            f"_k{node['attrs']['kernel_width']}.comp"
        )
    if op == "gated_delta_step":
        attrs = node["attrs"]
        dtype_tokens = {"F32": "f32", "BF16": "bf16"}
        parameter_tokens: dict[str, str] = {}
        for parameter_id in ("delta_a_log", "delta_dt_bias", "delta_norm"):
            actual_dtype = parameter_dtype_for_id(circuit, parameter_id, tensor_index)
            if actual_dtype not in dtype_tokens:
                raise ModelCompileError(
                    f"gated-delta parameter {parameter_id} has dtype {actual_dtype}; "
                    "expected F32 or BF16"
                )
            parameter_tokens[parameter_id] = dtype_tokens[actual_dtype]
        return (
            f"gated_delta_step_k{attrs['key_heads']}x{attrs['key_head_width']}"
            f"_v{attrs['value_heads']}x{attrs['value_head_width']}"
            f"_a{parameter_tokens['delta_a_log']}"
            f"_dt{parameter_tokens['delta_dt_bias']}"
            f"_n{parameter_tokens['delta_norm']}"
            f"_eps{shader_float_token(float(attrs['norm_eps']))}.comp"
        )
    if op == "rg_lru_step":
        attrs = node["attrs"]
        binding = stream_control_binding_for_node(circuit, node)
        return (
            f"rg_lru_step_bf16_h{attrs['width']}_b{attrs['heads']}x{attrs['block_width']}"
            f"_k{attrs['conv_kernel_width']}__sc{binding}.comp"
        )
    if op == "moe_topk":
        attrs = node["attrs"]
        activation = str(attrs.get("activation"))
        normalize_selected = bool(attrs.get("normalize_selected"))
        logit_softcap = float(attrs.get("logit_softcap"))
        has_bias = bool(attrs.get("selection_bias"))
        bias_dtype = None
        if activation not in {"sigmoid", "softmax"}:
            raise ModelCompileError(
                f"MoE router node {node['id']!r} has unsupported activation {activation!r}"
            )
        if not math.isfinite(logit_softcap) or logit_softcap < 0.0:
            raise ModelCompileError(
                f"MoE router node {node['id']!r} has invalid logit softcap {logit_softcap}"
            )
        if has_bias:
            if len(node.get("params", [])) != 1:
                raise ModelCompileError(
                    f"MoE router node {node['id']!r} is missing its selection bias"
                )
            bias_id = node["params"][0]
            if (
                parameter_shape_for_id(circuit, bias_id, tensor_index)
                != [int(attrs["num_experts"])]
                or parameter_dtype_for_id(circuit, bias_id, tensor_index)
                not in {"F32", "BF16"}
                or parameter_layout_for_id(circuit, bias_id, tensor_index)
                != ROW_MAJOR_LAYOUT
            ):
                raise ModelCompileError(
                    f"MoE router node {node['id']!r} has incompatible selection bias"
                )
            bias_dtype = parameter_dtype_for_id(circuit, bias_id, tensor_index)
        elif node.get("params"):
            raise ModelCompileError(
                f"MoE router node {node['id']!r} has an undeclared selection bias"
            )
        if (
            activation == "softmax"
            and normalize_selected
            and logit_softcap == 0.0
            and not has_bias
        ):
            return f"moe_topk_bf16_e{attrs['num_experts']}_k{attrs['experts_per_token']}.comp"
        return (
            f"moe_topk_{activation}_bf16_e{attrs['num_experts']}_"
            f"k{attrs['experts_per_token']}_norm{int(normalize_selected)}_"
            f"cap{shader_float_token(logit_softcap)}_"
            f"{'bias' + bias_dtype.lower() if bias_dtype else 'nobias'}.comp"
        )
    if op in {"sparse_moe_gate_up", "sparse_moe_down"}:
        attrs = node["attrs"]
        parameter_dtype = parameter_dtype_for_node(circuit, node, tensor_index)
        stage = "gate_up" if op == "sparse_moe_gate_up" else "down"
        if parameter_dtype == "F8_E4M3":
            block_rows, block_columns = fp8_moe_block_shape_for_stage(
                circuit, node, tensor_index, stage=stage
            )
            return (
                f"sparse_moe_{stage}_fp8_e4m3_b{block_rows}x{block_columns}_"
                f"h{attrs['hidden_size']}_i{attrs['intermediate_size']}_"
                f"e{attrs['num_experts']}_k{attrs['experts_per_token']}.comp"
            )
        if parameter_dtype == "I32":
            group_size, scale_dtype = compressed_tensors_int4_moe_shape_for_stage(
                circuit, node, tensor_index, stage=stage
            )
            return (
                f"sparse_moe_{stage}_int4_ct_s{scale_dtype.lower()}_g{group_size}_"
                f"h{attrs['hidden_size']}_i{attrs['intermediate_size']}_"
                f"e{attrs['num_experts']}_k{attrs['experts_per_token']}.comp"
            )
        if parameter_dtype != "BF16":
            raise ModelCompileError(
                f"sparse MoE {stage} node {node['id']!r} has unsupported expert dtype "
                f"{parameter_dtype}"
            )
        return (
            f"sparse_moe_{stage}_bf16_h{attrs['hidden_size']}_i{attrs['intermediate_size']}"
            f"_e{attrs['num_experts']}_k{attrs['experts_per_token']}.comp"
        )
    if op == "moe_reduce":
        attrs = node["attrs"]
        routed_scale = float(attrs["routed_scaling_factor"])
        if not math.isfinite(routed_scale) or routed_scale <= 0.0:
            raise ModelCompileError(
                f"MoE reduction node {node['id']!r} has invalid routed scale {routed_scale}"
            )
        return (
            f"moe_reduce_bf16_h{attrs['hidden_size']}"
            f"_k{attrs['experts_per_token']}"
            f"_scale{shader_float_token(routed_scale)}.comp"
        )

    raise ModelCompileError(
        f"no Vulkan shader selector for op {op!r} in node {node['id']!r}"
    )


def workgroup_count_x_for_node(circuit: Json, node: Json, tensor_index: Json) -> int:
    if node["op"] == "linear_split_recurrent_depthwise_gate":
        hidden_size = int(state_port(circuit, node["state_reads"][0])["shape"][1])
        return hidden_size // 2
    if node["op"] == "parallel_head_norm_rope_2way":
        return sum(
            int(branch["norm"]["head_count"]) for branch in node["attrs"]["branches"]
        )
    if node["op"] in {"parallel_linear_2way", "parallel_linear_3way"}:
        branch_count = int(node["attrs"]["branch_count"])
        branch_parameter_counts = [
            int(count)
            for count in node["attrs"].get(
                "branch_parameter_counts", [1] * branch_count
            )
        ]
        branch_weight_ids = []
        offset = 0
        for count in branch_parameter_counts:
            branch_weight_ids.append(node["params"][offset])
            offset += count
        if {
            parameter_dtype_for_id(circuit, parameter_id, tensor_index)
            for parameter_id in branch_weight_ids
        } == {"F8_E4M3"}:
            output_sizes = [
                int(parameter_shape_for_id(circuit, parameter_id, tensor_index)[0])
                for parameter_id in branch_weight_ids
            ]
            return max(
                (output_size + fp8_linear_tile_rows(output_size) - 1)
                // fp8_linear_tile_rows(output_size)
                for output_size in output_sizes
            )
        return sum(
            (int(parameter_shape_for_id(circuit, parameter_id, tensor_index)[0]) + 1)
            // 2
            for parameter_id in branch_weight_ids
        )
    if node["op"] == "parallel_linear_silu_multiply":
        out_features, _ = parameter_shape_for_id(
            circuit, node["params"][0], tensor_index
        )
        if (
            parameter_dtype_for_id(circuit, node["params"][0], tensor_index)
            == "F8_E4M3"
        ):
            return (
                int(out_features) + FP8_FUSED_FFN_TILE_ROWS - 1
            ) // FP8_FUSED_FFN_TILE_ROWS
        return (int(out_features) + 1) // 2
    if node["op"] in {"linear", "linear_residual", "linear_split_3way"}:
        out_features, _ = parameter_shape_for_node(circuit, node, tensor_index)
        parameter_dtype = parameter_dtype_for_node(circuit, node, tensor_index)
        if parameter_dtype == "I32":
            quantization_format = packed_linear_quantization_format_for_node(
                circuit, node, tensor_index
            )
            tile_rows = (
                INT4_GPTQ_OUTPUT_TILE_ROWS
                if quantization_format == "auto_gptq"
                else INT4_CT_OUTPUT_TILE_ROWS
                if quantization_format == "compressed_tensors_pack_quantized"
                else 0
            )
            if tile_rows == 0:
                raise ModelCompileError(
                    f"packed linear node {node['id']!r} has unsupported format "
                    f"{quantization_format!r}"
                )
            return (int(out_features) + tile_rows - 1) // tile_rows
        if parameter_dtype == "F8_E4M3":
            tile_rows = fp8_linear_tile_rows(int(out_features))
            return (int(out_features) + tile_rows - 1) // tile_rows
        if parameter_dtype == "Q8_0":
            return (
                int(out_features) + Q8_0_OUTPUT_TILE_ROWS - 1
            ) // Q8_0_OUTPUT_TILE_ROWS
        # One workgroup collaboratively computes and packs two BF16 output rows.
        return (int(out_features) + 1) // 2
    if node["op"] in {
        "scaled_dot_product_attention",
        "append_scaled_dot_product_attention",
    }:
        attrs = (
            node["attrs"]["attention"]
            if node["op"] == "append_scaled_dot_product_attention"
            else node["attrs"]
        )
        return int(attrs["query_heads"])
    if node["op"] == "gated_delta_step":
        return int(node["attrs"]["value_heads"])
    if node["op"] == "rg_lru_step":
        return int(node["attrs"]["heads"])
    if node["op"] == "sparse_moe_gate_up":
        attrs = node["attrs"]
        parameter_dtype = parameter_dtype_for_node(circuit, node, tensor_index)
        if parameter_dtype == "F8_E4M3":
            return int(attrs["experts_per_token"]) * (
                (int(attrs["intermediate_size"]) + FP8_SPARSE_GATE_UP_TILE_ROWS - 1)
                // FP8_SPARSE_GATE_UP_TILE_ROWS
            )
        if parameter_dtype == "I32":
            return int(attrs["experts_per_token"]) * (
                (int(attrs["intermediate_size"]) + INT4_CT_OUTPUT_TILE_ROWS - 1)
                // INT4_CT_OUTPUT_TILE_ROWS
            )
        return int(attrs["experts_per_token"]) * (
            (int(attrs["intermediate_size"]) + 1) // 2
        )
    if node["op"] == "sparse_moe_down":
        attrs = node["attrs"]
        parameter_dtype = parameter_dtype_for_node(circuit, node, tensor_index)
        if parameter_dtype == "F8_E4M3":
            return int(attrs["experts_per_token"]) * (
                (int(attrs["hidden_size"]) + FP8_SPARSE_DOWN_TILE_ROWS - 1)
                // FP8_SPARSE_DOWN_TILE_ROWS
            )
        if parameter_dtype == "I32":
            return int(attrs["experts_per_token"]) * (
                (int(attrs["hidden_size"]) + INT4_CT_OUTPUT_TILE_ROWS - 1)
                // INT4_CT_OUTPUT_TILE_ROWS
            )
        return int(attrs["experts_per_token"]) * ((int(attrs["hidden_size"]) + 1) // 2)
    if node["op"] in {
        "rms_norm_per_head",
        "rms_norm_per_head_unscaled",
        "rotary_position_embedding",
    }:
        return int(node["attrs"]["head_count"])
    return 1


def local_size_x_for_node(node: Json) -> int:
    # The tiled attention kernel maps sixteen 64-wide head reductions onto one
    # workgroup. This execution geometry belongs to the compiled model package.
    if node["op"] in {
        "scaled_dot_product_attention",
        "append_scaled_dot_product_attention",
    }:
        attrs = (
            node["attrs"]["attention"]
            if node["op"] == "append_scaled_dot_product_attention"
            else node["attrs"]
        )
        return attention_workgroup_shape(int(attrs["head_width"]))[0]
    if node["op"] == "gated_delta_step":
        return int(node["attrs"]["value_head_width"])
    if node["op"] == "rg_lru_step":
        return int(node["attrs"]["block_width"])
    return 64


def local_size_x_for_shader_file(shader_file: str, node: Json) -> int:
    if re.fullmatch(
        r"(linear|linear_residual)_fp8_e4m3_b\d+x\d+_\d+x\d+\.comp",
        shader_file,
    ) or re.fullmatch(
        r"parallel_linear_silu_multiply_fp8_e4m3_b\d+x\d+_\d+x\d+\.comp",
        shader_file,
    ):
        return 1024
    return local_size_x_for_node(node)


def fp8_linear_tile_rows(output_size: int) -> int:
    if output_size <= 0:
        raise ModelCompileError("FP8 linear output width must be positive")
    for tile_rows in reversed(FP8_LINEAR_TILE_ROWS):
        if (output_size + tile_rows - 1) // tile_rows >= FP8_LINEAR_MIN_WORKGROUPS:
            return tile_rows
    return FP8_LINEAR_TILE_ROWS[0]


def validate_native_int4_shader_shape(
    shader_file: str, group_size: int, input_size: int, output_size: int
) -> None:
    if (
        group_size <= 0
        or group_size % INT4_VALUES_PER_PACKED_WORD != 0
        or input_size <= 0
        or input_size % group_size != 0
        or output_size <= 0
        or output_size % 2 != 0
    ):
        raise ModelCompileError(f"invalid native INT4 shader shape {shader_file!r}")


def int4_shader_replacements(
    *,
    operation: str,
    quantization_format: str,
    scale_dtype: str,
    batch_tile_width: int | None,
) -> dict[str, str]:
    if operation not in {"linear", "linear_bias", "linear_residual"}:
        raise ModelCompileError(f"unsupported native INT4 operation {operation!r}")
    if quantization_format not in {"gptq", "ct"}:
        raise ModelCompileError(
            f"unsupported native INT4 quantization format {quantization_format!r}"
        )
    if scale_dtype == "f16":
        read_scale_body = (
            "    vec2 values = unpackHalf2x16(scales.words[index >> 1u]);\n"
            "    return (index & 1u) == 0u ? values.x : values.y;"
        )
    elif scale_dtype == "bf16":
        read_scale_body = "    return read_bf16_word(scales.words[index >> 1u], index);"
    else:
        raise ModelCompileError(f"unsupported native INT4 scale dtype {scale_dtype!r}")

    has_residual = operation == "linear_residual"
    has_bias = operation == "linear_bias"
    output_binding = 2 if has_residual else 1
    qweight_binding = output_binding + 1
    qzeros_binding = qweight_binding + 1 if quantization_format == "gptq" else None
    scales_binding = (
        (qzeros_binding + 1) if qzeros_binding is not None else qweight_binding + 1
    )
    auxiliary_binding = scales_binding + 1 if has_bias else None

    if has_residual:
        auxiliary_buffer = (
            "layout(set = 0, binding = 1) readonly buffer ResidualFrames { "
            "uint words[]; } residual_frames;"
        )
        finalize_output = (
            "float finalize_output(uint batch_index, uint row, float value) {\n"
            "    uint index = batch_index * OUTPUT_WORDS + (row >> 1u);\n"
            "    return read_bf16_word(residual_frames.words[index], row) + value;\n"
            "}"
        )
    elif has_bias:
        auxiliary_buffer = (
            f"layout(set = 0, binding = {auxiliary_binding}) readonly buffer Bias {{ "
            "uint words[]; } bias;"
        )
        finalize_output = (
            "float finalize_output(uint batch_index, uint row, float value) {\n"
            "    return read_bf16_word(bias.words[row >> 1u], row) + value;\n"
            "}"
        )
    else:
        auxiliary_buffer = ""
        finalize_output = (
            "float finalize_output(uint batch_index, uint row, float value) {\n"
            "    return value;\n"
            "}"
        )

    if batch_tile_width is None:
        batch_control = ""
        batch_tile_width = 1
        batch_start = "0u"
        batch_width = "1u"
    else:
        batch_control = (
            "layout(push_constant) uniform BatchControl { uint batch_width; } "
            "batch_control;"
        )
        batch_start = "gl_WorkGroupID.y * BATCH_TILE_WIDTH"
        batch_width = "batch_control.batch_width"

    replacements = {
        "OUTPUT_BINDING": str(output_binding),
        "QWEIGHT_BINDING": str(qweight_binding),
        "SCALES_BINDING": str(scales_binding),
        "AUXILIARY_BUFFER": auxiliary_buffer,
        "FINALIZE_OUTPUT_FUNCTION": finalize_output,
        "BATCH_CONTROL": batch_control,
        "BATCH_TILE_WIDTH": str(batch_tile_width),
        "BATCH_START": batch_start,
        "BATCH_WIDTH": batch_width,
        "READ_SCALE_BODY": read_scale_body,
    }
    if qzeros_binding is not None:
        replacements["QZEROS_BINDING"] = str(qzeros_binding)
    return replacements


def attention_workgroup_shape(head_width: int) -> tuple[int, int]:
    padded_head_width = ((head_width + 63) // 64) * 64
    physical_tile_tokens = 1024 // padded_head_width
    if physical_tile_tokens == 0:
        return 0, 0
    # Keep attention scratch below the 32 KiB Vulkan floor while amortizing the
    # four workgroup barriers over more KV tokens than the physical tile holds.
    shared_float_budget = (32 * 1024) // 4
    fixed_shared_floats = 2 * head_width + 4
    tile_shared_floats = head_width + ((head_width + 31) // 32) + 3
    max_token_batches = (shared_float_budget - fixed_shared_floats) // (
        physical_tile_tokens * tile_shared_floats
    )
    token_batches = max(1, min(7, max_token_batches))
    return (
        padded_head_width * physical_tile_tokens,
        physical_tile_tokens * token_batches,
    )


def rms_norm_shader_file(hidden_size: int, eps: float, weight_offset: float) -> str:
    return (
        f"rms_norm_bf16_h{hidden_size}_eps{shader_float_token(eps)}"
        f"_offset{shader_float_token(weight_offset)}.comp"
    )


def shader_float_token(value: float) -> str:
    return format(value, ".9g")


def rope_shader_suffix(attrs: Json) -> str:
    rope_type = str(attrs.get("rope_type", "default"))
    layout = (
        "proportional"
        if rope_type == "proportional"
        else "interleaved"
        if attrs.get("interleaved")
        else "half"
    )
    theta = float(attrs["theta"])
    scaling = attrs.get("scaling")
    if rope_type == "yarn":
        if not isinstance(scaling, dict) or scaling.get("type") != "yarn":
            raise ModelCompileError("YaRN RoPE node has no compiled scaling profile")
        return (
            f"theta{shader_float_token(theta)}_yarn"
            f"_f{shader_float_token(float(scaling['factor']))}"
            f"_lo{shader_float_token(float(scaling['correction_low']))}"
            f"_hi{shader_float_token(float(scaling['correction_high']))}"
            f"_a{shader_float_token(float(scaling['attention_factor']))}_{layout}"
        )
    if scaling is not None:
        raise ModelCompileError(
            f"RoPE type {rope_type!r} unexpectedly declares a scaling profile"
        )
    return f"theta{shader_float_token(theta)}_{layout}"
