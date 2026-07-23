from nerve.model_package_common import *


def attention_tile_token_width(head_width: int) -> int:
    padded_head_width = ((head_width + 63) // 64) * 64
    physical_tile_tokens = 1024 // padded_head_width
    if physical_tile_tokens == 0:
        return 0
    shared_float_budget = (32 * 1024) // 4
    fixed_shared_floats = 2 * head_width + 4
    tile_shared_floats = head_width + ((head_width + 31) // 32) + 3
    max_token_batches = (shared_float_budget - fixed_shared_floats) // (
        physical_tile_tokens * tile_shared_floats
    )
    token_batches = max(1, min(7, max_token_batches))
    return physical_tile_tokens * token_batches

def write_compiled_tensor(
    *,
    tensor_name: str,
    info: Json,
    source: Path,
    destination: Path,
    layout: str,
) -> tuple[int, str]:
    byte_count = int(info["byte_count"])
    header = {
        "__metadata__": {"format": "nerve", "layout": layout},
        tensor_name: {
            "dtype": info["dtype"],
            "shape": info["shape"],
            "data_offsets": [0, byte_count],
        },
    }
    header_payload = json.dumps(header, separators=(",", ":")).encode("utf-8")
    header_payload += b" " * (-len(header_payload) % 8)
    source_header_bytes = int(
        info.get("source_header_bytes") or read_safetensors_header(source)[0]
    )
    source_start = 8 + source_header_bytes + int(info["data_offsets"][0])
    data_digest = sha256()

    with (
        source.open("rb") as source_handle,
        destination.open("wb") as destination_handle,
    ):
        destination_handle.write(struct.pack("<Q", len(header_payload)))
        destination_handle.write(header_payload)
        source_handle.seek(source_start)
        copy_exact_bytes(
            source_handle, destination_handle, byte_count, digest=data_digest
        )
    return len(header_payload), data_digest.hexdigest()


def write_compiled_composite_tensor(
    *,
    tensor_name: str,
    info: Json,
    destination: Path,
    layout: str,
) -> tuple[int, str]:
    if layout != ROW_MAJOR_LAYOUT:
        raise ModelCompileError(
            f"composite tensor {tensor_name!r} requires unsupported layout {layout!r}"
        )
    byte_count = int(info["byte_count"])
    header = {
        "__metadata__": {"format": "nerve", "layout": layout},
        tensor_name: {
            "dtype": info["dtype"],
            "shape": info["shape"],
            "data_offsets": [0, byte_count],
        },
    }
    header_payload = json.dumps(header, separators=(",", ":")).encode("utf-8")
    header_payload += b" " * (-len(header_payload) % 8)
    written = 0
    data_digest = sha256()
    with destination.open("wb") as destination_handle:
        destination_handle.write(struct.pack("<Q", len(header_payload)))
        destination_handle.write(header_payload)
        for part in info["source_parts"]:
            source = Path(part["source_file"])
            if not source.is_file():
                raise ModelCompileError(
                    f"composite tensor {tensor_name!r} source does not exist: {source}"
                )
            source_header_bytes = int(part["source_header_bytes"])
            offsets = [int(value) for value in part["data_offsets"]]
            part_bytes = int(part["byte_count"])
            if offsets[1] - offsets[0] != part_bytes:
                raise ModelCompileError(
                    f"composite tensor {tensor_name!r} part {part['tensor']!r} "
                    "has inconsistent byte offsets"
                )
            with source.open("rb") as source_handle:
                source_handle.seek(8 + source_header_bytes + offsets[0])
                copy_exact_bytes(
                    source_handle,
                    destination_handle,
                    part_bytes,
                    digest=data_digest,
                )
            written += part_bytes
    if written != byte_count:
        raise ModelCompileError(
            f"composite tensor {tensor_name!r} wrote {written} bytes; expected {byte_count}"
        )
    return len(header_payload), data_digest.hexdigest()


def copy_exact_bytes(
    source_handle: Any,
    destination_handle: Any,
    byte_count: int,
    *,
    digest: Any | None = None,
) -> None:
    remaining = byte_count
    while remaining:
        chunk = source_handle.read(min(remaining, 8 * 1024 * 1024))
        if not chunk:
            raise ModelCompileError(
                "unexpected end of tensor source while compiling package"
            )
        destination_handle.write(chunk)
        if digest is not None:
            digest.update(chunk)
        remaining -= len(chunk)


def parameter_shape_for_node(
    circuit: Json, node: Json, tensor_index: Json
) -> list[int]:
    parameter_id = node["params"][0]
    parameter = circuit["parameters"]["refs"][parameter_id]
    return tensor_shape(tensor_index, parameter["tensor"])


def can_fuse_bf16_linear_split(circuit: Json, node: Json, tensor_index: Json) -> bool:
    shape = parameter_shape_for_node(circuit, node, tensor_index)
    return (
        len(shape) == 2
        and all(int(dimension) > 0 and int(dimension) % 2 == 0 for dimension in shape)
        and parameter_dtype_for_node(circuit, node, tensor_index) == "BF16"
        and parameter_layout_for_node(circuit, node, tensor_index) == ROW_MAJOR_LAYOUT
    )


def can_fuse_native_parallel_linears(
    circuit: Json, nodes: list[Json], tensor_index: Json
) -> bool:
    shapes = [parameter_shape_for_node(circuit, node, tensor_index) for node in nodes]
    try:
        parameter_groups = [node["params"] for node in nodes]
        dtypes = {
            parameter_dtype_for_node(circuit, node, tensor_index) for node in nodes
        }
        layouts = {
            parameter_layout_for_node(circuit, node, tensor_index) for node in nodes
        }
    except (KeyError, ModelCompileError):
        return False
    valid_geometry = (
        len(nodes) in {2, 3}
        and len(shapes) == len(nodes)
        and all(
            len(shape) == 2
            and all(
                int(dimension) > 0 and int(dimension) % 2 == 0 for dimension in shape
            )
            for shape in shapes
        )
        and len({int(shape[1]) for shape in shapes}) == 1
        and layouts == {ROW_MAJOR_LAYOUT}
    )
    if not valid_geometry:
        return False
    if dtypes == {"BF16"}:
        return all(len(parameters) == 1 for parameters in parameter_groups)
    if dtypes != {"F8_E4M3"}:
        return False
    try:
        block_shapes = {
            fp8_block_shape_for_node(circuit, node, tensor_index) for node in nodes
        }
    except ModelCompileError:
        return False
    return (
        all(len(parameters) == 2 for parameters in parameter_groups)
        and len(block_shapes) == 1
        and all(shapes[0] == shape for shape in shapes)
    )


def can_fuse_parallel_linear_silu_multiply(
    circuit: Json,
    projection: Json,
    activation: Json,
    tensor_index: Json,
) -> bool:
    if (
        projection.get("op") != "parallel_linear_2way"
        or activation.get("op") != "silu_multiply"
    ):
        return False
    try:
        params = projection["params"]
        element_count = int(activation.get("attrs", {})["element_count"])
    except (KeyError, TypeError, ValueError, ModelCompileError):
        return False

    if len(params) == 2:
        weight_ids = params
        try:
            layouts = {
                parameter_layout_for_id(circuit, parameter_id, tensor_index)
                for parameter_id in weight_ids
            }
            supported_parameters = {
                parameter_dtype_for_id(circuit, parameter_id, tensor_index)
                for parameter_id in weight_ids
            } == {"BF16"} and layouts == {ROW_MAJOR_LAYOUT}
        except ModelCompileError:
            return False
    elif len(params) == 4:
        weight_ids = [params[0], params[2]]
        branch_params = [params[:2], params[2:]]
        try:
            supported_parameters = (
                all(
                    parameter_dtype_for_id(circuit, weight_id, tensor_index)
                    == "F8_E4M3"
                    and parameter_layout_for_id(circuit, weight_id, tensor_index)
                    == ROW_MAJOR_LAYOUT
                    for weight_id in weight_ids
                )
                and len(
                    {
                        fp8_block_shape_for_node(
                            circuit,
                            {
                                "id": f"{projection['id']}__branch_{index}",
                                "params": parameter_ids,
                            },
                            tensor_index,
                        )
                        for index, parameter_ids in enumerate(branch_params)
                    }
                )
                == 1
            )
        except ModelCompileError:
            return False
    else:
        return False

    try:
        shapes = [
            parameter_shape_for_id(circuit, parameter_id, tensor_index)
            for parameter_id in weight_ids
        ]
    except ModelCompileError:
        return False
    return (
        supported_parameters
        and len(shapes) == 2
        and shapes[0] == shapes[1]
        and len(shapes[0]) == 2
        and all(
            int(dimension) > 0 and int(dimension) % 2 == 0 for dimension in shapes[0]
        )
        and element_count == int(shapes[0][0])
        and activation.get("attrs", {}).get("intermediate_rounding") == "BF16"
    )


def can_fuse_bf16_parallel_head_norm_rope(
    circuit: Json,
    branches: list[tuple[Json, Json]],
    tensor_index: Json,
) -> bool:
    if len(branches) != 2:
        return False
    norms = [norm for norm, _rope in branches]
    ropes = [rope for _norm, rope in branches]
    try:
        head_counts = [int(norm["attrs"]["head_count"]) for norm in norms]
        head_widths = {
            int(node["attrs"]["head_width"]) for branch in branches for node in branch
        }
        rotary_widths = {int(rope["attrs"]["rotary_width"]) for rope in ropes}
        common_values = (
            {float(norm["attrs"]["eps"]) for norm in norms},
            {float(norm["attrs"]["weight_offset"]) for norm in norms},
            {float(rope["attrs"]["theta"]) for rope in ropes},
            {str(rope["attrs"].get("rope_type", "default")) for rope in ropes},
            {
                json.dumps(
                    rope["attrs"].get("scaling"),
                    sort_keys=True,
                    separators=(",", ":"),
                )
                for rope in ropes
            },
            {bool(rope["attrs"]["interleaved"]) for rope in ropes},
            {str(rope["attrs"]["position_source"]) for rope in ropes},
        )
        parameter_shapes = [
            parameter_shape_for_node(circuit, norm, tensor_index) for norm in norms
        ]
    except (KeyError, TypeError, ValueError):
        return False
    if len(head_widths) != 1 or len(rotary_widths) != 1:
        return False
    head_width = next(iter(head_widths))
    rotary_width = next(iter(rotary_widths))
    return (
        all(count > 0 for count in head_counts)
        and head_width > 0
        and head_width % 2 == 0
        and rotary_width > 0
        and rotary_width % 2 == 0
        and rotary_width <= head_width
        and all(len(values) == 1 for values in common_values)
        and common_values[-1] == {"stream_tick"}
        and all(
            int(norm["attrs"]["head_count"]) == int(rope["attrs"]["head_count"])
            for norm, rope in branches
        )
        and all(shape == [head_width] for shape in parameter_shapes)
        and all(
            parameter_dtype_for_node(circuit, norm, tensor_index) == "BF16"
            for norm in norms
        )
        and all(
            parameter_layout_for_node(circuit, norm, tensor_index) == ROW_MAJOR_LAYOUT
            for norm in norms
        )
    )


def can_fuse_bf16_multiply_rolling_depthwise(
    circuit: Json,
    multiply: Json,
    rolling: Json,
    depthwise: Json,
    tensor_index: Json,
) -> bool:
    if (
        multiply.get("op") != "multiply"
        or len(multiply.get("inputs", [])) != 2
        or rolling.get("attrs", {}).get("update") != "shift_append"
        or len(rolling.get("state_reads", [])) != 1
        or rolling.get("state_reads") != rolling.get("state_writes")
        or len(depthwise.get("params", [])) != 1
    ):
        return False
    temporal_memory = state_port(circuit, rolling["state_reads"][0])
    state_shape = list(map(int, temporal_memory.get("shape", [])))
    if len(state_shape) != 2:
        return False
    frames, hidden_size = state_shape
    kernel_shape = parameter_shape_for_node(circuit, depthwise, tensor_index)
    return (
        temporal_memory.get("dtype") == "BF16"
        and temporal_memory.get("update") == "shift_append"
        and frames >= 2
        and hidden_size > 0
        and hidden_size % 2 == 0
        and depthwise.get("attrs", {}).get("groups") == hidden_size
        and kernel_shape in ([hidden_size, frames], [hidden_size, 1, frames])
        and parameter_dtype_for_node(circuit, depthwise, tensor_index) == "BF16"
        and parameter_layout_for_node(circuit, depthwise, tensor_index)
        == ROW_MAJOR_LAYOUT
    )


def can_fuse_bf16_recurrent_output_gate(
    circuit: Json,
    recurrent: Json,
    gate: Json,
    tensor_index: Json,
) -> bool:
    if (
        recurrent.get("op") != "multiply_rolling_depthwise"
        or gate.get("op") != "multiply"
        or len(recurrent.get("state_reads", [])) != 1
        or recurrent.get("state_reads") != recurrent.get("state_writes")
        or len(recurrent.get("params", [])) != 1
    ):
        return False
    temporal_memory = state_port(circuit, recurrent["state_reads"][0])
    state_shape = list(map(int, temporal_memory.get("shape", [])))
    if len(state_shape) != 2:
        return False
    frames, hidden_size = state_shape
    kernel_shape = parameter_shape_for_node(circuit, recurrent, tensor_index)
    return (
        temporal_memory.get("dtype") == "BF16"
        and temporal_memory.get("update") == "shift_append"
        and frames >= 2
        and hidden_size > 0
        and hidden_size % 2 == 0
        and kernel_shape in ([hidden_size, frames], [hidden_size, 1, frames])
        and parameter_dtype_for_node(circuit, recurrent, tensor_index) == "BF16"
        and parameter_layout_for_node(circuit, recurrent, tensor_index)
        == ROW_MAJOR_LAYOUT
    )


def can_fuse_bf16_linear_split_recurrent(
    circuit: Json,
    projection: Json,
    recurrent: Json,
    tensor_index: Json,
) -> bool:
    if (
        len(projection.get("params", [])) != 1
        or len(recurrent.get("params", [])) != 1
        or len(recurrent.get("state_reads", [])) != 1
        or recurrent.get("state_reads") != recurrent.get("state_writes")
    ):
        return False
    temporal_memory = state_port(circuit, recurrent["state_reads"][0])
    state_shape = list(map(int, temporal_memory.get("shape", [])))
    if len(state_shape) != 2:
        return False
    frames, hidden_size = state_shape
    projection_shape = parameter_shape_for_node(circuit, projection, tensor_index)
    kernel_shape = parameter_shape_for_node(circuit, recurrent, tensor_index)
    part_widths = [
        int(width) for width in projection.get("attrs", {}).get("part_widths", [])
    ]
    return (
        temporal_memory.get("dtype") == "BF16"
        and temporal_memory.get("update") == "shift_append"
        and frames >= 2
        and hidden_size > 0
        and hidden_size % 2 == 0
        and len(projection_shape) == 2
        and projection_shape[0] == 3 * hidden_size
        and projection_shape[1] > 0
        and projection_shape[1] % 2 == 0
        and part_widths == [hidden_size] * 3
        and kernel_shape in ([hidden_size, frames], [hidden_size, 1, frames])
        and parameter_dtype_for_node(circuit, projection, tensor_index) == "BF16"
        and parameter_layout_for_node(circuit, projection, tensor_index)
        == ROW_MAJOR_LAYOUT
        and parameter_dtype_for_node(circuit, recurrent, tensor_index) == "BF16"
        and parameter_layout_for_node(circuit, recurrent, tensor_index)
        == ROW_MAJOR_LAYOUT
    )


def can_fuse_bf16_append_attention(
    circuit: Json,
    append: Json,
    attention: Json,
    tensor_index: Json,
) -> bool:
    if (
        append.get("op") != "append_state_update"
        or attention.get("op") != "scaled_dot_product_attention"
        or len(append.get("inputs", [])) != 3
        or len(append.get("state_reads", [])) != 1
        or append.get("state_reads") != append.get("state_writes")
        or append["inputs"][2] != append["state_reads"][0]
        or attention.get("attrs", {}).get("causal") is not True
    ):
        return False
    append_attrs = append.get("attrs", {})
    attention_attrs = attention.get("attrs", {})
    geometry_keys = (
        "query_heads",
        "key_value_heads",
        "head_width",
        "query_groups_per_kv_head",
    )
    try:
        append_geometry = tuple(int(append_attrs[key]) for key in geometry_keys)
        attention_geometry = tuple(int(attention_attrs[key]) for key in geometry_keys)
        query_heads, kv_heads, head_width, query_groups = attention_geometry
        memory = state_port(circuit, append["state_reads"][0])
        key_shape = list(map(int, memory.get("key_shape_per_token", [])))
        value_shape = list(map(int, memory.get("value_shape_per_token", [])))
        window_size = attention_attrs.get("window_size")
        if window_size is not None:
            window_size = int(window_size)
        scale = float(attention_attrs["scale"])
    except (KeyError, TypeError, ValueError, ModelCompileError):
        return False
    params = attention.get("params", [])
    has_sinks = bool(attention_attrs.get("attention_sinks"))
    if len(params) != int(has_sinks):
        return False
    if params:
        try:
            sink_shape = parameter_shape_for_id(circuit, params[0], tensor_index)
            sink_dtype = parameter_dtype_for_id(circuit, params[0], tensor_index)
            sink_layout = parameter_layout_for_id(circuit, params[0], tensor_index)
        except (KeyError, ModelCompileError):
            return False
        if (
            sink_shape != [query_heads]
            or sink_dtype != "BF16"
            or sink_layout != ROW_MAJOR_LAYOUT
        ):
            return False
    return (
        append_geometry == attention_geometry
        and append_attrs.get("growth") == "per_activation"
        and query_heads > 0
        and kv_heads > 0
        and query_heads % kv_heads == 0
        and query_groups == query_heads // kv_heads
        and head_width >= 2
        and head_width % 2 == 0
        and attention_tile_token_width(head_width) > 0
        and scale > 0.0
        and (window_size is None or window_size > 0)
        and memory.get("type") == "append_only_attention_memory"
        and memory.get("dtype") == "BF16"
        and memory.get("growth") == "per_activation"
        and memory.get("layout") == "append_only_kv"
        and memory.get("source_layout") == "batch_kvheads_seq_headdim"
        and key_shape == [kv_heads, head_width]
        and value_shape == [kv_heads, head_width]
    )


def parameter_shape_for_id(
    circuit: Json, parameter_id: str, tensor_index: Json
) -> list[int]:
    parameter = circuit["parameters"]["refs"][parameter_id]
    return tensor_shape(tensor_index, parameter["tensor"])


def parameter_dtype_for_node(circuit: Json, node: Json, tensor_index: Json) -> str:
    return parameter_dtype_for_id(circuit, node["params"][0], tensor_index)


def parameter_layout_for_node(circuit: Json, node: Json, tensor_index: Json) -> str:
    parameter_id = node["params"][0]
    parameter = circuit["parameters"]["refs"][parameter_id]
    return tensor_layout(tensor_index, parameter["tensor"])


def parameter_layout_for_id(
    circuit: Json, parameter_id: str, tensor_index: Json
) -> str:
    parameter = circuit["parameters"]["refs"][parameter_id]
    return tensor_layout(tensor_index, parameter["tensor"])


def parameter_dtype_for_id(circuit: Json, parameter_id: str, tensor_index: Json) -> str:
    parameter = circuit["parameters"]["refs"][parameter_id]
    return str(tensor_index["tensors"][parameter["tensor"]]["dtype"])


def scalar_parameter_read_expression(buffer: str, dtype_token: str) -> str:
    if dtype_token == "f32":
        return f"uintBitsToFloat({buffer}.words[index])"
    if dtype_token == "bf16":
        return f"unpack_bf16({buffer}.words[index >> 1u], index)"
    raise ModelCompileError(
        f"unsupported scalar parameter shader dtype token {dtype_token!r}"
    )


def packed_int4_linear_group_size_for_node(
    circuit: Json, node: Json, tensor_index: Json
) -> int:
    weight_id = str(node["params"][0])
    qzeros_id = f"{weight_id}_qzeros"
    scales_id = f"{weight_id}_scales"
    expected_params = [weight_id, qzeros_id, scales_id]
    actual_params = list(node.get("params", []))
    if len(actual_params) == 4:
        expected_params.append(f"{weight_id}_bias")
    if actual_params != expected_params:
        raise ModelCompileError(
            f"packed INT4 linear node {node['id']!r} must bind {expected_params}; "
            f"got {actual_params}"
        )
    weight_ref = circuit["parameters"]["refs"][weight_id]
    weight_info = tensor_index["tensors"][weight_ref["tensor"]]
    quantization = weight_info.get("quantization")
    if not isinstance(quantization, dict):
        raise ModelCompileError(
            f"packed INT4 linear node {node['id']!r} has no compiled quantization metadata"
        )
    if (
        quantization.get("format") != "auto_gptq"
        or int(quantization.get("bits") or 0) != 4
        or int(quantization.get("zero_point_add") or 0) != 1
    ):
        raise ModelCompileError(
            f"packed INT4 linear node {node['id']!r} has unsupported quantization "
            f"{quantization}"
        )
    out_features, in_features = parameter_shape_for_id(circuit, weight_id, tensor_index)
    packed_shape = [int(value) for value in weight_info.get("shape", [])]
    qzeros_shape = parameter_shape_for_id(circuit, qzeros_id, tensor_index)
    scales_shape = parameter_shape_for_id(circuit, scales_id, tensor_index)
    group_size = int(quantization.get("group_size") or 0)
    group_count = (in_features + group_size - 1) // group_size if group_size else 0
    if packed_shape != [in_features // 8, out_features]:
        raise ModelCompileError(
            f"packed INT4 weight shape {packed_shape} does not encode "
            f"{[out_features, in_features]}"
        )
    if qzeros_shape != [group_count, (out_features + 7) // 8]:
        raise ModelCompileError(
            f"packed INT4 zero-point shape {qzeros_shape} is incompatible with "
            f"{[out_features, in_features]}"
        )
    if scales_shape != [group_count, out_features]:
        raise ModelCompileError(
            f"packed INT4 scale shape {scales_shape} is incompatible with "
            f"{[out_features, in_features]}"
        )
    if parameter_dtype_for_id(circuit, qzeros_id, tensor_index) != "I32":
        raise ModelCompileError("packed INT4 zero points must use I32 storage")
    if parameter_dtype_for_id(circuit, scales_id, tensor_index) != "F16":
        raise ModelCompileError("packed INT4 scales must use F16 storage")
    if any(
        parameter_layout_for_id(circuit, parameter_id, tensor_index) != ROW_MAJOR_LAYOUT
        for parameter_id in (weight_id, qzeros_id, scales_id)
    ):
        raise ModelCompileError("packed INT4 parameters must use row-major storage")
    if (
        len(actual_params) == 4
        and parameter_dtype_for_id(circuit, actual_params[3], tensor_index) != "BF16"
    ):
        raise ModelCompileError("packed INT4 linear bias must use BF16 storage")
    return group_size


def packed_linear_quantization_format_for_node(
    circuit: Json, node: Json, tensor_index: Json
) -> str:
    weight_id = str(node["params"][0])
    weight_ref = circuit["parameters"]["refs"][weight_id]
    quantization = tensor_index["tensors"][weight_ref["tensor"]].get("quantization")
    if not isinstance(quantization, dict) or not quantization.get("format"):
        raise ModelCompileError(
            f"packed linear node {node['id']!r} has no quantization format"
        )
    return str(quantization["format"])


def packed_int4_scale_dtype_for_node(
    circuit: Json, node: Json, tensor_index: Json
) -> str:
    scale_parameter_id = f"{node['params'][0]}_scales"
    if scale_parameter_id not in node.get("params", []):
        raise ModelCompileError(
            f"packed INT4 linear node {node['id']!r} has no scale parameter"
        )
    scale_dtype = parameter_dtype_for_id(circuit, scale_parameter_id, tensor_index)
    if scale_dtype not in {"F16", "BF16"}:
        raise ModelCompileError(
            f"packed INT4 linear node {node['id']!r} has unsupported scale dtype "
            f"{scale_dtype!r}"
        )
    return scale_dtype


def compressed_tensors_int4_group_size_for_node(
    circuit: Json, node: Json, tensor_index: Json
) -> int:
    weight_id = str(node["params"][0])
    scales_id = f"{weight_id}_scales"
    expected_params = [weight_id, scales_id]
    actual_params = list(node.get("params", []))
    if node["op"] == "linear" and len(actual_params) == 3:
        expected_params.append(f"{weight_id}_bias")
    if actual_params != expected_params:
        raise ModelCompileError(
            f"compressed-tensors INT4 node {node['id']!r} must bind "
            f"{expected_params}; got {actual_params}"
        )
    weight_ref = circuit["parameters"]["refs"][weight_id]
    weight_info = tensor_index["tensors"][weight_ref["tensor"]]
    quantization = weight_info.get("quantization")
    if (
        not isinstance(quantization, dict)
        or quantization.get("format") != "compressed_tensors_pack_quantized"
        or int(quantization.get("bits") or 0) != 4
        or int(quantization.get("signed_offset") or 0) != 8
        or not bool(quantization.get("symmetric"))
    ):
        raise ModelCompileError(
            f"compressed-tensors INT4 node {node['id']!r} has unsupported "
            f"quantization {quantization}"
        )
    out_features, in_features = parameter_shape_for_id(circuit, weight_id, tensor_index)
    packed_shape = [int(value) for value in weight_info.get("shape", [])]
    scales_shape = parameter_shape_for_id(circuit, scales_id, tensor_index)
    group_size = int(quantization.get("group_size") or 0)
    if packed_shape != [out_features, (in_features + 7) // 8]:
        raise ModelCompileError(
            f"compressed-tensors INT4 weight shape {packed_shape} does not encode "
            f"{[out_features, in_features]}"
        )
    if group_size <= 0 or scales_shape != [
        out_features,
        (in_features + group_size - 1) // group_size,
    ]:
        raise ModelCompileError(
            f"compressed-tensors INT4 scale shape {scales_shape} is incompatible "
            f"with {[out_features, in_features]}"
        )
    if parameter_dtype_for_id(circuit, scales_id, tensor_index) != "BF16":
        raise ModelCompileError("compressed-tensors INT4 scales must use BF16 storage")
    if any(
        parameter_layout_for_id(circuit, parameter_id, tensor_index) != ROW_MAJOR_LAYOUT
        for parameter_id in (weight_id, scales_id)
    ):
        raise ModelCompileError(
            "compressed-tensors INT4 parameters must use row-major storage"
        )
    if (
        len(actual_params) == 3
        and parameter_dtype_for_id(circuit, actual_params[2], tensor_index) != "BF16"
    ):
        raise ModelCompileError(
            "compressed-tensors INT4 linear bias must use BF16 storage"
        )
    return group_size


def fp8_block_shape_for_node(
    circuit: Json, node: Json, tensor_index: Json
) -> tuple[int, int]:
    weight_id = str(node["params"][0])
    scale_id = f"{weight_id}_scale_inv"
    if len(node.get("params", [])) < 2 or node["params"][1] != scale_id:
        raise ModelCompileError(
            f"FP8 linear node {node['id']!r} does not bind {scale_id!r} "
            "immediately after its weight"
        )
    out_features, in_features = parameter_shape_for_id(circuit, weight_id, tensor_index)
    scale_shape = parameter_shape_for_id(circuit, scale_id, tensor_index)
    if len(scale_shape) != 2 or any(value <= 0 for value in scale_shape):
        raise ModelCompileError(
            f"FP8 linear node {node['id']!r} has invalid scale shape {scale_shape}"
        )
    if parameter_dtype_for_id(circuit, scale_id, tensor_index) != "BF16":
        raise ModelCompileError(
            f"FP8 linear node {node['id']!r} requires a BF16 block scale"
        )
    if parameter_layout_for_id(circuit, scale_id, tensor_index) != ROW_MAJOR_LAYOUT:
        raise ModelCompileError(
            f"FP8 linear node {node['id']!r} requires row-major block scales"
        )
    block_rows = (out_features + scale_shape[0] - 1) // scale_shape[0]
    block_columns = (in_features + scale_shape[1] - 1) // scale_shape[1]
    expected_scale_shape = [
        (out_features + block_rows - 1) // block_rows,
        (in_features + block_columns - 1) // block_columns,
    ]
    if scale_shape != expected_scale_shape:
        raise ModelCompileError(
            f"FP8 linear node {node['id']!r} scale shape {scale_shape} is not "
            f"a regular block grid for weight shape {[out_features, in_features]}"
        )
    if (
        block_columns != 128
        or in_features % block_columns != 0
        or in_features % 4 != 0
        or out_features % 2 != 0
    ):
        raise ModelCompileError(
            f"native FP8 linear node {node['id']!r} requires 128-column blocks, "
            f"a four-aligned input width, and an even output width; got "
            f"block {[block_rows, block_columns]} for "
            f"{[out_features, in_features]}"
        )
    return block_rows, block_columns


def fp8_moe_block_shape_for_stage(
    circuit: Json,
    node: Json,
    tensor_index: Json,
    *,
    stage: str,
) -> tuple[int, int]:
    if stage == "gate_up":
        weight_id = "moe_input"
        scale_id = "moe_input_scale_inv"
    elif stage == "down":
        weight_id = "moe_output"
        scale_id = "moe_output_scale_inv"
    else:
        raise ModelCompileError(f"unknown sparse MoE stage {stage!r}")
    expected_params = [weight_id, scale_id]
    if node.get("params") != expected_params:
        raise ModelCompileError(
            f"FP8 sparse MoE {stage} node {node['id']!r} must bind {expected_params}; "
            f"got {node.get('params')}"
        )
    attrs = node["attrs"]
    experts = int(attrs["num_experts"])
    hidden = int(attrs["hidden_size"])
    intermediate = int(attrs["intermediate_size"])
    expected_weight_shape = (
        [experts, intermediate * 2, hidden]
        if stage == "gate_up"
        else [experts, hidden, intermediate]
    )
    weight_shape = parameter_shape_for_id(circuit, weight_id, tensor_index)
    scale_shape = parameter_shape_for_id(circuit, scale_id, tensor_index)
    if weight_shape != expected_weight_shape:
        raise ModelCompileError(
            f"FP8 sparse MoE {stage} weight shape {weight_shape} does not match "
            f"{expected_weight_shape}"
        )
    if parameter_layout_for_id(circuit, weight_id, tensor_index) != ROW_MAJOR_LAYOUT:
        raise ModelCompileError(
            f"FP8 sparse MoE {stage} weight {weight_id!r} must be row-major"
        )
    if parameter_dtype_for_id(circuit, scale_id, tensor_index) != "BF16":
        raise ModelCompileError(f"FP8 sparse MoE scale {scale_id!r} must be BF16")
    if parameter_layout_for_id(circuit, scale_id, tensor_index) != ROW_MAJOR_LAYOUT:
        raise ModelCompileError(f"FP8 sparse MoE scale {scale_id!r} must be row-major")
    if len(scale_shape) != 3 or scale_shape[0] != experts:
        raise ModelCompileError(
            f"FP8 sparse MoE {stage} scale shape is invalid: {scale_shape}"
        )
    block_rows, block_columns = regular_block_shape(
        expected_weight_shape[1:], scale_shape[1:]
    )
    input_width = hidden if stage == "gate_up" else intermediate
    output_width = intermediate if stage == "gate_up" else hidden
    if (
        block_columns != 128
        or input_width % block_columns != 0
        or input_width % 4 != 0
        or output_width % 2 != 0
    ):
        raise ModelCompileError(
            f"native FP8 sparse MoE {stage} requires 128-column blocks, "
            f"a four-aligned input width, and an even output width; got "
            f"block {[block_rows, block_columns]} with input {input_width} and "
            f"output {output_width}"
        )
    return block_rows, block_columns


def compressed_tensors_int4_moe_shape_for_stage(
    circuit: Json,
    node: Json,
    tensor_index: Json,
    *,
    stage: str,
) -> tuple[int, str]:
    if stage == "gate_up":
        weight_id = "moe_input"
        scale_id = "moe_input_scales"
    elif stage == "down":
        weight_id = "moe_output"
        scale_id = "moe_output_scales"
    else:
        raise ModelCompileError(f"unknown sparse MoE stage {stage!r}")
    expected_params = [weight_id, scale_id]
    if node.get("params") != expected_params:
        raise ModelCompileError(
            f"INT4 sparse MoE {stage} node {node['id']!r} must bind "
            f"{expected_params}; got {node.get('params')}"
        )

    attrs = node["attrs"]
    experts = int(attrs["num_experts"])
    hidden = int(attrs["hidden_size"])
    intermediate = int(attrs["intermediate_size"])
    expected_logical_shape = (
        [experts, intermediate * 2, hidden]
        if stage == "gate_up"
        else [experts, hidden, intermediate]
    )
    weight_ref = circuit["parameters"]["refs"][weight_id]
    weight_info = tensor_index["tensors"][weight_ref["tensor"]]
    quantization = weight_info.get("quantization")
    if (
        parameter_dtype_for_id(circuit, weight_id, tensor_index) != "I32"
        or parameter_layout_for_id(circuit, weight_id, tensor_index) != ROW_MAJOR_LAYOUT
        or not isinstance(quantization, dict)
        or quantization.get("format") != "compressed_tensors_pack_quantized"
        or int(quantization.get("bits", 0)) != 4
        or not bool(quantization.get("symmetric"))
        or int(quantization.get("signed_offset", -1)) != 8
    ):
        raise ModelCompileError(
            f"INT4 sparse MoE {stage} weight has an incompatible quantization contract"
        )
    group_size = int(quantization.get("group_size", 0))
    input_width = expected_logical_shape[2]
    output_rows = expected_logical_shape[1]
    if (
        parameter_shape_for_id(circuit, weight_id, tensor_index)
        != expected_logical_shape
        or [int(value) for value in weight_info.get("shape", [])]
        != [experts, output_rows, input_width // 8]
        or group_size <= 0
        or group_size % INT4_VALUES_PER_PACKED_WORD != 0
        or input_width % group_size != 0
        or output_rows % 2 != 0
    ):
        raise ModelCompileError(
            f"INT4 sparse MoE {stage} has incompatible weight shape or group geometry"
        )

    scale_shape = parameter_shape_for_id(circuit, scale_id, tensor_index)
    scale_dtype = parameter_dtype_for_id(circuit, scale_id, tensor_index)
    if (
        scale_shape != [experts, output_rows, input_width // group_size]
        or scale_dtype not in {"F16", "BF16"}
        or parameter_layout_for_id(circuit, scale_id, tensor_index) != ROW_MAJOR_LAYOUT
    ):
        raise ModelCompileError(
            f"INT4 sparse MoE {stage} scale shape or dtype is incompatible"
        )
    return group_size, scale_dtype


def regular_block_shape(
    matrix_shape: list[int], scale_shape: list[int]
) -> tuple[int, int]:
    if (
        len(matrix_shape) != 2
        or len(scale_shape) != 2
        or any(value <= 0 for value in scale_shape)
    ):
        raise ModelCompileError(
            f"invalid block-scaled matrix shape {matrix_shape} / {scale_shape}"
        )
    rows, columns = matrix_shape
    block_rows = (rows + scale_shape[0] - 1) // scale_shape[0]
    block_columns = (columns + scale_shape[1] - 1) // scale_shape[1]
    expected_scale_shape = [
        (rows + block_rows - 1) // block_rows,
        (columns + block_columns - 1) // block_columns,
    ]
    if scale_shape != expected_scale_shape:
        raise ModelCompileError(
            f"scale shape {scale_shape} is not a regular block grid for {matrix_shape}"
        )
    return block_rows, block_columns


def state_port(circuit: Json, state_id: str) -> Json:
    for port in circuit.get("state_ports", []):
        if port["id"] == state_id:
            return port
    raise ModelCompileError(f"circuit {circuit['id']} has no state port {state_id!r}")


def tensor_shape(tensor_index: Json, tensor: str) -> list[int]:
    info = tensor_index["tensors"][tensor]
    return [int(dim) for dim in info.get("logical_shape", info["shape"])]


def tensor_dtype(tensor_index: Json, tensor: str) -> str:
    return str(tensor_index["tensors"][tensor]["dtype"])


def tensor_byte_count(tensor_index: Json, tensor: str) -> int:
    return int(tensor_index["tensors"][tensor]["byte_count"])


def tensor_layout(tensor_index: Json, tensor: str) -> str:
    return str(tensor_index["tensors"][tensor].get("layout", ROW_MAJOR_LAYOUT))


def dtype_byte_count(dtype: str) -> int:
    byte_counts = {
        "BF16": 2,
        "F16": 2,
        "F32": 4,
    }
    try:
        return byte_counts[dtype]
    except KeyError as error:
        raise ModelCompileError(f"unsupported dtype {dtype!r}") from error


def compiled_model_slug(model_dir: Path) -> str:
    digest = blake2s(
        str(model_dir.resolve()).encode("utf-8"), digest_size=4
    ).hexdigest()
    return f"model_{digest}"
