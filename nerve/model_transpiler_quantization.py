from nerve.model_transpiler_types import *


def tensor_matrix_shape(tensors: dict[str, Json], tensor_name: str) -> list[int]:
    info = tensors[tensor_name]
    return [int(value) for value in info.get("logical_shape") or info["shape"]]


def find_bias_for_weight(tensors: dict[str, Json], weight: str) -> str | None:
    suffix = (
        ".qweight"
        if weight.endswith(".qweight")
        else ".weight_packed"
        if weight.endswith(".weight_packed")
        else ".weight"
    )
    if not weight.endswith(suffix):
        return None
    bias = f"{weight[: -len(suffix)]}.bias"
    return bias if bias in tensors else None

def synthesize_packed_expert_tensors(
    tensors: dict[str, Json], layer_prefix: str
) -> None:
    """Describe separately stored experts as packed executable tensors.

    The compiler package owns the physical packing.  The model graph only sees
    the common [expert, row, column] circuit parameters used by every sparse
    MoE component, regardless of how the source checkpoint sharded its experts.
    """
    packed_input = f"{layer_prefix}.mlp.experts.gate_up_proj"
    packed_output = f"{layer_prefix}.mlp.experts.down_proj"
    if packed_input in tensors or packed_output in tensors:
        return

    expert_pattern = re.compile(
        rf"^{re.escape(layer_prefix)}\.mlp\.experts\.(\d+)\.gate_proj\."
        r"(weight|weight_packed)$"
    )
    expert_storage = {
        int(match.group(1)): match.group(2)
        for tensor_name in tensors
        if (match := expert_pattern.fullmatch(tensor_name)) is not None
    }
    expert_indices = sorted(expert_storage)
    if not expert_indices:
        return
    if expert_indices != list(range(len(expert_indices))):
        raise ModelTranspileError(
            f"layer prefix {layer_prefix!r} has non-contiguous expert indices"
        )
    storage_suffixes = set(expert_storage.values())
    if len(storage_suffixes) != 1:
        raise ModelTranspileError(
            f"layer prefix {layer_prefix!r} mixes expert storage formats"
        )
    storage_suffix = storage_suffixes.pop()

    gate_weights: list[str] = []
    up_weights: list[str] = []
    down_weights: list[str] = []
    gate_scales: list[str] = []
    up_scales: list[str] = []
    down_scales: list[str] = []
    for expert in expert_indices:
        base = f"{layer_prefix}.mlp.experts.{expert}"
        gate = f"{base}.gate_proj.{storage_suffix}"
        up = f"{base}.up_proj.{storage_suffix}"
        down = f"{base}.down_proj.{storage_suffix}"
        required = (gate, up, down)
        missing = [name for name in required if name not in tensors]
        if missing:
            raise ModelTranspileError(
                f"layer prefix {layer_prefix!r} expert {expert} is missing {missing}"
            )
        gate_weights.append(gate)
        up_weights.append(up)
        down_weights.append(down)
        if tensors[gate].get("dtype") == "F8_E4M3":
            scales = tuple(f"{name}_scale_inv" for name in required)
            missing_scales = [name for name in scales if name not in tensors]
            if missing_scales:
                raise ModelTranspileError(
                    f"layer prefix {layer_prefix!r} expert {expert} is missing "
                    f"FP8 scales {missing_scales}"
                )
            gate_scales.append(scales[0])
            up_scales.append(scales[1])
            down_scales.append(scales[2])
        elif storage_suffix == "weight_packed":
            quantizations = [tensors[name].get("quantization") for name in required]
            if any(
                not isinstance(quantization, dict)
                or quantization.get("format") != "compressed_tensors_pack_quantized"
                for quantization in quantizations
            ):
                raise ModelTranspileError(
                    f"layer prefix {layer_prefix!r} expert {expert} has unsupported "
                    "packed quantization"
                )
            scales = tuple(
                str(quantization["scales"]) for quantization in quantizations
            )
            missing_scales = [name for name in scales if name not in tensors]
            if missing_scales:
                raise ModelTranspileError(
                    f"layer prefix {layer_prefix!r} expert {expert} is missing "
                    f"INT4 scales {missing_scales}"
                )
            gate_scales.append(scales[0])
            up_scales.append(scales[1])
            down_scales.append(scales[2])

    gate_shape = tensor_matrix_shape(tensors, gate_weights[0])
    up_shape = tensor_matrix_shape(tensors, up_weights[0])
    down_shape = tensor_matrix_shape(tensors, down_weights[0])
    if gate_shape != up_shape or down_shape != [gate_shape[1], gate_shape[0]]:
        raise ModelTranspileError(
            f"layer prefix {layer_prefix!r} has incompatible expert projection "
            f"shapes gate={gate_shape}, up={up_shape}, down={down_shape}"
        )
    for gate, up, down in zip(gate_weights, up_weights, down_weights, strict=True):
        if (
            tensor_matrix_shape(tensors, gate) != gate_shape
            or tensor_matrix_shape(tensors, up) != up_shape
            or tensor_matrix_shape(tensors, down) != down_shape
        ):
            raise ModelTranspileError(
                f"layer prefix {layer_prefix!r} has inconsistent expert projection shapes"
            )
    input_parts = [
        tensor_name
        for gate, up in zip(gate_weights, up_weights, strict=True)
        for tensor_name in (gate, up)
    ]
    if storage_suffix == "weight_packed":
        gate_storage_shape = [int(value) for value in tensors[gate_weights[0]]["shape"]]
        up_storage_shape = [int(value) for value in tensors[up_weights[0]]["shape"]]
        down_storage_shape = [int(value) for value in tensors[down_weights[0]]["shape"]]
        if (
            gate_storage_shape != up_storage_shape
            or len(gate_storage_shape) != 2
            or len(down_storage_shape) != 2
        ):
            raise ModelTranspileError(
                f"layer prefix {layer_prefix!r} has incompatible packed expert storage"
            )
        tensors[packed_input] = composite_tensor(
            tensors,
            input_parts,
            [
                len(expert_indices),
                gate_storage_shape[0] * 2,
                gate_storage_shape[1],
            ],
        )
        tensors[packed_input]["logical_shape"] = [
            len(expert_indices),
            gate_shape[0] * 2,
            gate_shape[1],
        ]
        tensors[packed_output] = composite_tensor(
            tensors,
            down_weights,
            [len(expert_indices), *down_storage_shape],
        )
        tensors[packed_output]["logical_shape"] = [
            len(expert_indices),
            down_shape[0],
            down_shape[1],
        ]
    else:
        tensors[packed_input] = composite_tensor(
            tensors,
            input_parts,
            [len(expert_indices), gate_shape[0] * 2, gate_shape[1]],
        )
        tensors[packed_output] = composite_tensor(
            tensors,
            down_weights,
            [len(expert_indices), down_shape[0], down_shape[1]],
        )

    if gate_scales:
        gate_scale_shape = tensor_matrix_shape(tensors, gate_scales[0])
        up_scale_shape = tensor_matrix_shape(tensors, up_scales[0])
        down_scale_shape = tensor_matrix_shape(tensors, down_scales[0])
        if gate_scale_shape != up_scale_shape:
            raise ModelTranspileError(
                f"layer prefix {layer_prefix!r} has incompatible gate/up scale grids"
            )
        input_scale_parts = [
            tensor_name
            for gate, up in zip(gate_scales, up_scales, strict=True)
            for tensor_name in (gate, up)
        ]
        input_scale_name = (
            f"{packed_input}_scales"
            if storage_suffix == "weight_packed"
            else f"{packed_input}_scale_inv"
        )
        output_scale_name = (
            f"{packed_output}_scales"
            if storage_suffix == "weight_packed"
            else f"{packed_output}_scale_inv"
        )
        tensors[input_scale_name] = composite_tensor(
            tensors,
            input_scale_parts,
            [
                len(expert_indices),
                gate_scale_shape[0] * 2,
                gate_scale_shape[1],
            ],
        )
        tensors[output_scale_name] = composite_tensor(
            tensors,
            down_scales,
            [len(expert_indices), *down_scale_shape],
        )
        if storage_suffix == "weight_packed":
            input_quantization = deepcopy(tensors[gate_weights[0]]["quantization"])
            input_quantization["scales"] = input_scale_name
            output_quantization = deepcopy(tensors[down_weights[0]]["quantization"])
            output_quantization["scales"] = output_scale_name
            tensors[packed_input]["quantization"] = input_quantization
            tensors[packed_output]["quantization"] = output_quantization

    synthesize_shared_expert_input(tensors, layer_prefix)


def synthesize_shared_expert_input(tensors: dict[str, Json], layer_prefix: str) -> None:
    base = f"{layer_prefix}.mlp.shared_expert"
    packed = f"{base}.gate_up_proj"
    gate = f"{base}.gate_proj.weight"
    up = f"{base}.up_proj.weight"
    if packed in tensors or gate not in tensors or up not in tensors:
        return
    gate_shape = tensor_matrix_shape(tensors, gate)
    if tensor_matrix_shape(tensors, up) != gate_shape:
        raise ModelTranspileError(
            f"layer prefix {layer_prefix!r} has incompatible shared expert gate/up shapes"
        )
    tensors[packed] = composite_tensor(
        tensors, [gate, up], [gate_shape[0] * 2, gate_shape[1]]
    )
    if tensors[gate].get("dtype") == "F8_E4M3":
        gate_scale = f"{gate}_scale_inv"
        up_scale = f"{up}_scale_inv"
        if gate_scale not in tensors or up_scale not in tensors:
            raise ModelTranspileError(
                f"layer prefix {layer_prefix!r} shared FP8 expert is missing scales"
            )
        scale_shape = tensor_matrix_shape(tensors, gate_scale)
        if tensor_matrix_shape(tensors, up_scale) != scale_shape:
            raise ModelTranspileError(
                f"layer prefix {layer_prefix!r} has incompatible shared expert scale grids"
            )
        tensors[f"{packed}_scale_inv"] = composite_tensor(
            tensors,
            [gate_scale, up_scale],
            [scale_shape[0] * 2, scale_shape[1]],
        )


def annotate_packed_linear_tensors(model_dir: Path, tensors: dict[str, Json]) -> None:
    config = read_json(model_dir / "config.json")
    quantization = config.get("quantization_config")
    if not isinstance(quantization, dict):
        return
    packing_format = str(
        quantization.get("packing_format") or quantization.get("format") or ""
    )
    if packing_format == "pack-quantized":
        annotate_compressed_tensors_packed_linears(quantization, tensors)
        return
    if packing_format not in {"auto_round:auto_gptq", "auto_round:gptq"}:
        return
    bits = int(quantization.get("bits") or 0)
    if bits <= 0 or 32 % bits:
        raise ModelTranspileError(
            f"packed linear format {packing_format!r} has invalid bit width {bits}"
        )
    pack_factor = 32 // bits
    configured_group_size = int(quantization.get("group_size") or 0)

    for name, info in tuple(tensors.items()):
        if not name.endswith(".qweight"):
            continue
        base = name[: -len(".qweight")]
        qzeros_name = f"{base}.qzeros"
        scales_name = f"{base}.scales"
        qzeros = tensors.get(qzeros_name)
        scales = tensors.get(scales_name)
        if qzeros is None or scales is None:
            raise ModelTranspileError(
                f"packed linear tensor {name!r} is missing qzeros or scales"
            )
        packed_shape = [int(value) for value in info.get("shape", [])]
        zero_shape = [int(value) for value in qzeros.get("shape", [])]
        scale_shape = [int(value) for value in scales.get("shape", [])]
        if info.get("dtype") != "I32" or qzeros.get("dtype") != "I32":
            raise ModelTranspileError(
                f"packed linear tensor {name!r} requires I32 qweight and qzeros"
            )
        if scales.get("dtype") not in {"F16", "BF16"}:
            raise ModelTranspileError(
                f"packed linear tensor {name!r} has unsupported scale dtype "
                f"{scales.get('dtype')!r}"
            )
        if len(packed_shape) != 2 or len(scale_shape) != 2 or len(zero_shape) != 2:
            raise ModelTranspileError(
                f"packed linear tensor {name!r} has invalid qweight/qzeros/scales shapes"
            )
        input_features = packed_shape[0] * pack_factor
        output_features = packed_shape[1]
        group_count = scale_shape[0]
        if group_count <= 0 or input_features % group_count:
            raise ModelTranspileError(
                f"packed linear tensor {name!r} cannot infer an integer group size"
            )
        group_size = input_features // group_count
        expected_zero_shape = [
            group_count,
            (output_features + pack_factor - 1) // pack_factor,
        ]
        if scale_shape[1] != output_features or zero_shape != expected_zero_shape:
            raise ModelTranspileError(
                f"packed linear tensor {name!r} has incompatible qzeros {zero_shape} "
                f"or scales {scale_shape}"
            )
        if configured_group_size > 0 and group_size != configured_group_size:
            raise ModelTranspileError(
                f"packed linear tensor {name!r} implies group size {group_size}, "
                f"not configured size {configured_group_size}"
            )
        info["logical_shape"] = [output_features, input_features]
        info["quantization"] = {
            "format": "auto_gptq",
            "bits": bits,
            "group_size": group_size,
            "symmetric": bool(quantization.get("sym", True)),
            "zero_point_add": 1,
            "qzeros": qzeros_name,
            "scales": scales_name,
        }


def annotate_compressed_tensors_packed_linears(
    quantization: Json, tensors: dict[str, Json]
) -> None:
    config_groups = quantization.get("config_groups")
    if not isinstance(config_groups, dict):
        raise ModelTranspileError(
            "compressed-tensors pack-quantized format has no config groups"
        )
    schemes = [
        group.get("weights")
        for group in config_groups.values()
        if isinstance(group, dict)
        and (group.get("format") or quantization.get("format")) == "pack-quantized"
        and isinstance(group.get("weights"), dict)
    ]
    if len(schemes) != 1:
        raise ModelTranspileError(
            "compressed-tensors pack-quantized format requires one structural weight scheme"
        )
    scheme = schemes[0]
    bits = int(scheme.get("num_bits") or 0)
    group_size = int(scheme.get("group_size") or 0)
    symmetric = bool(scheme.get("symmetric", True))
    if bits <= 0 or 32 % bits or group_size <= 0:
        raise ModelTranspileError(
            f"compressed-tensors packed linear has invalid {bits}-bit group size {group_size}"
        )
    pack_factor = 32 // bits

    for name, info in tuple(tensors.items()):
        if not name.endswith(".weight_packed"):
            continue
        base = name[: -len(".weight_packed")]
        scale_name = f"{base}.weight_scale"
        scale = tensors.get(scale_name)
        if scale is None:
            raise ModelTranspileError(
                f"compressed packed linear tensor {name!r} is missing {scale_name!r}"
            )
        packed_shape = [int(value) for value in info.get("shape", [])]
        scale_shape = [int(value) for value in scale.get("shape", [])]
        if info.get("dtype") != "I32" or len(packed_shape) != 2:
            raise ModelTranspileError(
                f"compressed packed linear tensor {name!r} must be an I32 matrix"
            )
        if scale.get("dtype") not in {"F16", "BF16"} or len(scale_shape) != 2:
            raise ModelTranspileError(
                f"compressed packed linear tensor {name!r} has incompatible scales"
            )
        output_features = packed_shape[0]
        input_features = scale_shape[1] * group_size
        expected_packed_shape = [
            output_features,
            (input_features + pack_factor - 1) // pack_factor,
        ]
        expected_scale_shape = [output_features, input_features // group_size]
        if packed_shape != expected_packed_shape or scale_shape != expected_scale_shape:
            raise ModelTranspileError(
                f"compressed packed linear tensor {name!r} has incompatible packed "
                f"shape {packed_shape} or scale shape {scale_shape}"
            )
        info["logical_shape"] = [output_features, input_features]
        info["quantization"] = {
            "format": "compressed_tensors_pack_quantized",
            "bits": bits,
            "group_size": group_size,
            "symmetric": symmetric,
            "signed_offset": 1 << (bits - 1),
            "scales": scale_name,
        }


def attach_packed_linear_quantization(
    tensors: dict[str, Json], layer_tensors: dict[str, str]
) -> None:
    additions: dict[str, str] = {}
    for parameter_id, tensor_name in tuple(layer_tensors.items()):
        quantization = tensors[tensor_name].get("quantization")
        if not isinstance(quantization, dict):
            continue
        if "qzeros" in quantization:
            additions[f"{parameter_id}_qzeros"] = str(quantization["qzeros"])
        additions[f"{parameter_id}_scales"] = str(quantization["scales"])
    layer_tensors.update(additions)


def composite_tensor(
    tensors: dict[str, Json], part_names: Iterable[str], shape: list[int]
) -> Json:
    names = list(part_names)
    if not names:
        raise ModelTranspileError("cannot create an empty composite tensor")
    dtype = tensors[names[0]].get("dtype")
    if any(tensors[name].get("dtype") != dtype for name in names):
        raise ModelTranspileError(f"composite tensor parts have mixed dtypes: {names}")
    byte_count = sum(int(tensors[name]["byte_count"]) for name in names)
    return {
        "dtype": dtype,
        "shape": shape,
        "data_offsets": [0, byte_count],
        "parameter_count": math.prod(shape),
        "byte_count": byte_count,
        "layout_hint": "row_major",
        "source_parts": [
            {
                "tensor": name,
                "source_file": tensors[name]["source_file"],
                "source_header_bytes": int(tensors[name]["source_header_bytes"]),
                "data_offsets": list(tensors[name]["data_offsets"]),
                "byte_count": int(tensors[name]["byte_count"]),
            }
            for name in names
        ],
    }


def attach_block_quantization_scales(
    tensors: dict[str, Json], layer_tensors: dict[str, str]
) -> None:
    """Attach scale tensors to quantized parameters by tensor structure.

    Safetensors FP8 checkpoints store a block scale beside each quantized
    matrix.  Keeping the scale as an explicit circuit parameter lets the
    backend execute the source representation directly instead of silently
    treating one-byte FP8 values as two-byte BF16 values.
    """
    additions: dict[str, str] = {}
    for parameter_id, tensor_name in tuple(layer_tensors.items()):
        tensor = tensors[tensor_name]
        if tensor.get("dtype") != "F8_E4M3":
            continue
        shape = [int(value) for value in tensor.get("shape", [])]
        if len(shape) not in (2, 3):
            raise ModelTranspileError(
                f"FP8 parameter {tensor_name!r} has unsupported shape {shape}; "
                "only block-scaled matrices and expert stacks are executable"
            )
        scale_name = f"{tensor_name}_scale_inv"
        scale = tensors.get(scale_name)
        if scale is None:
            raise ModelTranspileError(
                f"FP8 parameter {tensor_name!r} is missing block scale tensor "
                f"{scale_name!r}"
            )
        scale_shape = [int(value) for value in scale.get("shape", [])]
        if scale.get("dtype") != "BF16" or len(scale_shape) != len(shape):
            raise ModelTranspileError(
                f"FP8 parameter {tensor_name!r} has incompatible block scale "
                f"dtype {scale.get('dtype')!r} and shape {scale_shape}"
            )
        if any(value <= 0 for value in scale_shape):
            raise ModelTranspileError(
                f"FP8 parameter {tensor_name!r} has empty block scale shape {scale_shape}"
            )
        additions[f"{parameter_id}_scale_inv"] = scale_name
    layer_tensors.update(additions)


def add_optional_linear_biases(
    tensors: dict[str, Json],
    layer_tensors: dict[str, str],
    weight_ids: Iterable[str],
) -> None:
    for weight_id in weight_ids:
        bias = find_bias_for_weight(tensors, layer_tensors[weight_id])
        if bias is not None:
            layer_tensors[f"{weight_id}_bias"] = bias

