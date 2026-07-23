from nerve.model_package_common import *
from nerve.model_package_tensors import dtype_byte_count, tensor_dtype, tensor_shape


def derive_output_projection_tensors(model_graph: Json, tensor_index: Json) -> None:
    """Compile the output projection into the best generic artifact we can use.

    Some FP8 checkpoints keep the final LM head in BF16 even when the decoder
    blocks are native FP8. Leaving that tensor as BF16 forces the runtime to
    stream a multi-gigabyte projection every generated token. The compiler owns
    this representation choice: it can derive a block-scaled FP8 projection
    tensor and expose the extra scale tensor through the ordinary component
    parameter contract.
    """

    output_projection = output_projection_component(model_graph)
    params = output_projection.setdefault("params", {})
    weight_ref = params.get("weight")
    if not isinstance(weight_ref, dict) or not isinstance(weight_ref.get("tensor"), str):
        raise ModelCompileError("output projection component has no weight tensor")
    source_tensor = weight_ref["tensor"]
    dtype = tensor_dtype(tensor_index, source_tensor)
    if dtype == "F8_E4M3":
        scale_tensor = f"{source_tensor}_scale_inv"
        if scale_tensor in tensor_index["tensors"]:
            params["weight_scale_inv"] = {"tensor": scale_tensor}
            update_draft_output_projection_params(
                model_graph,
                source_tensor=source_tensor,
                weight_tensor=source_tensor,
                scale_tensor=scale_tensor,
            )
        return
    if dtype != "BF16":
        return

    shape = tensor_shape(tensor_index, source_tensor)
    if len(shape) != 2:
        return
    output_rows, input_columns = shape
    block_rows = FP8_LINEAR_TILE_ROWS[-1]
    block_columns = 128
    if input_columns % block_columns != 0:
        return

    source_info = tensor_index["tensors"][source_tensor]
    derived_weight = f"{source_tensor}.__nerve_output_fp8_e4m3"
    derived_scale = f"{derived_weight}_scale_inv"
    group = f"output_projection:{source_tensor}"
    scale_shape = [
        (output_rows + block_rows - 1) // block_rows,
        (input_columns + block_columns - 1) // block_columns,
    ]
    tensor_index["tensors"][derived_weight] = {
        "dtype": "F8_E4M3",
        "shape": shape,
        "parameter_count": output_rows * input_columns,
        "byte_count": output_rows * input_columns * dtype_byte_count("F8_E4M3"),
        "derived": {
            "kind": "bf16_to_fp8_e4m3",
            "group": group,
            "source_tensor": source_tensor,
            "source_file": source_info["source_file"],
            "source_header_bytes": int(source_info["source_header_bytes"]),
            "data_offsets": list(source_info["data_offsets"]),
            "source_shape": shape,
            "block_rows": block_rows,
            "block_columns": block_columns,
            "scale_tensor": derived_scale,
        },
    }
    tensor_index["tensors"][derived_scale] = {
        "dtype": "BF16",
        "shape": scale_shape,
        "parameter_count": scale_shape[0] * scale_shape[1],
        "byte_count": scale_shape[0] * scale_shape[1] * dtype_byte_count("BF16"),
        "derived": {
            "kind": "bf16_to_fp8_e4m3_scale",
            "group": group,
            "source_tensor": source_tensor,
        },
    }
    params["weight"] = {"tensor": derived_weight}
    params["weight_scale_inv"] = {"tensor": derived_scale}
    output_projection["compiled_parameter_dtype"] = "F8_E4M3"
    output_projection["compiled_from_tensor"] = source_tensor
    update_draft_output_projection_params(
        model_graph,
        source_tensor=source_tensor,
        weight_tensor=derived_weight,
        scale_tensor=derived_scale,
    )


def update_draft_output_projection_params(
    model_graph: Json,
    *,
    source_tensor: str,
    weight_tensor: str,
    scale_tensor: str,
) -> None:
    for draft in model_graph["graph"].get("draft_execution_graphs", []):
        output = draft.get("output_transducer", {})
        params = output.get("params", {})
        projection = params.get("projection")
        if not isinstance(projection, dict) or projection.get("tensor") != source_tensor:
            continue
        params["projection"] = {"tensor": weight_tensor}
        params["weight_scale_inv"] = {"tensor": scale_tensor}


def output_projection_component(model_graph: Json) -> Json:
    for component in model_graph["graph"]["output_transducer"]["components"]:
        if component.get("type") == "linear_projection":
            return component
    raise ModelCompileError("model graph has no output linear projection component")


def derive_internal_q8_linear_tensors(
    lowered_index: Json, lowered_dir: Path, tensor_index: Json
) -> None:
    """Rewrite eligible lowered FP8 linears to NERVE's internal Q8_0 format.

    This is intentionally structure-driven rather than model-name-driven.  If a
    lowered circuit exposes a standalone FP8 linear/residual-linear node with
    its adjacent block-scale parameter, the package compiler can choose a
    runtime-native Q8_0 tensor for that node.  Fused and parallel linears stay in
    their source format until their Q8 kernels exist.
    """

    for circuit_ref in lowered_circuit_refs(lowered_index):
        circuit_path = lowered_dir / circuit_ref["circuit"]
        circuit = read_json(circuit_path)
        if rewrite_circuit_fp8_linears_to_q8(circuit, tensor_index):
            write_json(circuit_path, circuit)


def lowered_circuit_refs(lowered_index: Json) -> list[Json]:
    refs = list(lowered_index["graph"]["circuits"])
    refs.extend(
        circuit_ref
        for draft in lowered_index.get("draft_execution_graphs", [])
        for circuit_ref in draft["circuits"]
    )
    return refs


def rewrite_circuit_fp8_linears_to_q8(circuit: Json, tensor_index: Json) -> bool:
    rewritten = False
    refs = circuit["parameters"]["refs"]
    used_params_after_rewrite: set[str] = set()
    for node in circuit.get("nodes", []):
        params = list(node.get("params", []))
        if node.get("op") not in {"linear", "linear_residual"} or not params:
            used_params_after_rewrite.update(params)
            continue
        weight_id = str(params[0])
        scale_id = f"{weight_id}_scale_inv"
        if len(params) < 2 or params[1] != scale_id:
            used_params_after_rewrite.update(params)
            continue
        weight_ref = refs.get(weight_id)
        scale_ref = refs.get(scale_id)
        if not isinstance(weight_ref, dict) or not isinstance(scale_ref, dict):
            used_params_after_rewrite.update(params)
            continue
        weight_tensor = weight_ref.get("tensor")
        scale_tensor = scale_ref.get("tensor")
        if (
            not isinstance(weight_tensor, str)
            or not isinstance(scale_tensor, str)
            or tensor_dtype(tensor_index, weight_tensor) != "F8_E4M3"
            or tensor_dtype(tensor_index, scale_tensor) != "BF16"
        ):
            used_params_after_rewrite.update(params)
            continue
        shape = tensor_shape(tensor_index, weight_tensor)
        if len(shape) != 2 or shape[1] % Q8_0_GROUP_SIZE:
            used_params_after_rewrite.update(params)
            continue
        q8_tensor = ensure_q8_tensor_for_fp8_pair(
            tensor_index,
            weight_tensor=weight_tensor,
            scale_tensor=scale_tensor,
        )
        weight_ref["tensor"] = q8_tensor
        node["params"] = [weight_id, *params[2:]]
        used_params_after_rewrite.update(node["params"])
        rewritten = True

    if rewritten:
        for parameter_id in list(refs):
            if parameter_id not in used_params_after_rewrite:
                refs.pop(parameter_id)
    return rewritten


def ensure_q8_tensor_for_fp8_pair(
    tensor_index: Json, *, weight_tensor: str, scale_tensor: str
) -> str:
    q8_tensor = f"{weight_tensor}.__nerve_q8_0"
    if q8_tensor in tensor_index["tensors"]:
        return q8_tensor

    weight_info = tensor_index["tensors"][weight_tensor]
    scale_info = tensor_index["tensors"][scale_tensor]
    shape = tensor_shape(tensor_index, weight_tensor)
    scale_shape = tensor_shape(tensor_index, scale_tensor)
    if len(shape) != 2 or len(scale_shape) != 2:
        raise ModelCompileError(
            f"cannot derive Q8_0 from non-matrix FP8 tensor {weight_tensor!r}"
        )
    output_rows, input_columns = shape
    if input_columns % Q8_0_GROUP_SIZE:
        raise ModelCompileError(
            f"cannot derive Q8_0 tensor {q8_tensor!r}; input width "
            f"{input_columns} is not {Q8_0_GROUP_SIZE}-aligned"
        )
    group_count = input_columns // Q8_0_GROUP_SIZE
    tensor_index["tensors"][q8_tensor] = {
        "dtype": "Q8_0",
        "shape": [output_rows, group_count, Q8_0_BLOCK_WORDS],
        "logical_shape": shape,
        "parameter_count": output_rows * input_columns,
        "byte_count": output_rows * group_count * Q8_0_BLOCK_BYTE_COUNT,
        "layout": ROW_MAJOR_LAYOUT,
        "quantization": {
            "format": "nerve_q8_0",
            "source_format": "block_scaled_fp8_e4m3",
            "group_size": Q8_0_GROUP_SIZE,
            "block_byte_count": Q8_0_BLOCK_BYTE_COUNT,
        },
        "derived": {
            "kind": "fp8_e4m3_to_q8_0",
            "source_tensor": weight_tensor,
            "source_file": weight_info["source_file"],
            "source_header_bytes": int(weight_info["source_header_bytes"]),
            "data_offsets": list(weight_info["data_offsets"]),
            "source_shape": shape,
            "scale_tensor": scale_tensor,
            "scale_source_file": scale_info["source_file"],
            "scale_source_header_bytes": int(scale_info["source_header_bytes"]),
            "scale_data_offsets": list(scale_info["data_offsets"]),
            "scale_shape": scale_shape,
        },
    }
    return q8_tensor
