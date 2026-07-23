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
