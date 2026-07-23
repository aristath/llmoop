from nerve.model_transpiler_types import *
from nerve.model_transpiler_quantization import annotate_packed_linear_tensors

def make_tensor_index(model_dir: Path) -> Json:
    tensor_entries: Json = {}
    total_params = 0
    total_bytes = 0
    source_files: list[Json] = []

    for weights_file in discover_safetensor_files(model_dir):
        header_len, header = read_safetensors_header(weights_file)
        source_files.append(
            {
                "path": str(weights_file),
                "safetensors_header_bytes": header_len,
                "metadata": header.get("__metadata__", {}),
            }
        )
        for name, info in sorted(header.items()):
            if name == "__metadata__":
                continue
            shape = info["shape"]
            offsets = info["data_offsets"]
            params = math.prod(shape)
            byte_count = offsets[1] - offsets[0]
            total_params += params
            total_bytes += byte_count
            tensor_entries[name] = {
                "dtype": info["dtype"],
                "shape": shape,
                "data_offsets": offsets,
                "parameter_count": params,
                "byte_count": byte_count,
                "source_file": str(weights_file),
                "source_header_bytes": header_len,
            }

    annotate_packed_linear_tensors(model_dir, tensor_entries)

    return {
        "schema": "nerve.tensor_index.v1",
        "source": {
            "model_dir": str(model_dir),
            "weights_file": source_files[0]["path"],
            "weights_files": source_files,
        },
        "totals": {
            "tensor_count": len(tensor_entries),
            "parameter_count": total_params,
            "byte_count": total_bytes,
        },
        "tensors": tensor_entries,
    }


def segment_per_layer_embedding_parameters(
    structure: ModelStructure, tensor_index: Json
) -> None:
    """Expose oversized packed PLE tables as row-aligned compiled tensors.

    Each alias points at a disjoint contiguous source range. Packaging then
    emits ordinary standalone tensors, so the runtime remains unaware of the
    source checkpoint's oversized allocation.
    """

    tensors = tensor_index["tensors"]
    segmented: dict[str, list[str]] = {}
    for layer in structure.layers:
        tensor_name = layer.tensors.get("per_layer_embedding")
        if tensor_name is None:
            continue
        if tensor_name not in segmented:
            info = tensors[tensor_name]
            shape = [int(value) for value in info["shape"]]
            if info.get("dtype") != "BF16" or len(shape) != 2:
                raise ModelTranspileError(
                    f"per-layer embedding tensor {tensor_name!r} must be a BF16 matrix"
                )
            row_bytes = shape[1] * 2
            rows_per_chunk = MAX_SHADER_PARAMETER_CHUNK_BYTES // row_bytes
            if rows_per_chunk == 0:
                raise ModelTranspileError(
                    f"per-layer embedding tensor {tensor_name!r} has a row wider than "
                    "the compiled shader parameter chunk limit"
                )
            source_offsets = [int(value) for value in info["data_offsets"]]
            chunk_names: list[str] = []
            for chunk_index, row_start in enumerate(range(0, shape[0], rows_per_chunk)):
                row_end = min(row_start + rows_per_chunk, shape[0])
                byte_start = row_start * row_bytes
                byte_end = row_end * row_bytes
                chunk_name = f"{tensor_name}.__nerve_chunk_{chunk_index:03d}"
                tensors[chunk_name] = {
                    "dtype": "BF16",
                    "shape": [row_end - row_start, shape[1]],
                    "data_offsets": [
                        source_offsets[0] + byte_start,
                        source_offsets[0] + byte_end,
                    ],
                    "parameter_count": (row_end - row_start) * shape[1],
                    "byte_count": byte_end - byte_start,
                    "source_file": info["source_file"],
                    "source_header_bytes": int(info["source_header_bytes"]),
                }
                chunk_names.append(chunk_name)
            segmented[tensor_name] = chunk_names

        del layer.tensors["per_layer_embedding"]
        for chunk_index, chunk_name in enumerate(segmented[tensor_name]):
            layer.tensors[f"per_layer_embedding_chunk_{chunk_index}"] = chunk_name


def discover_safetensor_files(model_dir: Path) -> tuple[Path, ...]:
    single = model_dir / "model.safetensors"
    if single.exists():
        return (single,)

    index_file = model_dir / "model.safetensors.index.json"
    if index_file.exists():
        index = read_json(index_file)
        files = sorted(
            {model_dir / filename for filename in index.get("weight_map", {}).values()}
        )
        if files:
            return tuple(files)

    files = tuple(sorted(model_dir.glob("*.safetensors")))
    if files:
        return files

    raise ModelTranspileError(f"no safetensors checkpoint files found in {model_dir}")


def read_safetensors_header(path: Path) -> tuple[int, Json]:
    with path.open("rb") as handle:
        header_len = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_len))
    return header_len, header


def find_first_tensor(
    tensors: dict[str, Json], candidates: Iterable[str], *, role: str
) -> str:
    match = find_first_existing_tensor(tensors, candidates)
    if match is None:
        raise ModelTranspileError(f"could not discover {role} tensor")
    return match


def find_first_existing_tensor(
    tensors: dict[str, Json], candidates: Iterable[str]
) -> str | None:
    for name in candidates:
        if name in tensors:
            return name
        if name.endswith(".weight"):
            base = name[: -len(".weight")]
            for packed_name in (f"{base}.qweight", f"{base}.weight_packed"):
                if packed_name in tensors and tensors[packed_name].get("quantization"):
                    return packed_name
    return None


def find_layer_tensor(
    tensors: dict[str, Json],
    prefix: str,
    suffixes: Iterable[str],
    *,
    role: str,
) -> str:
    return find_first_tensor(
        tensors, (f"{prefix}.{suffix}" for suffix in suffixes), role=role
    )


def find_optional_layer_tensor(
    tensors: dict[str, Json], prefix: str, suffixes: Iterable[str]
) -> str | None:
    return find_first_existing_tensor(
        tensors, (f"{prefix}.{suffix}" for suffix in suffixes)
    )


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


def tensor_matrix_shape(tensors: dict[str, Json], tensor_name: str) -> list[int]:
    info = tensors[tensor_name]
    shape = [int(value) for value in info.get("logical_shape", info.get("shape", []))]
    if len(shape) != 2:
        raise ModelTranspileError(
            f"tensor {tensor_name!r} is not a matrix: shape {shape}"
        )
    return shape


def tensor_ref(name: str) -> dict[str, str]:
    return {"tensor": name}
