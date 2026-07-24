from nerve.model_package_common import *
from nerve.model_package_tensors import *

def stream_control_binding_for_node(circuit: Json, node: Json) -> int:
    state_view_signals = {
        output
        for producer in circuit["nodes"]
        if producer.get("op") in {"append_state_update", "rolling_state_update"}
        for output in producer.get("outputs", [])
    }
    signal_bindings = [*node.get("inputs", []), *node.get("outputs", [])]
    state_view_binding_count = sum(
        signal in state_view_signals for signal in signal_bindings
    )
    return (
        len(node.get("inputs", []))
        + len(node.get("outputs", []))
        + len(node.get("params", []))
        + len(node.get("state_reads", []))
        + len(node.get("state_writes", []))
        + state_view_binding_count
    )


def copy_tokenizer_package(model_dir: Path, dest_dir: Path) -> Json:
    tokenizer_json = model_dir / "tokenizer.json"
    if not tokenizer_json.is_file():
        raise ModelCompileError(
            f"source model does not contain required tokenizer file {tokenizer_json}"
        )

    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    copied_files = []
    for filename in TOKENIZER_PACKAGE_FILES:
        source = model_dir / filename
        if source.is_file():
            shutil.copy2(source, dest_dir / filename)
            copied_files.append(filename)

    if "chat_template.jinja" not in copied_files:
        tokenizer_config_path = model_dir / "tokenizer_config.json"
        if tokenizer_config_path.is_file():
            tokenizer_config = read_json(tokenizer_config_path)
            inline_template = tokenizer_config.get("chat_template")
            if isinstance(inline_template, str):
                (dest_dir / "chat_template.jinja").write_text(inline_template)
                copied_files.append("chat_template.jinja")

    return {
        "path": TOKENIZER_PACKAGE_DIR,
        "files": copied_files,
    }


def write_runtime_config_package(model_graph: Json, package_dir: Path) -> None:
    token_ids = model_graph["token_ids"]
    write_json(
        package_dir / CONFIG_PACKAGE_FILE,
        {
            "schema": "nerve.runtime_model_config.v1",
            "bos_token_id": token_ids["bos"],
            "eos_token_id": token_ids["eos"],
            "pad_token_id": token_ids["pad"],
            "dimensions": model_graph["dimensions"],
            "numerics": model_graph["numerics"],
        },
    )


def referenced_tensor_index(
    tensor_index: Json,
    *,
    model_graph: Json,
    lowered_index: Json,
    lowered_dir: Path,
) -> Json:
    referenced = {
        model_graph["graph"]["input_transducer"]["params"]["weight"]["tensor"]
    }
    for component in model_graph["graph"]["output_transducer"]["components"]:
        referenced.update(ref["tensor"] for ref in component.get("params", {}).values())
    for circuit_ref in all_lowered_circuit_refs(lowered_index):
        circuit = read_json(lowered_dir / circuit_ref["circuit"])
        referenced.update(
            ref["tensor"] for ref in circuit["parameters"]["refs"].values()
        )

    missing = sorted(referenced - set(tensor_index["tensors"]))
    if missing:
        raise ModelCompileError(
            f"compiled circuit graph references missing tensors: {', '.join(missing)}"
        )
    selected = deepcopy(tensor_index)
    selected["tensors"] = {
        name: deepcopy(tensor_index["tensors"][name]) for name in sorted(referenced)
    }
    selected["totals"] = {
        "tensor_count": len(selected["tensors"]),
        "parameter_count": sum(
            int(info["parameter_count"]) for info in selected["tensors"].values()
        ),
        "byte_count": sum(
            int(info["byte_count"]) for info in selected["tensors"].values()
        ),
    }
    return selected


def all_lowered_circuit_refs(lowered_index: Json) -> list[Json]:
    refs = list(lowered_index["graph"]["circuits"])
    refs.extend(
        circuit_ref
        for draft in lowered_index.get("draft_execution_graphs", [])
        for circuit_ref in draft["circuits"]
    )
    return refs


def copy_tensor_package(
    tensor_index: Json,
    package_dir: Path,
    *,
    progress: Callable[[int, int, str], None] | None = None,
    cancel_requested: Callable[[], bool] | None = None,
) -> Json:
    weights_dir = package_dir / WEIGHTS_PACKAGE_DIR
    if weights_dir.exists():
        shutil.rmtree(weights_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)

    if not tensor_index["tensors"]:
        raise ModelCompileError("tensor index does not declare any source_file entries")

    packaged = deepcopy(tensor_index)
    compiled_sources = []
    tensors = sorted(packaged["tensors"].items())
    total = len(tensors)
    derived_groups_written: set[str] = set()
    derived_tensors_written: set[str] = set()
    for index, (tensor_name, info) in enumerate(tensors, start=1):
        check_compile_cancelled(cancel_requested)
        if tensor_name in derived_tensors_written:
            continue
        if progress is not None:
            progress(index, total, tensor_name)
        layout = ROW_MAJOR_LAYOUT
        digest = blake2s(tensor_name.encode("utf-8"), digest_size=8).hexdigest()
        destination = weights_dir / f"tensor_{digest}.safetensors"
        derivation = info.get("derived")
        if (
            isinstance(derivation, dict)
            and derivation.get("kind") == "bf16_to_fp8_e4m3_scale"
        ):
            if str(derivation["group"]) not in derived_groups_written:
                raise ModelCompileError(
                    f"derived FP8 scale tensor {tensor_name!r} was visited before "
                    "its weight tensor"
                )
            continue
        if (
            isinstance(derivation, dict)
            and derivation.get("kind") == "bf16_to_fp8_e4m3"
        ):
            scale_tensor_name = str(derivation["scale_tensor"])
            scale_info = packaged["tensors"].get(scale_tensor_name)
            if not isinstance(scale_info, dict):
                raise ModelCompileError(
                    f"derived FP8 weight tensor {tensor_name!r} references missing "
                    f"scale tensor {scale_tensor_name!r}"
                )
            scale_digest = blake2s(
                scale_tensor_name.encode("utf-8"), digest_size=8
            ).hexdigest()
            scale_destination = weights_dir / f"tensor_{scale_digest}.safetensors"
            group_headers_and_digests = (
                write_compiled_derived_fp8_e4m3_output_projection(
                    weight_tensor_name=tensor_name,
                    weight_info=info,
                    weight_destination=destination,
                    scale_tensor_name=scale_tensor_name,
                    scale_info=scale_info,
                    scale_destination=scale_destination,
                    layout=layout,
                )
            )
            for emitted_name, emitted_destination in (
                (tensor_name, destination),
                (scale_tensor_name, scale_destination),
            ):
                emitted_info = packaged["tensors"][emitted_name]
                header_bytes, data_sha256 = group_headers_and_digests[emitted_name]
                relative_destination = relative_json_path(package_dir, emitted_destination)
                emitted_info["source_file"] = relative_destination
                emitted_info["data_offsets"] = [0, int(emitted_info["byte_count"])]
                emitted_info["data_sha256"] = data_sha256
                emitted_info["layout"] = layout
                emitted_info.pop("derived", None)
                compiled_sources.append(
                    {
                        "path": relative_destination,
                        "safetensors_header_bytes": header_bytes,
                        "metadata": {
                            "format": "nerve",
                            "layout": layout,
                        },
                    }
                )
            derived_groups_written.add(str(derivation["group"]))
            derived_tensors_written.update({tensor_name, scale_tensor_name})
            continue
        if (
            isinstance(derivation, dict)
            and derivation.get("kind") == "fp8_e4m3_to_bf16"
        ):
            header_bytes, data_sha256 = write_compiled_derived_bf16_from_fp8_e4m3(
                tensor_name=tensor_name,
                info=info,
                destination=destination,
                layout=layout,
            )
            relative_destination = relative_json_path(package_dir, destination)
            info["source_file"] = relative_destination
            info["data_offsets"] = [0, int(info["byte_count"])]
            info["data_sha256"] = data_sha256
            info["layout"] = layout
            info.pop("derived", None)
            compiled_sources.append(
                {
                    "path": relative_destination,
                    "safetensors_header_bytes": header_bytes,
                    "metadata": {
                        "format": "nerve",
                        "layout": layout,
                    },
                }
            )
            continue
        if (
            isinstance(derivation, dict)
            and derivation.get("kind") == "fp8_e4m3_to_q8_0"
        ):
            header_bytes, data_sha256 = write_compiled_derived_q8_0_from_fp8_e4m3(
                tensor_name=tensor_name,
                info=info,
                destination=destination,
                layout=layout,
            )
            relative_destination = relative_json_path(package_dir, destination)
            info["source_file"] = relative_destination
            info["data_offsets"] = [0, int(info["byte_count"])]
            info["data_sha256"] = data_sha256
            info["layout"] = layout
            info.pop("derived", None)
            compiled_sources.append(
                {
                    "path": relative_destination,
                    "safetensors_header_bytes": header_bytes,
                    "metadata": {
                        "format": "nerve",
                        "layout": layout,
                    },
                }
            )
            continue
        if info.get("source_parts"):
            header_bytes, data_sha256 = write_compiled_composite_tensor(
                tensor_name=tensor_name,
                info=info,
                destination=destination,
                layout=layout,
            )
        else:
            source = Path(info["source_file"])
            if not source.is_file():
                raise ModelCompileError(f"tensor source file does not exist: {source}")
            header_bytes, data_sha256 = write_compiled_tensor(
                tensor_name=tensor_name,
                info=info,
                source=source,
                destination=destination,
                layout=layout,
            )
        relative_destination = relative_json_path(package_dir, destination)
        info["source_file"] = relative_destination
        info["data_offsets"] = [0, int(info["byte_count"])]
        info["data_sha256"] = data_sha256
        info["layout"] = layout
        info.pop("source_parts", None)
        info.pop("source_header_bytes", None)
        info.pop("layout_hint", None)
        compiled_sources.append(
            {
                "path": relative_destination,
                "safetensors_header_bytes": header_bytes,
                "metadata": {
                    "format": "nerve",
                    "layout": layout,
                },
            }
        )

    packaged["source"] = {
        "packaged": True,
        "compiled": True,
        "weights_dir": WEIGHTS_PACKAGE_DIR,
        "weights_file": compiled_sources[0]["path"],
        "weights_files": compiled_sources,
    }

    write_json(package_dir / "tensors.json", packaged)
    return packaged
