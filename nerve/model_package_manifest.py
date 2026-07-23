from nerve.model_package_common import *
from nerve.model_package_assets import all_lowered_circuit_refs
from nerve.model_package_batching import *
from nerve.model_package_shaders import *
from nerve.model_package_shader_compiler import (
    compile_shader_artifacts,
    compiled_shader_path,
)
from nerve.model_package_shader_templates import copy_shader_templates
from nerve.model_package_tensors import *

def build_vulkan_resident_package_manifest(
    *,
    model_graph: Json,
    tensor_index: Json,
    lowered_index: Json,
    lowered_dir: Path,
    package_dir: Path,
    package_id: str,
    shader_source_dir: Path,
    tokenizer_manifest: Json,
    event_sink: Callable[[Json], None] | None = None,
    cancel_requested: Callable[[], bool] | None = None,
) -> Json:
    dimensions = model_graph["dimensions"]
    hidden_size = int(dimensions["hidden_size"])
    vocab_size = int(dimensions["vocab_size"])
    norm_eps = float(model_graph["numerics"]["rms_norm_eps"])
    norm_weight_offset = float(model_graph["numerics"]["rms_norm_weight_offset"])
    embedding_scale = float(model_graph["numerics"]["embedding_scale"])
    logits_scale = float(model_graph["numerics"]["logits_scale"])
    configured_context = dimensions.get("max_position_embeddings")
    max_context_activations = int(
        configured_context
        if configured_context is not None
        else dimensions.get("attention_window_size") or 4096
    )
    dtype = "BF16"
    dtype_bytes = dtype_byte_count(dtype)
    frame_bytes = hidden_size * dtype_bytes
    logits_bytes = vocab_size * dtype_byte_count("F32")
    sampling = model_graph["sampling"]
    sampler_method = str(sampling["method"])
    sampler_presence_penalty = float(sampling["presence_penalty"])
    sampler_repetition_penalty = float(sampling["repetition_penalty"])
    sampler_uses_token_state = (
        sampler_presence_penalty != 0.0 or sampler_repetition_penalty != 1.0
    )
    sampler_partition_count = 128
    sampler_candidate_local_size_x = 256
    sampler_merge_local_size_x = 256
    sampler_top_k_capacity = min(
        vocab_size,
        max(256, int(sampling.get("top_k", 1))),
    )
    sampler_scratch_byte_capacity = sampler_partition_count * sampler_top_k_capacity * 8
    sampler_state_kernels = []
    if sampler_uses_token_state:
        sampler_state_kernels = [
            {
                "role": "record_current_token",
                "shader_path": compiled_shader_path(
                    f"shaders/record_seen_token_{vocab_size}.comp"
                ),
                "local_size_x": 1,
                "workgroup_count_x": 1,
            },
            {
                "role": "record_token_batch",
                "shader_path": compiled_shader_path(
                    f"shaders/record_seen_tokens_batch64_{vocab_size}.comp"
                ),
                "local_size_x": 64,
                "workgroup_count_x": 1,
            },
        ]
    if sampler_method == "greedy":
        sampler_id = "greedy_sampler"
        sampler_shader_file = (
            f"greedy_sampler_repetition_f32_{vocab_size}"
            f"_rp{shader_float_token(sampler_repetition_penalty)}"
            f"_pp{shader_float_token(sampler_presence_penalty)}.comp"
            if sampler_uses_token_state
            else f"greedy_sampler_f32_{vocab_size}.comp"
        )
        sampler_temperature = 1.0
        sampler_top_k = 1
        sampler_top_p = 1.0
        sampler_min_p = 0.0
        sampler_kernels = sampler_state_kernels + [
            {
                "role": "sample_logits",
                "shader_path": compiled_shader_path(f"shaders/{sampler_shader_file}"),
                "local_size_x": 1024,
                "workgroup_count_x": 1,
            }
        ]
    elif sampler_method == "temperature_top_k_top_p":
        sampler_id = "temperature_top_k_top_p_sampler"
        sampler_temperature = float(sampling["temperature"])
        sampler_top_k = int(sampling["top_k"])
        sampler_top_p = float(sampling["top_p"])
        sampler_min_p = float(sampling["min_p"])
        sampler_candidate_shader_file = (
            f"temperature_top_k_candidates_repetition_f32_{vocab_size}"
            f"_rp{shader_float_token(sampler_repetition_penalty)}"
            f"_pp{shader_float_token(sampler_presence_penalty)}"
            f"_k{sampler_top_k}_g{sampler_partition_count}"
            f"_l{sampler_candidate_local_size_x}.comp"
            if sampler_uses_token_state
            else f"temperature_top_k_candidates_f32_{vocab_size}"
            f"_k{sampler_top_k}_g{sampler_partition_count}"
            f"_l{sampler_candidate_local_size_x}.comp"
        )
        sampler_shader_file = (
            f"temperature_top_k_top_p_sampler_f32"
            f"_t{shader_float_token(sampler_temperature)}"
            f"_k{sampler_top_k}_p{shader_float_token(sampler_top_p)}"
            f"_m{shader_float_token(sampler_min_p)}"
            f"_g{sampler_partition_count}_l{sampler_merge_local_size_x}.comp"
        )
        sampler_kernels = sampler_state_kernels + [
            {
                "role": "partition_top_k",
                "shader_path": compiled_shader_path(
                    f"shaders/{sampler_candidate_shader_file}"
                ),
                "local_size_x": sampler_candidate_local_size_x,
                "workgroup_count_x": sampler_partition_count,
            },
            {
                "role": "sample_candidates",
                "shader_path": compiled_shader_path(f"shaders/{sampler_shader_file}"),
                "local_size_x": sampler_merge_local_size_x,
                "workgroup_count_x": 1,
            },
        ]
    else:
        raise ModelCompileError(f"unsupported sampling method {sampler_method!r}")

    runtime_sampler_kernels = [
        {
            "role": "runtime_record_current_token",
            "shader_path": compiled_shader_path(
                f"shaders/record_seen_token_{vocab_size}.comp"
            ),
            "local_size_x": 1,
            "workgroup_count_x": 1,
        },
        {
            "role": "runtime_record_token_batch",
            "shader_path": compiled_shader_path(
                f"shaders/record_seen_tokens_batch64_{vocab_size}.comp"
            ),
            "local_size_x": 64,
            "workgroup_count_x": 1,
        },
    ]
    runtime_sampler_kernels.extend(
        [
            {
                "role": "runtime_sample_logits",
                "shader_path": compiled_shader_path(
                    f"shaders/greedy_sampler_runtime_f32_{vocab_size}.comp"
                ),
                "local_size_x": 1024,
                "workgroup_count_x": 1,
            },
            {
                "role": "runtime_partition_top_k",
                "shader_path": compiled_shader_path(
                    f"shaders/temperature_top_k_candidates_runtime_f32_{vocab_size}"
                    f"_kc{sampler_top_k_capacity}_g{sampler_partition_count}"
                    f"_l{sampler_candidate_local_size_x}.comp"
                ),
                "local_size_x": sampler_candidate_local_size_x,
                "workgroup_count_x": sampler_partition_count,
            },
            {
                "role": "runtime_sample_candidates",
                "shader_path": compiled_shader_path(
                    f"shaders/temperature_top_k_top_p_sampler_runtime_f32"
                    f"_kc{sampler_top_k_capacity}_g{sampler_partition_count}"
                    f"_l{sampler_merge_local_size_x}.comp"
                ),
                "local_size_x": sampler_merge_local_size_x,
                "workgroup_count_x": 1,
            },
        ]
    )
    sampler_kernels += runtime_sampler_kernels

    embed_tensor = model_graph["graph"]["input_transducer"]["params"]["weight"][
        "tensor"
    ]
    output_components = model_graph["graph"]["output_transducer"]["components"]
    norm_tensor = next(
        component["params"]["weight"]["tensor"]
        for component in output_components
        if component["type"] == "rms_norm"
    )
    projection_tensor = next(
        component["params"]["weight"]["tensor"]
        for component in output_components
        if component["type"] == "linear_projection"
    )
    projection_scale_tensor = next(
        (
            component["params"].get("weight_scale_inv", {}).get("tensor")
            for component in output_components
            if component["type"] == "linear_projection"
        ),
        None,
    )
    for role, tensor in (
        ("input embedding", embed_tensor),
        ("output normalization", norm_tensor),
    ):
        actual_dtype = tensor_dtype(tensor_index, tensor)
        if actual_dtype != dtype:
            raise ModelCompileError(
                f"{role} tensor {tensor!r} has dtype {actual_dtype}; expected {dtype}"
            )
    projection_dtype = tensor_dtype(tensor_index, projection_tensor)
    embedding_layout = tensor_layout(tensor_index, embed_tensor)
    projection_layout = tensor_layout(tensor_index, projection_tensor)
    if embedding_layout != ROW_MAJOR_LAYOUT or projection_layout != ROW_MAJOR_LAYOUT:
        raise ModelCompileError(
            "compiled input and output transducers require row-major tensors"
        )
    projection_scale_shape = None
    projection_scale_byte_capacity = None
    projection_scale_dtype = None
    projection_block_rows = None
    projection_block_columns = None
    if projection_dtype == "BF16":
        if projection_scale_tensor is not None:
            raise ModelCompileError(
                "BF16 output projection must not bind an FP8 scale tensor"
            )
    elif projection_dtype == "F8_E4M3":
        if not isinstance(projection_scale_tensor, str):
            raise ModelCompileError(
                f"FP8 output projection tensor {projection_tensor!r} has no scale tensor"
            )
        projection_scale_dtype = tensor_dtype(tensor_index, projection_scale_tensor)
        if projection_scale_dtype != "BF16":
            raise ModelCompileError(
                f"FP8 output projection scale tensor {projection_scale_tensor!r} "
                f"has dtype {projection_scale_dtype}; expected BF16"
            )
        if tensor_layout(tensor_index, projection_scale_tensor) != ROW_MAJOR_LAYOUT:
            raise ModelCompileError("FP8 output projection scales must be row-major")
        projection_shape = tensor_shape(tensor_index, projection_tensor)
        projection_scale_shape = tensor_shape(tensor_index, projection_scale_tensor)
        projection_block_rows = FP8_LINEAR_TILE_ROWS[-1]
        projection_block_columns = 128
        expected_scale_shape = [
            (projection_shape[0] + projection_block_rows - 1) // projection_block_rows,
            (projection_shape[1] + projection_block_columns - 1)
            // projection_block_columns,
        ]
        if (
            projection_shape != [vocab_size, hidden_size]
            or projection_shape[1] % projection_block_columns != 0
            or projection_scale_shape != expected_scale_shape
        ):
            raise ModelCompileError(
                f"FP8 output projection tensor {projection_tensor!r} shape "
                f"{projection_shape} and scale shape {projection_scale_shape} "
                f"do not match expected {[vocab_size, hidden_size]} / "
                f"{expected_scale_shape}"
            )
        projection_scale_byte_capacity = tensor_byte_count(
            tensor_index, projection_scale_tensor
        )
    else:
        raise ModelCompileError(
            f"unsupported output projection dtype {projection_dtype!r}"
        )
    embedding_shader_file = (
        f"embedding_lookup_bf16_{vocab_size}x{hidden_size}"
        f"_scale{shader_float_token(embedding_scale)}.comp"
    )
    embedding_batch_shader_file = embedding_shader_file.replace(
        "embedding_lookup_", "embedding_lookup_batch_", 1
    )
    output_scale = 1.0 / logits_scale
    if projection_dtype == "F8_E4M3":
        projection_tile_rows = FP8_OUTPUT_PROJECTION_TILE_ROWS
        projection_shader_file = (
            f"tied_output_projection_fp8_e4m3_b{projection_block_rows}x"
            f"{projection_block_columns}_{vocab_size}x{hidden_size}"
            f"_scale{shader_float_token(output_scale)}_to_f32.comp"
        )
        projection_batch_lane_tile_width = 1
        projection_batch_shader_file = (
            f"tied_output_projection_batch{projection_batch_lane_tile_width}_"
            f"fp8_e4m3_b{projection_block_rows}x{projection_block_columns}_"
            f"{vocab_size}x{hidden_size}_scale{shader_float_token(output_scale)}"
            "_to_f32.comp"
        )
        projection_workgroup_count_x = (
            vocab_size + projection_tile_rows - 1
        ) // projection_tile_rows
        projection_local_size_x = 1024
    else:
        projection_shader_file = (
            f"tied_output_projection_bf16_{vocab_size}x{hidden_size}"
            f"_scale{shader_float_token(output_scale)}_to_f32.comp"
        )
        projection_batch_lane_tile_width = 4
        projection_batch_shader_file = (
            f"tied_output_projection_batch{projection_batch_lane_tile_width}_bf16_"
            f"{vocab_size}x{hidden_size}_scale{shader_float_token(output_scale)}_to_f32.comp"
        )
        # The BF16 projection shader collaboratively computes two vocabulary
        # rows per workgroup.
        projection_workgroup_count_x = (vocab_size + 1) // 2
        projection_local_size_x = 64
    norm_shader_file = rms_norm_shader_file(hidden_size, norm_eps, norm_weight_offset)
    norm_batch_lane_tile_width = projection_batch_lane_tile_width
    norm_batch_shader_file = weight_shared_batch_shader_file(
        norm_shader_file, tile_width=norm_batch_lane_tile_width
    )
    if norm_batch_shader_file is None:
        raise ModelCompileError(
            f"output normalization shader {norm_shader_file!r} has no batch implementation"
        )

    source_circuits = {}
    compiled_circuits = {}
    for circuit_ref in all_lowered_circuit_refs(lowered_index):
        circuit = read_json(lowered_dir / circuit_ref["circuit"])
        source_circuits[circuit_ref["id"]] = circuit
        compiled_circuits[circuit_ref["id"]] = optimize_circuit_for_vulkan(
            circuit,
            can_fuse_linear_split=lambda node, circuit=circuit: (
                can_fuse_bf16_linear_split(circuit, node, tensor_index)
            ),
            can_fuse_parallel_linears=lambda nodes, circuit=circuit: (
                can_fuse_native_parallel_linears(circuit, nodes, tensor_index)
            ),
            can_fuse_parallel_linear_silu_multiply=lambda projection, activation, circuit=circuit: (
                can_fuse_parallel_linear_silu_multiply(
                    circuit, projection, activation, tensor_index
                )
            ),
            can_fuse_parallel_head_norm_rope=lambda branches, circuit=circuit: (
                can_fuse_bf16_parallel_head_norm_rope(circuit, branches, tensor_index)
            ),
            can_fuse_multiply_rolling_depthwise=lambda multiply, rolling, depthwise, circuit=circuit: (
                can_fuse_bf16_multiply_rolling_depthwise(
                    circuit, multiply, rolling, depthwise, tensor_index
                )
            ),
            can_fuse_recurrent_output_gate=lambda recurrent, gate, circuit=circuit: (
                can_fuse_bf16_recurrent_output_gate(
                    circuit, recurrent, gate, tensor_index
                )
            ),
            can_fuse_linear_split_recurrent=lambda projection, recurrent, circuit=circuit: (
                can_fuse_bf16_linear_split_recurrent(
                    circuit, projection, recurrent, tensor_index
                )
            ),
            can_fuse_append_attention=lambda append, attention, circuit=circuit: (
                can_fuse_bf16_append_attention(circuit, append, attention, tensor_index)
            ),
        )
    behavioral_validation = build_behavioral_validation(
        model_graph=model_graph,
        tensor_index=tensor_index,
        lowered_index=lowered_index,
        source_circuits=source_circuits,
        candidate_circuits=compiled_circuits,
    )
    write_json(package_dir / "behavioral_validation.json", behavioral_validation)
    component_executions = component_execution_specs(
        lowered_index=lowered_index,
        compiled_circuits=compiled_circuits,
        tensor_index=tensor_index,
        dimensions=dimensions,
    )
    speculative_decoders = speculative_decoder_specs(
        lowered_index=lowered_index,
        lowered_dir=lowered_dir,
        compiled_circuits=compiled_circuits,
        tensor_index=tensor_index,
        dimensions=dimensions,
        projection_shader_file=projection_shader_file,
        norm_shader_file=norm_shader_file,
        frame_bytes=frame_bytes,
        logits_bytes=logits_bytes,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
    )
    all_component_executions = [
        *component_executions,
        *(
            execution
            for decoder in speculative_decoders
            for execution in decoder["component_executions"]
        ),
    ]
    shader_files = required_shader_files(
        all_component_executions,
        embedding_shader_file=embedding_shader_file,
        embedding_batch_shader_file=embedding_batch_shader_file,
        projection_shader_file=projection_shader_file,
        projection_batch_shader_file=projection_batch_shader_file,
        norm_shader_file=norm_shader_file,
        norm_batch_shader_file=norm_batch_shader_file,
        sampler_shader_files={
            kernel["shader_path"].removeprefix("shaders/").removesuffix(".spv")
            + ".comp"
            for kernel in sampler_kernels
        }
        | {
            decoder["output_transducer"]["norm_shader_path"]
            .removeprefix("shaders/")
            .removesuffix(".spv")
            + ".comp"
            for decoder in speculative_decoders
        }
        | {
            decoder["output_transducer"]["projection_shader_path"]
            .removeprefix("shaders/")
            .removesuffix(".spv")
            + ".comp"
            for decoder in speculative_decoders
        },
    )
    copy_shader_templates(
        shader_source_dir,
        package_dir / "shaders",
        shader_files,
    )
    for execution in all_component_executions:
        for kernel in execution["kernels"]:
            for implementation in kernel["batch_implementations"]:
                implementation["device_requirements"]["vulkan_device_extensions"] = (
                    required_vulkan_device_extensions(
                        package_dir / "shaders",
                        {
                            stage["shader_path"].removeprefix("shaders/")
                            for stage in implementation["stages"]
                        },
                    )
                )
    optional_device_shader_files = {
        stage["shader_path"].removeprefix("shaders/")
        for execution in all_component_executions
        for kernel in execution["kernels"]
        for implementation in kernel["batch_implementations"]
        for stage in implementation["stages"]
    }
    mandatory_shader_files = shader_files - optional_device_shader_files
    required_device_extensions = required_vulkan_device_extensions(
        package_dir / "shaders", mandatory_shader_files
    )
    compile_shader_artifacts(
        package_dir / "shaders",
        progress=lambda current, total, shader_name: emit_compile_event(
            event_sink,
            "ShaderCompilationStarted",
            current=current,
            total=total,
            shader_name=shader_name,
        ),
        cancel_requested=cancel_requested,
    )
    # The bytecode is authoritative. Requirements are discovered from the SPIR-V
    # emitted by the compiler rather than inferred from model or kernel names.
    mandatory_spirv_files = {
        str(Path(shader_file).with_suffix(".spv"))
        for shader_file in mandatory_shader_files
    }
    required_device_features = required_vulkan_features(
        package_dir / "shaders",
        mandatory_spirv_files,
    )
    required_subgroup_operations = required_vulkan_subgroup_operations(
        package_dir / "shaders",
        mandatory_spirv_files,
    )
    for execution in all_component_executions:
        for kernel in execution["kernels"]:
            kernel["shader_path"] = compiled_shader_path(kernel["shader_path"])
            for implementation in kernel["batch_implementations"]:
                implementation_spirv_files = {
                    Path(compiled_shader_path(stage["shader_path"])).name
                    for stage in implementation["stages"]
                }
                implementation["device_requirements"]["vulkan_features"] = (
                    required_vulkan_features(
                        package_dir / "shaders",
                        implementation_spirv_files,
                    )
                )
                implementation["device_requirements"]["subgroup_operations"] = (
                    required_vulkan_subgroup_operations(
                        package_dir / "shaders",
                        implementation_spirv_files,
                    )
                )
                for stage in implementation["stages"]:
                    stage["shader_path"] = compiled_shader_path(stage["shader_path"])
    return {
        "schema": PACKAGE_SCHEMA,
        "package_id": package_id,
        "compiler_fingerprint": package_compiler_fingerprint(shader_source_dir),
        "circuit_graph": package_circuit_graph(
            lowered_index, lowered_dir, compiled_circuits
        ),
        "tensor_index_path": "tensors.json",
        "behavioral_validation_path": "behavioral_validation.json",
        "config_path": CONFIG_PACKAGE_FILE,
        "tokenizer": tokenizer_manifest,
        "activation_element_bytes": dtype_bytes,
        "max_context_activations": max_context_activations,
        "required_vulkan_device_extensions": required_device_extensions,
        "required_vulkan_features": required_device_features,
        "required_vulkan_subgroup_operations": required_subgroup_operations,
        "input_transducer": {
            "spec": {
                "transducer_id": "input_transducer.token_embedding",
                "parameter_tensor": embed_tensor,
                "parameter_dtype": dtype,
                "parameter_shape": tensor_shape(tensor_index, embed_tensor),
                "parameter_byte_capacity": tensor_byte_count(
                    tensor_index, embed_tensor
                ),
                "output_signal_id": "input_frame",
                "output_frame_byte_capacity": frame_bytes,
                "output_frame_word_count": frame_bytes // 4,
                "local_size_x": 256,
            },
            "shader_path": compiled_shader_path(f"shaders/{embedding_shader_file}"),
            "batch_shader_path": compiled_shader_path(
                f"shaders/{embedding_batch_shader_file}"
            ),
        },
        "output_transducer": {
            "spec": {
                "transducer_id": "output_transducer",
                "input_signal_id": "output_frame",
                "node_ids": [
                    "output_norm",
                    "output_projection",
                ],
                "norm_parameter_tensor": norm_tensor,
                "norm_parameter_dtype": dtype,
                "norm_parameter_shape": tensor_shape(tensor_index, norm_tensor),
                "norm_parameter_byte_capacity": tensor_byte_count(
                    tensor_index, norm_tensor
                ),
                "projection_parameter_tensor": projection_tensor,
                "projection_parameter_dtype": projection_dtype,
                "projection_parameter_shape": tensor_shape(
                    tensor_index, projection_tensor
                ),
                "projection_parameter_byte_capacity": tensor_byte_count(
                    tensor_index, projection_tensor
                ),
                "projection_scale_parameter_tensor": projection_scale_tensor,
                "projection_scale_parameter_dtype": projection_scale_dtype,
                "projection_scale_parameter_shape": projection_scale_shape,
                "projection_scale_parameter_byte_capacity": projection_scale_byte_capacity,
                "input_frame_byte_capacity": frame_bytes,
                "normalized_frame_byte_capacity": frame_bytes,
                "logits_byte_capacity": logits_bytes,
                "projection_workgroup_count_x": projection_workgroup_count_x,
                "norm_local_size_x": 64,
                "projection_local_size_x": projection_local_size_x,
            },
            "embedding_norm_shader_path": compiled_shader_path(
                f"shaders/{norm_shader_file}"
            ),
            "embedding_norm_batch_shader_path": compiled_shader_path(
                f"shaders/{norm_batch_shader_file}"
            ),
            "embedding_norm_batch_lane_tile_width": norm_batch_lane_tile_width,
            "projection_shader_path": compiled_shader_path(
                f"shaders/{projection_shader_file}"
            ),
            "projection_batch_shader_path": compiled_shader_path(
                f"shaders/{projection_batch_shader_file}"
            ),
            "projection_batch_lane_tile_width": projection_batch_lane_tile_width,
        },
        "sampler": {
            "spec": {
                "sampler_id": sampler_id,
                "method": sampler_method,
                "temperature": sampler_temperature,
                "top_k": sampler_top_k,
                "top_p": sampler_top_p,
                "min_p": sampler_min_p,
                "presence_penalty": sampler_presence_penalty,
                "repetition_penalty": sampler_repetition_penalty,
                "top_k_capacity": sampler_top_k_capacity,
                "runtime_parameterized": False,
                "logits_byte_capacity": logits_bytes,
                "output_byte_capacity": 16,
                "scratch_byte_capacity": sampler_scratch_byte_capacity,
            },
            "kernels": sampler_kernels,
        },
        "component_executions": component_executions,
        "speculative_decoders": speculative_decoders,
    }


def package_circuit_graph(
    lowered_index: Json,
    lowered_dir: Path,
    compiled_circuits: dict[str, Json],
) -> Json:
    graph = lowered_index["graph"]
    components = []
    for circuit_ref in graph["circuits"]:
        components.append(
            {
                "component_id": circuit_ref["id"],
                "operator_type": circuit_ref["operator_type"],
                "runtime_role": circuit_ref["runtime_role"],
                "implementation": circuit_ref["implementation"],
                "behavioral_role": circuit_ref["behavioral_role"],
                "circuit": deepcopy(compiled_circuits[circuit_ref["id"]]),
                "params": read_json(lowered_dir / circuit_ref["params"]),
                "state": read_json(lowered_dir / circuit_ref["state"]),
            }
        )

    return {
        "topology": graph["topology"],
        "edges": deepcopy(graph["edges"]),
        "boundary": deepcopy(graph["boundary"]),
        "architecture": deepcopy(lowered_index.get("architecture", {})),
        "dimensions": deepcopy(lowered_index.get("dimensions", {})),
        "input_transducer": deepcopy(graph.get("input_transducer", {})),
        "output_transducer": deepcopy(graph.get("output_transducer", {})),
        "components": components,
    }


def component_execution_specs(
    *,
    lowered_index: Json,
    compiled_circuits: dict[str, Json],
    tensor_index: Json,
    dimensions: Json,
) -> list[Json]:
    executions: list[Json] = []

    for circuit_ref in lowered_index["graph"]["circuits"]:
        if circuit_ref["runtime_role"] != "signal_processor":
            continue
        circuit = compiled_circuits[circuit_ref["id"]]
        kernels = []
        for index, node in enumerate(circuit["nodes"]):
            shader_file = shader_file_for_node(
                circuit,
                node,
                tensor_index,
                dimensions,
            )
            kernels.append(
                component_kernel_spec(
                    execution_index=index,
                    node=node,
                    shader_file=shader_file,
                    local_size_x=local_size_x_for_shader_file(shader_file, node),
                    workgroup_count_x=workgroup_count_x_for_node(
                        circuit, node, tensor_index
                    ),
                )
            )
        executions.append(
            {
                "component_id": circuit_ref["id"],
                "operator_type": circuit_ref["operator_type"],
                "implementation": circuit_ref["implementation"],
                "kernels": kernels,
            }
        )

    return executions


def speculative_decoder_specs(
    *,
    lowered_index: Json,
    lowered_dir: Path,
    compiled_circuits: dict[str, Json],
    tensor_index: Json,
    dimensions: Json,
    projection_shader_file: str,
    norm_shader_file: str,
    frame_bytes: int,
    logits_bytes: int,
    vocab_size: int,
    hidden_size: int,
) -> list[Json]:
    decoders = []
    for draft in lowered_index.get("draft_execution_graphs", []):
        circuit_refs = draft["circuits"]
        input_ref = next(
            ref for ref in circuit_refs if ref["runtime_role"] == "draft_input_adapter"
        )
        output_ref = next(
            ref
            for ref in circuit_refs
            if ref["runtime_role"] == "draft_output_transducer"
        )
        executable_refs = [
            ref
            for ref in circuit_refs
            if ref["runtime_role"] in {"draft_input_adapter", "draft_processor"}
        ]
        executions = [
            component_execution_spec(
                circuit_ref=ref,
                circuit=compiled_circuits[ref["id"]],
                tensor_index=tensor_index,
                dimensions=dimensions,
            )
            for ref in executable_refs
        ]
        output_circuit = compiled_circuits[output_ref["id"]]
        output_refs = output_circuit["parameters"]["refs"]
        norm_tensor = output_refs["norm"]["tensor"]
        projection_tensor = output_refs["projection"]["tensor"]
        projection_scale_tensor = output_refs.get("weight_scale_inv", {}).get(
            "tensor"
        )
        projection_dtype = tensor_dtype(tensor_index, projection_tensor)
        if projection_dtype == "F8_E4M3" and not isinstance(
            projection_scale_tensor, str
        ):
            raise ModelCompileError(
                f"FP8 draft output projection tensor {projection_tensor!r} has no scale tensor"
            )
        if projection_dtype == "BF16" and projection_scale_tensor is not None:
            raise ModelCompileError(
                "BF16 draft output projection must not bind an FP8 scale tensor"
            )
        projection_scale_dtype = (
            tensor_dtype(tensor_index, projection_scale_tensor)
            if isinstance(projection_scale_tensor, str)
            else None
        )
        projection_scale_shape = (
            tensor_shape(tensor_index, projection_scale_tensor)
            if isinstance(projection_scale_tensor, str)
            else None
        )
        projection_scale_byte_capacity = (
            tensor_byte_count(tensor_index, projection_scale_tensor)
            if isinstance(projection_scale_tensor, str)
            else None
        )
        decoders.append(
            {
                "id": draft["id"],
                "type": draft["type"],
                "source_prefix": draft["source_prefix"],
                "circuit_graph": package_auxiliary_circuit_graph(
                    draft, lowered_dir, compiled_circuits
                ),
                "input_adapter": {
                    "component_id": input_ref["id"],
                    "token_embedding_signal_id": "token_embedding",
                    "target_hidden_signal_id": "target_hidden",
                    "output_signal_id": "output_frame",
                    "input_frame_byte_capacity": frame_bytes,
                    "target_hidden_byte_capacity": frame_bytes,
                    "output_frame_byte_capacity": frame_bytes,
                },
                "output_transducer": {
                    "component_id": output_ref["id"],
                    "input_signal_id": "output_frame",
                    "hidden_signal_id": "output_hidden",
                    "logits_signal_id": "output_logits",
                    "norm_parameter_tensor": norm_tensor,
                    "norm_parameter_dtype": tensor_dtype(tensor_index, norm_tensor),
                    "norm_parameter_shape": tensor_shape(tensor_index, norm_tensor),
                    "norm_parameter_byte_capacity": tensor_byte_count(
                        tensor_index, norm_tensor
                    ),
                    "projection_parameter_tensor": projection_tensor,
                    "projection_parameter_dtype": projection_dtype,
                    "projection_parameter_shape": tensor_shape(
                        tensor_index, projection_tensor
                    ),
                    "projection_parameter_byte_capacity": tensor_byte_count(
                        tensor_index, projection_tensor
                    ),
                    "projection_scale_parameter_tensor": projection_scale_tensor,
                    "projection_scale_parameter_dtype": projection_scale_dtype,
                    "projection_scale_parameter_shape": projection_scale_shape,
                    "projection_scale_parameter_byte_capacity": projection_scale_byte_capacity,
                    "input_frame_byte_capacity": frame_bytes,
                    "output_hidden_byte_capacity": frame_bytes,
                    "logits_byte_capacity": logits_bytes,
                    "vocabulary_size": vocab_size,
                    "hidden_size": hidden_size,
                    "projection_workgroup_count_x": (vocab_size + 1) // 2,
                    "norm_local_size_x": 64,
                    "projection_local_size_x": 64,
                    "norm_shader_path": compiled_shader_path(
                        f"shaders/{norm_shader_file}"
                    ),
                    "projection_shader_path": compiled_shader_path(
                        f"shaders/{projection_shader_file}"
                    ),
                },
                "component_executions": executions,
                "state_contract": deepcopy(draft["state_contract"]),
                "verification_contract": {
                    "target_execution": "multi_token",
                    "state_updates": "transactional",
                    "acceptance": "longest_matching_prefix",
                },
            }
        )
    return decoders


def component_execution_spec(
    *,
    circuit_ref: Json,
    circuit: Json,
    tensor_index: Json,
    dimensions: Json,
) -> Json:
    kernels = []
    for index, node in enumerate(circuit["nodes"]):
        shader_file = shader_file_for_node(circuit, node, tensor_index, dimensions)
        kernels.append(
            component_kernel_spec(
                execution_index=index,
                node=node,
                shader_file=shader_file,
                local_size_x=local_size_x_for_shader_file(shader_file, node),
                workgroup_count_x=workgroup_count_x_for_node(
                    circuit, node, tensor_index
                ),
            )
        )
    return {
        "component_id": circuit_ref["id"],
        "operator_type": circuit_ref["operator_type"],
        "implementation": circuit_ref["implementation"],
        "kernels": kernels,
    }


def component_kernel_spec(
    *,
    execution_index: int,
    node: Json,
    shader_file: str,
    local_size_x: int,
    workgroup_count_x: int,
) -> Json:
    causal_scan_stages = causal_scan_batch_stages(shader_file, local_size_x)
    direct_frame_parallel_shader_file = (
        None
        if causal_scan_stages is not None
        else frame_parallel_batch_shader_file(shader_file)
    )
    scalar_batch_shader_file = (
        None
        if causal_scan_stages is not None
        or direct_frame_parallel_shader_file is not None
        else weight_shared_batch_shader_file(shader_file)
    )
    cooperative_shader_file = (
        cooperative_bfloat16_batch_shader_file(shader_file)
        if scalar_batch_shader_file is not None
        else None
    )
    frame_parallel_shader_file = direct_frame_parallel_shader_file or (
        frame_parallel_batch_shader_file(scalar_batch_shader_file)
        if scalar_batch_shader_file is not None
        else None
    )
    spec = {
        "execution_index": execution_index,
        "node_id": node["id"],
        "op": node["op"],
        "execution_domain": "decode",
        "shader_path": f"shaders/{shader_file}",
        "local_size_x": local_size_x,
        "workgroup_count_x": workgroup_count_x,
        "batch_mode": (
            "causal_scan"
            if causal_scan_stages
            else "weight_shared"
            if scalar_batch_shader_file or frame_parallel_shader_file
            else "serial_lanes"
        ),
        "batch_implementations": [],
    }
    if causal_scan_stages is not None:
        spec["batch_implementations"].append(
            {
                "execution_domain": "prefill",
                "lane_tile_width": CAUSAL_SCAN_LANE_TILE_WIDTH,
                "exact_primary_equivalence": False,
                "exact_causal_sequence_equivalence": True,
                "device_requirements": {
                    "vulkan_device_extensions": [],
                    "vulkan_features": [],
                    "subgroup_operations": [],
                },
                "stages": causal_scan_stages,
            }
        )
    elif scalar_batch_shader_file is not None or frame_parallel_shader_file is not None:
        if scalar_batch_shader_file is not None and cooperative_shader_file is not None:
            spec["batch_implementations"].append(
                {
                    "execution_domain": "prefill",
                    "lane_tile_width": COOPERATIVE_BATCH_LANE_TILE_WIDTH,
                    "exact_primary_equivalence": False,
                    "exact_causal_sequence_equivalence": False,
                    "device_requirements": {
                        "vulkan_device_extensions": [],
                        "vulkan_features": [],
                        "subgroup_operations": [],
                        "cooperative_bfloat16_shape": COOPERATIVE_BFLOAT16_SHAPE,
                        "subgroup_size": 64,
                    },
                    "stages": [
                        {
                            "shader_path": f"shaders/{cooperative_shader_file}",
                            "local_size_x": 256,
                            "workgroup_count_x": cooperative_bfloat16_workgroup_count_x(
                                shader_file
                            ),
                        }
                    ],
                }
            )
        if frame_parallel_shader_file is not None:
            spec["batch_implementations"].append(
                {
                    "execution_domain": "prefill",
                    "lane_tile_width": 1,
                    "exact_primary_equivalence": True,
                    "exact_causal_sequence_equivalence": True,
                    "device_requirements": {
                        "vulkan_device_extensions": [],
                        "vulkan_features": [],
                        "subgroup_operations": [],
                        "subgroup_size": 64,
                    },
                    "stages": [
                        {
                            "shader_path": f"shaders/{frame_parallel_shader_file}",
                            "local_size_x": local_size_x,
                            "workgroup_count_x": workgroup_count_x,
                        }
                    ],
                }
            )
        for tile_width in (
            EXACT_BATCH_LANE_TILE_WIDTHS if scalar_batch_shader_file is not None else ()
        ):
            exact_shader_file = weight_shared_batch_shader_file(
                shader_file, tile_width=tile_width
            )
            if exact_shader_file is None:
                raise ModelCompileError(
                    f"shader {shader_file!r} lost its exact batch implementation"
                )
            spec["batch_implementations"].append(
                {
                    "execution_domain": "decode_and_prefill",
                    "lane_tile_width": tile_width,
                    "exact_primary_equivalence": True,
                    "exact_causal_sequence_equivalence": True,
                    "device_requirements": {
                        "vulkan_device_extensions": [],
                        "vulkan_features": [],
                        "subgroup_operations": [],
                    },
                    "stages": [
                        {
                            "shader_path": f"shaders/{exact_shader_file}",
                            "local_size_x": local_size_x,
                            "workgroup_count_x": workgroup_count_x,
                        }
                    ],
                }
            )
    return spec


def package_auxiliary_circuit_graph(
    draft: Json,
    lowered_dir: Path,
    compiled_circuits: dict[str, Json],
) -> Json:
    components = []
    for circuit_ref in draft["circuits"]:
        components.append(
            {
                "component_id": circuit_ref["id"],
                "operator_type": circuit_ref["operator_type"],
                "runtime_role": circuit_ref["runtime_role"],
                "implementation": circuit_ref["implementation"],
                "behavioral_role": circuit_ref["behavioral_role"],
                "circuit": deepcopy(compiled_circuits[circuit_ref["id"]]),
                "params": read_json(lowered_dir / circuit_ref["params"]),
                "state": read_json(lowered_dir / circuit_ref["state"]),
            }
        )
    return {
        "topology": draft["topology"],
        "edges": deepcopy(draft["edges"]),
        "boundary": deepcopy(draft["boundary"]),
        "components": components,
    }
