from nerve.model_package_integrity import *
from nerve.model_package_common import *
from nerve.model_package_assets import *
from nerve.model_package_shaders import *
from nerve.model_package_tensors import *

def validate_compiled_circuit_graph(manifest: Json) -> dict[str, Json]:
    graph = manifest.get("circuit_graph")
    if not isinstance(graph, dict) or graph.get("topology") != "explicit_graph":
        raise ModelCompileError(
            "compiled package must contain an explicit circuit graph"
        )
    components = graph.get("components")
    if not isinstance(components, list) or not components:
        raise ModelCompileError("compiled package circuit graph contains no components")

    candidates: dict[str, Json] = {}
    for component in components:
        component_id = component.get("component_id") if isinstance(component, dict) else None
        if not isinstance(component_id, str) or not component_id:
            raise ModelCompileError(
                "compiled package circuit graph contains a component without an id"
            )
        if component_id in candidates:
            raise ModelCompileError(
                f"compiled package circuit graph repeats component {component_id!r}"
            )
        circuit = component.get("circuit")
        if not isinstance(circuit, dict):
            raise ModelCompileError(
                f"compiled package component {component_id!r} has no circuit"
            )
        report = validate_circuit(circuit)
        if not report.ok:
            try:
                report.raise_for_errors()
            except ValueError as error:
                raise ModelCompileError(str(error)) from error
        source = circuit.get("source")
        if not isinstance(source, dict) or source.get("component_id") != component_id:
            raise ModelCompileError(
                f"compiled package component {component_id!r} circuit identity does not match"
            )
        if component.get("operator_type") != source.get("source_operator_type"):
            raise ModelCompileError(
                f"compiled package component {component_id!r} operator identity does not match"
            )
        if component.get("runtime_role") != circuit.get("runtime_role"):
            raise ModelCompileError(
                f"compiled package component {component_id!r} runtime role does not match"
            )
        if component.get("implementation") != circuit.get("implementation"):
            raise ModelCompileError(
                f"compiled package component {component_id!r} implementation does not match"
            )
        if component.get("behavioral_role") != circuit.get("behavioral_role"):
            raise ModelCompileError(
                f"compiled package component {component_id!r} behavioral role does not match"
            )
        params = component.get("params")
        state = component.get("state")
        if (
            not isinstance(params, dict)
            or params.get("schema") != "nerve.circuit_params.v1"
            or params.get("circuit") != circuit.get("id")
            or params.get("layout") != circuit.get("parameters", {}).get("layout")
            or params.get("storage") != circuit.get("parameters", {}).get("storage")
            or params.get("refs") != circuit.get("parameters", {}).get("refs")
        ):
            raise ModelCompileError(
                f"compiled package component {component_id!r} parameter artifact does not match its circuit"
            )
        if (
            not isinstance(state, dict)
            or state.get("schema") != "nerve.circuit_state.v1"
            or state.get("circuit") != circuit.get("id")
            or state.get("state_ports", []) != circuit.get("state_ports", [])
        ):
            raise ModelCompileError(
                f"compiled package component {component_id!r} state artifact does not match its circuit"
            )
        candidates[component_id] = circuit

    compiler_owned_placement = {"device_id", "placement"}.intersection(manifest)
    if compiler_owned_placement:
        raise ModelCompileError(
            "compiled package must not contain runtime placement fields "
            f"{sorted(compiler_owned_placement)}"
        )

    edges = graph.get("edges")
    if not isinstance(edges, list):
        raise ModelCompileError("compiled package circuit graph edges must be a list")
    edge_ids: set[str] = set()
    connected_outputs: set[tuple[str, str]] = set()
    connected_inputs: set[tuple[str, str]] = set()
    forward_inputs: set[tuple[str, str]] = set()
    feedback_inputs: set[tuple[str, str]] = set()
    forward_indegree = {component_id: 0 for component_id in candidates}
    forward_destinations: dict[str, list[str]] = {
        component_id: [] for component_id in candidates
    }
    for edge in edges:
        edge_id = edge.get("id") if isinstance(edge, dict) else None
        source = edge.get("source") if isinstance(edge, dict) else None
        destination = edge.get("destination") if isinstance(edge, dict) else None
        if not isinstance(edge_id, str) or not edge_id or edge_id in edge_ids:
            raise ModelCompileError(
                f"compiled package circuit graph contains invalid edge id {edge_id!r}"
            )
        edge_ids.add(edge_id)
        if not isinstance(source, dict) or not isinstance(destination, dict):
            raise ModelCompileError(
                f"compiled package edge {edge_id!r} has invalid endpoints"
            )
        source_id = source.get("component_id")
        destination_id = destination.get("component_id")
        if source_id not in candidates or destination_id not in candidates:
            raise ModelCompileError(
                f"compiled package edge {edge_id!r} references an unknown component"
            )
        connection = edge.get("connection")
        if not isinstance(connection, dict):
            raise ModelCompileError(
                f"compiled package edge {edge_id!r} has no connection contract"
            )
        connection_kind = connection.get("kind")
        if connection_kind not in {"forward", "temporal_feedback"}:
            raise ModelCompileError(
                f"compiled package edge {edge_id!r} has unsupported connection kind {connection_kind!r}"
            )
        if connection_kind == "temporal_feedback":
            delay = connection.get("delay_activations")
            if not isinstance(delay, int) or isinstance(delay, bool) or delay < 1:
                raise ModelCompileError(
                    f"compiled package temporal feedback edge {edge_id!r} must delay at least one activation"
                )
        if source_id == destination_id and connection_kind == "forward":
            raise ModelCompileError(
                f"compiled package edge {edge_id!r} creates an instantaneous self-loop"
            )
        output = _port_by_id(
            candidates[source_id]["boundary"]["outputs"], source.get("port_id")
        )
        input_port = _port_by_id(
            candidates[destination_id]["boundary"]["inputs"],
            destination.get("port_id"),
        )
        if output is None or input_port is None:
            raise ModelCompileError(
                f"compiled package edge {edge_id!r} references an unknown port"
            )
        if output.get("signal") != input_port.get("signal") or output.get(
            "shape"
        ) != input_port.get("shape"):
            raise ModelCompileError(
                f"compiled package edge {edge_id!r} connects incompatible ports"
            )
        source_endpoint = (source_id, source["port_id"])
        destination_endpoint = (destination_id, destination["port_id"])
        destination_set = (
            forward_inputs if connection_kind == "forward" else feedback_inputs
        )
        if destination_endpoint in destination_set:
            raise ModelCompileError(
                f"compiled package input {destination_id}.{destination['port_id']} has multiple {connection_kind} edges"
            )
        destination_set.add(destination_endpoint)
        connected_outputs.add(source_endpoint)
        connected_inputs.add(destination_endpoint)
        if connection_kind == "forward":
            forward_indegree[destination_id] += 1
            forward_destinations[source_id].append(destination_id)

    remaining = set(candidates)
    while remaining:
        ready = next(
            (
                component_id
                for component_id in candidates
                if component_id in remaining and forward_indegree[component_id] == 0
            ),
            None,
        )
        if ready is None:
            raise ModelCompileError(
                "compiled package circuit graph contains an instantaneous cycle"
            )
        remaining.remove(ready)
        for destination_id in forward_destinations[ready]:
            forward_indegree[destination_id] -= 1

    boundary = graph.get("boundary")
    if not isinstance(boundary, dict):
        raise ModelCompileError("compiled package circuit graph has no boundary")
    external_inputs = _validate_package_graph_boundary_ports(
        boundary.get("external_inputs"),
        candidates,
        kind="external input",
        direction="inputs",
    )
    public_outputs = _validate_package_graph_boundary_ports(
        boundary.get("public_outputs"),
        candidates,
        kind="public output",
        direction="outputs",
    )

    unrouted_inputs = []
    unrouted_outputs = []
    for component_id, circuit in candidates.items():
        unrouted_inputs.extend(
            (component_id, port["id"])
            for port in circuit["boundary"]["inputs"]
            if (component_id, port["id"]) not in connected_inputs
            and (component_id, port["id"]) not in external_inputs
        )
        unrouted_outputs.extend(
            (component_id, port["id"])
            for port in circuit["boundary"]["outputs"]
            if (component_id, port["id"]) not in connected_outputs
            and (component_id, port["id"]) not in public_outputs
        )
    if unrouted_inputs or unrouted_outputs:
        raise ModelCompileError(
            "compiled package circuit graph has unrouted ports; "
            f"inputs={unrouted_inputs}, outputs={unrouted_outputs}"
        )
    return candidates


def _validate_package_graph_boundary_ports(
    ports: Any,
    candidates: dict[str, Json],
    *,
    kind: str,
    direction: str,
) -> set[tuple[str, str]]:
    if not isinstance(ports, list) or not ports:
        raise ModelCompileError(
            f"compiled package circuit graph must declare at least one {kind}"
        )
    ids: set[str] = set()
    endpoints: set[tuple[str, str]] = set()
    for port in ports:
        port_id = port.get("id") if isinstance(port, dict) else None
        endpoint = port.get("endpoint") if isinstance(port, dict) else None
        if not isinstance(port_id, str) or not port_id or port_id in ids:
            raise ModelCompileError(
                f"compiled package circuit graph has invalid or duplicate {kind} id {port_id!r}"
            )
        ids.add(port_id)
        if not isinstance(endpoint, dict):
            raise ModelCompileError(
                f"compiled package circuit graph {kind} {port_id!r} has no endpoint"
            )
        component_id = endpoint.get("component_id")
        endpoint_port_id = endpoint.get("port_id")
        circuit = candidates.get(component_id)
        if (
            circuit is None
            or _port_by_id(circuit["boundary"][direction], endpoint_port_id) is None
        ):
            raise ModelCompileError(
                f"compiled package circuit graph {kind} {port_id!r} references an unknown {direction[:-1]}"
            )
        key = (component_id, endpoint_port_id)
        if key in endpoints:
            raise ModelCompileError(
                f"compiled package circuit graph repeats {kind} endpoint {component_id}.{endpoint_port_id}"
            )
        endpoints.add(key)
    return endpoints


def _port_by_id(ports: list[Json], port_id: Any) -> Json | None:
    return next((port for port in ports if port.get("id") == port_id), None)


def validate_compiled_speculative_decoders(manifest: Json) -> dict[str, Json]:
    raw_decoders = manifest.get("speculative_decoders", [])
    if not isinstance(raw_decoders, list):
        raise ModelCompileError("compiled package speculative decoders must be a list")
    decoder_ids: set[str] = set()
    candidates: dict[str, Json] = {}
    for decoder in raw_decoders:
        decoder_id = decoder.get("id") if isinstance(decoder, dict) else None
        if (
            not isinstance(decoder_id, str)
            or not decoder_id
            or decoder_id in decoder_ids
        ):
            raise ModelCompileError(
                f"compiled package contains invalid or duplicate speculative decoder {decoder_id!r}"
            )
        decoder_ids.add(decoder_id)
        if decoder.get("type") != "multi_token_prediction":
            raise ModelCompileError(
                f"speculative decoder {decoder_id!r} has unsupported type {decoder.get('type')!r}"
            )
        graph = decoder.get("circuit_graph")
        graph_candidates = validate_compiled_circuit_graph({"circuit_graph": graph})
        duplicate = set(candidates).intersection(graph_candidates)
        if duplicate:
            raise ModelCompileError(
                f"speculative decoder {decoder_id!r} repeats component ids {sorted(duplicate)}"
            )
        roles: dict[str, list[str]] = {}
        for component_id, circuit in graph_candidates.items():
            roles.setdefault(str(circuit.get("runtime_role")), []).append(component_id)
        if (
            len(roles.get("draft_input_adapter", [])) != 1
            or not roles.get("draft_processor")
            or len(roles.get("draft_output_transducer", [])) != 1
            or set(roles)
            != {
                "draft_input_adapter",
                "draft_processor",
                "draft_output_transducer",
            }
        ):
            raise ModelCompileError(
                f"speculative decoder {decoder_id!r} must contain one input adapter, "
                "at least one draft processor, and one output transducer"
            )
        execution_by_component = {
            execution.get("component_id"): execution
            for execution in decoder.get("component_executions", [])
            if isinstance(execution, dict)
        }
        executable_ids = set(roles["draft_input_adapter"]) | set(
            roles["draft_processor"]
        )
        if set(execution_by_component) != executable_ids:
            raise ModelCompileError(
                f"speculative decoder {decoder_id!r} executions do not cover its executable components"
            )
        for component_id in executable_ids:
            circuit = graph_candidates[component_id]
            execution = execution_by_component[component_id]
            kernels = execution.get("kernels")
            if not isinstance(kernels, list) or len(kernels) != len(circuit["nodes"]):
                raise ModelCompileError(
                    f"speculative decoder {decoder_id!r} execution for {component_id!r} "
                    "does not cover every circuit node"
                )
            for index, (kernel, node) in enumerate(
                zip(kernels, circuit["nodes"], strict=True)
            ):
                if (
                    kernel.get("execution_index") != index
                    or kernel.get("node_id") != node.get("id")
                    or kernel.get("op") != node.get("op")
                    or not isinstance(kernel.get("shader_path"), str)
                    or not kernel["shader_path"]
                ):
                    raise ModelCompileError(
                        f"speculative decoder {decoder_id!r} kernel {component_id}.{index} "
                        "does not match its circuit node"
                    )
        output_id = roles["draft_output_transducer"][0]
        output_spec = decoder.get("output_transducer")
        output_refs = graph_candidates[output_id]["parameters"]["refs"]
        if (
            not isinstance(output_spec, dict)
            or output_spec.get("component_id") != output_id
            or output_spec.get("norm_parameter_tensor")
            != output_refs.get("norm", {}).get("tensor")
            or output_spec.get("projection_parameter_tensor")
            != output_refs.get("projection", {}).get("tensor")
            or any(
                not isinstance(output_spec.get(field), str) or not output_spec[field]
                for field in ("norm_shader_path", "projection_shader_path")
            )
        ):
            raise ModelCompileError(
                f"speculative decoder {decoder_id!r} output execution does not match its circuit"
            )
        expected_state_contract = {
            "ownership": "per_stream_per_node_instance",
            "draft_updates": "tentative",
            "acceptance": "commit_accepted_prefix",
            "rejection": "restore_last_committed_state",
        }
        if decoder.get("state_contract") != expected_state_contract:
            raise ModelCompileError(
                f"speculative decoder {decoder_id!r} has no transactional state contract"
            )
        candidates.update(graph_candidates)
    return candidates


def validate_compiled_component_executions(
    manifest: Json,
    candidate_circuits: dict[str, Json],
) -> None:
    executions = manifest.get("component_executions")
    if not isinstance(executions, list):
        raise ModelCompileError("compiled package has no component execution list")
    execution_by_component: dict[str, Json] = {}
    for execution in executions:
        component_id = execution.get("component_id") if isinstance(execution, dict) else None
        if (
            not isinstance(component_id, str)
            or not component_id
            or component_id in execution_by_component
        ):
            raise ModelCompileError(
                f"compiled package contains invalid or duplicate component execution {component_id!r}"
            )
        execution_by_component[component_id] = execution
    executable_circuits = {
        component_id: circuit
        for component_id, circuit in candidate_circuits.items()
        if circuit.get("runtime_role") == "signal_processor"
    }
    if set(execution_by_component) != set(executable_circuits):
        raise ModelCompileError(
            "compiled package component executions do not match its signal-processing circuits"
        )
    for component_id, circuit in executable_circuits.items():
        execution = execution_by_component[component_id]
        source = circuit.get("source", {})
        if execution.get("operator_type") != source.get(
            "source_operator_type"
        ) or execution.get("implementation") != circuit.get("implementation"):
            raise ModelCompileError(
                f"compiled package component {component_id!r} execution identity does not match its circuit"
            )
        kernels = execution.get("kernels")
        nodes = circuit.get("nodes", [])
        if not isinstance(kernels, list) or len(kernels) != len(nodes):
            raise ModelCompileError(
                f"compiled package component {component_id!r} execution does not cover every circuit node"
            )
        for index, (kernel, node) in enumerate(zip(kernels, nodes, strict=True)):
            if (
                not isinstance(kernel, dict)
                or kernel.get("execution_index") != index
                or kernel.get("node_id") != node.get("id")
                or kernel.get("op") != node.get("op")
                or kernel.get("execution_domain") not in {
                    "decode",
                    "decode_and_prefill",
                }
                or not isinstance(kernel.get("shader_path"), str)
                or not kernel["shader_path"]
            ):
                raise ModelCompileError(
                    f"compiled package component {component_id!r} kernel {index} does not match its circuit node"
                )
            batch_mode = kernel.get("batch_mode")
            batch_implementations = kernel.get("batch_implementations")
            if batch_mode == "serial_lanes" and batch_implementations == []:
                continue
            if (
                batch_mode in {"weight_shared", "causal_scan"}
                and isinstance(batch_implementations, list)
                and batch_implementations
                and all(
                    valid_batch_implementation(implementation)
                    for implementation in batch_implementations
                )
                and (
                    batch_mode != "causal_scan"
                    or all(
                        implementation["execution_domain"]
                        in {"prefill", "decode_and_prefill"}
                        and implementation["exact_causal_sequence_equivalence"]
                        for implementation in batch_implementations
                    )
                )
            ):
                continue
            raise ModelCompileError(
                f"compiled package component {component_id!r} kernel {index} has an invalid batch execution contract"
            )


def valid_batch_implementation(implementation: Any) -> bool:
    if not isinstance(implementation, dict):
        return False
    execution_domain = implementation.get("execution_domain")
    requirements = implementation.get("device_requirements")
    extensions = (
        requirements.get("vulkan_device_extensions")
        if isinstance(requirements, dict)
        else None
    )
    features = requirements.get("vulkan_features") if requirements else None
    subgroup_operations = (
        requirements.get("subgroup_operations") if requirements else None
    )
    shape = requirements.get("cooperative_bfloat16_shape") if requirements else None
    subgroup_size = requirements.get("subgroup_size") if requirements else None
    stages = implementation.get("stages")
    return (
        execution_domain in KNOWN_COMPONENT_KERNEL_EXECUTION_DOMAINS
        and
        isinstance(implementation.get("lane_tile_width"), int)
        and not isinstance(implementation.get("lane_tile_width"), bool)
        and implementation["lane_tile_width"] > 0
        and isinstance(implementation.get("exact_primary_equivalence"), bool)
        and isinstance(implementation.get("exact_causal_sequence_equivalence"), bool)
        and isinstance(stages, list)
        and bool(stages)
        and all(valid_batch_stage(stage) for stage in stages)
        and isinstance(extensions, list)
        and all(isinstance(extension, str) and extension for extension in extensions)
        and extensions == sorted(set(extensions))
        and isinstance(features, list)
        and all(feature in KNOWN_VULKAN_FEATURES for feature in features)
        and features == sorted(set(features))
        and isinstance(subgroup_operations, list)
        and all(
            operation in KNOWN_VULKAN_SUBGROUP_OPERATIONS
            for operation in subgroup_operations
        )
        and subgroup_operations == sorted(set(subgroup_operations))
        and (
            shape is None
            or (
                isinstance(shape, list)
                and len(shape) == 3
                and all(
                    isinstance(dimension, int)
                    and not isinstance(dimension, bool)
                    and dimension > 0
                    for dimension in shape
                )
            )
        )
        and (
            subgroup_size is None
            or (
                isinstance(subgroup_size, int)
                and not isinstance(subgroup_size, bool)
                and subgroup_size > 0
            )
        )
    )


def valid_batch_stage(stage: Any) -> bool:
    return (
        isinstance(stage, dict)
        and isinstance(stage.get("shader_path"), str)
        and bool(stage["shader_path"])
        and isinstance(stage.get("local_size_x"), int)
        and not isinstance(stage.get("local_size_x"), bool)
        and stage["local_size_x"] > 0
        and isinstance(stage.get("workgroup_count_x"), int)
        and not isinstance(stage.get("workgroup_count_x"), bool)
        and stage["workgroup_count_x"] > 0
    )


def validate_compiled_generation_contract(
    manifest: Json,
    candidate_circuits: dict[str, Json],
) -> None:
    role_ids: dict[str, list[str]] = {}
    for component_id, circuit in candidate_circuits.items():
        role_ids.setdefault(str(circuit.get("runtime_role")), []).append(component_id)
    for role in ("input_transducer", "output_transducer", "sampler"):
        if len(role_ids.get(role, [])) != 1:
            raise ModelCompileError(
                f"compiled generation graph must contain exactly one {role} component"
            )
    if not role_ids.get("signal_processor"):
        raise ModelCompileError(
            "compiled generation graph must contain at least one signal processor"
        )

    input_id = role_ids["input_transducer"][0]
    output_id = role_ids["output_transducer"][0]
    sampler_id = role_ids["sampler"][0]
    processor_ids = set(role_ids["signal_processor"])
    graph = manifest["circuit_graph"]
    forward = [
        edge for edge in graph["edges"] if edge["connection"]["kind"] == "forward"
    ]
    feedback = [
        edge
        for edge in graph["edges"]
        if edge["connection"]["kind"] == "temporal_feedback"
    ]
    input_edges = [
        edge
        for edge in forward
        if edge["source"]["component_id"] == input_id
        and edge["destination"]["component_id"] in processor_ids
    ]
    output_edges = [
        edge
        for edge in forward
        if edge["source"]["component_id"] in processor_ids
        and edge["destination"]["component_id"] == output_id
    ]
    sampler_edges = [
        edge
        for edge in forward
        if edge["source"]["component_id"] == output_id
        and edge["destination"]["component_id"] == sampler_id
    ]
    generation_feedback = [
        edge
        for edge in feedback
        if edge["source"]["component_id"] == sampler_id
        and edge["destination"]["component_id"] == input_id
    ]
    if any(
        len(edges) != 1
        for edges in (
            input_edges,
            output_edges,
            sampler_edges,
            generation_feedback,
        )
    ):
        raise ModelCompileError(
            "compiled generation graph must wire input transducer -> processors -> "
            "output transducer -> sampler with one delayed sampler feedback edge"
        )

    input_circuit = candidate_circuits[input_id]
    output_circuit = candidate_circuits[output_id]
    sampler_circuit = candidate_circuits[sampler_id]
    input_nodes = input_circuit.get("nodes", [])
    output_nodes = output_circuit.get("nodes", [])
    sampler_nodes = sampler_circuit.get("nodes", [])
    if (
        len(input_nodes) != 1
        or len(input_nodes[0].get("inputs", [])) != 1
        or len(input_nodes[0].get("outputs", [])) != 1
        or len(output_nodes) != 2
        or len(output_nodes[0].get("inputs", [])) != 1
        or len(output_nodes[-1].get("outputs", [])) != 1
        or len(sampler_nodes) != 1
        or len(sampler_nodes[0].get("inputs", [])) != 2
        or len(sampler_nodes[0].get("outputs", [])) != 1
    ):
        raise ModelCompileError(
            "compiled generation system components have invalid node boundaries"
        )
    input_token_port = input_nodes[0]["inputs"][0]
    input_frame_port = input_nodes[0]["outputs"][0]
    output_frame_port = output_nodes[0]["inputs"][0]
    output_logits_port = output_nodes[-1]["outputs"][0]
    sampler_logits_port, sampler_random_port = sampler_nodes[0]["inputs"]
    sampler_token_port = sampler_nodes[0]["outputs"][0]
    if (
        input_edges[0]["source"]["port_id"] != input_frame_port
        or output_edges[0]["destination"]["port_id"] != output_frame_port
        or sampler_edges[0]["source"]["port_id"] != output_logits_port
        or sampler_edges[0]["destination"]["port_id"] != sampler_logits_port
        or generation_feedback[0]["source"]["port_id"] != sampler_token_port
        or generation_feedback[0]["destination"]["port_id"] != input_token_port
    ):
        raise ModelCompileError(
            "compiled generation graph edges do not match system-component ports"
        )

    boundary = graph["boundary"]
    external_endpoints = {
        (port["endpoint"]["component_id"], port["endpoint"]["port_id"])
        for port in boundary["external_inputs"]
    }
    public_endpoints = {
        (port["endpoint"]["component_id"], port["endpoint"]["port_id"])
        for port in boundary["public_outputs"]
    }
    if (
        len(boundary["external_inputs"]) != 2
        or external_endpoints
        != {(input_id, input_token_port), (sampler_id, sampler_random_port)}
        or len(boundary["public_outputs"]) != 1
        or public_endpoints != {(sampler_id, sampler_token_port)}
    ):
        raise ModelCompileError(
            "compiled generation graph boundaries must expose one user input, one "
            "sampler random seed, and one sampler public output"
        )

    input_package = manifest.get("input_transducer")
    output_package = manifest.get("output_transducer")
    sampler_package = manifest.get("sampler")
    if not all(
        isinstance(value, dict)
        for value in (input_package, output_package, sampler_package)
    ):
        raise ModelCompileError(
            "compiled generation package is missing a system-component execution spec"
        )

    input_spec = input_package.get("spec")
    input_refs = input_circuit.get("parameters", {}).get("refs", {})
    if (
        not isinstance(input_spec, dict)
        or len(input_nodes) != 1
        or input_nodes[0].get("op") != "embedding_lookup"
        or input_spec.get("parameter_tensor")
        != input_refs.get("weight", {}).get("tensor")
        or input_spec.get("output_signal_id")
        != input_edges[0]["destination"]["port_id"]
        or not isinstance(input_package.get("shader_path"), str)
        or not input_package["shader_path"]
        or not isinstance(input_package.get("batch_shader_path"), str)
        or not input_package["batch_shader_path"]
    ):
        raise ModelCompileError(
            "compiled input-transducer execution does not match its circuit component"
        )

    output_spec = output_package.get("spec")
    output_refs = output_circuit.get("parameters", {}).get("refs", {})
    if (
        not isinstance(output_spec, dict)
        or [node.get("id") for node in output_nodes] != output_spec.get("node_ids")
        or [node.get("op") for node in output_nodes]
        != ["rms_norm", "linear_projection"]
        or output_spec.get("norm_parameter_tensor")
        != output_refs.get("output_norm.weight", {}).get("tensor")
        or output_spec.get("projection_parameter_tensor")
        != output_refs.get("output_projection.weight", {}).get("tensor")
        or output_spec.get("input_signal_id") != output_edges[0]["source"]["port_id"]
        or any(
            not isinstance(output_package.get(field), str) or not output_package[field]
            for field in (
                "embedding_norm_shader_path",
                "embedding_norm_batch_shader_path",
                "projection_shader_path",
                "projection_batch_shader_path",
            )
        )
        or not isinstance(
            output_package.get("embedding_norm_batch_lane_tile_width"), int
        )
        or isinstance(output_package.get("embedding_norm_batch_lane_tile_width"), bool)
        or output_package["embedding_norm_batch_lane_tile_width"] <= 0
        or not isinstance(output_package.get("projection_batch_lane_tile_width"), int)
        or isinstance(output_package.get("projection_batch_lane_tile_width"), bool)
        or output_package["projection_batch_lane_tile_width"] <= 0
    ):
        raise ModelCompileError(
            "compiled output-transducer execution does not match its circuit component"
        )

    sampler_spec = sampler_package.get("spec")
    sampler_attrs = sampler_nodes[0].get("attrs", {}) if len(sampler_nodes) == 1 else {}
    if (
        not isinstance(sampler_spec, dict)
        or len(sampler_nodes) != 1
        or sampler_nodes[0].get("op") != "sample_token"
        or sampler_attrs.get("randomness") != "seed_and_stream_tick"
        or any(
            sampler_spec.get(field) != sampler_attrs.get(field)
            for field in (
                "method",
                "temperature",
                "top_k",
                "top_p",
                "min_p",
                "presence_penalty",
                "repetition_penalty",
            )
        )
        or not isinstance(sampler_package.get("kernels"), list)
        or not sampler_package["kernels"]
    ):
        raise ModelCompileError(
            "compiled sampler execution does not match its circuit component"
        )



def validate_compiled_package(package_dir: Path, manifest: Json) -> None:
    if manifest.get("schema") != PACKAGE_SCHEMA:
        raise ModelCompileError(
            f"compiled package has unsupported schema {manifest.get('schema')!r}"
        )
    if not isinstance(manifest.get("package_id"), str) or not manifest["package_id"]:
        raise ModelCompileError("compiled package has no package id")
    compiler_fingerprint = manifest.get("compiler_fingerprint")
    if (
        not isinstance(compiler_fingerprint, str)
        or re.fullmatch(
            rf"{re.escape(COMPILER_FINGERPRINT_SCHEMA)}:[0-9a-f]{{64}}",
            compiler_fingerprint,
        )
        is None
    ):
        raise ModelCompileError(
            "compiled package has no valid compiler fingerprint; recompile the model"
        )
    if (
        not isinstance(manifest.get("max_context_activations"), int)
        or isinstance(manifest.get("max_context_activations"), bool)
        or manifest["max_context_activations"] <= 0
    ):
        raise ModelCompileError(
            "compiled package max context activation capacity must be positive"
        )
    required_device_extensions = manifest.get("required_vulkan_device_extensions")
    if (
        not isinstance(required_device_extensions, list)
        or any(
            not isinstance(extension, str) or not extension
            for extension in required_device_extensions
        )
        or len(required_device_extensions) != len(set(required_device_extensions))
        or required_device_extensions != sorted(required_device_extensions)
    ):
        raise ModelCompileError(
            "compiled package required Vulkan device extensions must be unique sorted names"
        )
    required_features = manifest.get("required_vulkan_features")
    if (
        not isinstance(required_features, list)
        or any(feature not in KNOWN_VULKAN_FEATURES for feature in required_features)
        or required_features != sorted(set(required_features))
    ):
        raise ModelCompileError(
            "compiled package required Vulkan features must be unique sorted known names"
        )
    required_subgroup_operations = manifest.get("required_vulkan_subgroup_operations")
    if (
        not isinstance(required_subgroup_operations, list)
        or any(
            operation not in KNOWN_VULKAN_SUBGROUP_OPERATIONS
            for operation in required_subgroup_operations
        )
        or required_subgroup_operations != sorted(set(required_subgroup_operations))
    ):
        raise ModelCompileError(
            "compiled package required Vulkan subgroup operations must be unique sorted known names"
        )
    required_files = (
        package_artifact_path(package_dir, manifest.get("config_path"), "config"),
        package_artifact_path(
            package_dir, manifest.get("tensor_index_path"), "tensor index"
        ),
    )
    for path in required_files:
        if not path.is_file():
            raise ModelCompileError(
                f"compiled package is missing required artifact {path}"
            )

    behavioral_path = package_artifact_path(
        package_dir,
        manifest.get("behavioral_validation_path"),
        "behavioral validation",
    )
    if not behavioral_path.is_file():
        raise ModelCompileError(
            f"compiled package is missing behavioral validation artifact {behavioral_path}"
        )
    behavioral = read_json(behavioral_path)
    candidate_circuits = validate_compiled_circuit_graph(manifest)
    auxiliary_circuits = validate_compiled_speculative_decoders(manifest)
    duplicate_circuits = set(candidate_circuits).intersection(auxiliary_circuits)
    if duplicate_circuits:
        raise ModelCompileError(
            "compiled package repeats circuit ids across target and draft graphs: "
            f"{sorted(duplicate_circuits)}"
        )
    all_candidate_circuits = {**candidate_circuits, **auxiliary_circuits}
    validate_compiled_generation_contract(manifest, candidate_circuits)
    validate_behavioral_validation_artifact(behavioral, all_candidate_circuits)
    validate_compiled_component_executions(manifest, candidate_circuits)

    tokenizer = manifest.get("tokenizer")
    if not isinstance(tokenizer, dict) or not tokenizer.get("path"):
        raise ModelCompileError("compiled package does not declare tokenizer artifacts")
    tokenizer_dir = package_artifact_path(
        package_dir, tokenizer["path"], "tokenizer directory"
    )
    tokenizer_files = tokenizer.get("files")
    if (
        not isinstance(tokenizer_files, list)
        or not tokenizer_files
        or any(
            not isinstance(filename, str) or not filename
            for filename in tokenizer_files
        )
    ):
        raise ModelCompileError(
            "compiled package tokenizer must declare at least one artifact"
        )
    for filename in tokenizer_files:
        path = package_artifact_path(tokenizer_dir, filename, "tokenizer artifact")
        if not path.is_file():
            raise ModelCompileError(
                f"compiled package is missing tokenizer artifact {path}"
            )

    tensor_index = read_json(required_files[1])
    if tensor_index.get("schema") != "nerve.tensor_index.v1":
        raise ModelCompileError("compiled package tensor index schema is invalid")
    tensors = tensor_index.get("tensors")
    if not isinstance(tensors, dict) or not tensors:
        raise ModelCompileError("compiled package tensor index contains no tensors")
    for tensor_name, info in tensors.items():
        if (
            not isinstance(tensor_name, str)
            or not tensor_name
            or not isinstance(info, dict)
        ):
            raise ModelCompileError(
                "compiled package tensor index contains an invalid tensor"
            )
        source = package_artifact_path(
            package_dir,
            info.get("source_file"),
            f"tensor {tensor_name!r} source",
        )
        if not source.is_file():
            raise ModelCompileError(
                f"compiled tensor {tensor_name!r} references missing artifact {source}"
            )
        data_digest = info.get("data_sha256")
        if (
            not isinstance(data_digest, str)
            or len(data_digest) != 64
            or any(character not in "0123456789abcdef" for character in data_digest)
        ):
            raise ModelCompileError(
                f"compiled tensor {tensor_name!r} has no valid data SHA-256"
            )

    shader_paths: set[str] = set()

    def collect_shader_paths(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key.endswith("shader_path") and isinstance(child, str):
                    shader_paths.add(child)
                else:
                    collect_shader_paths(child)
        elif isinstance(value, list):
            for child in value:
                collect_shader_paths(child)

    collect_shader_paths(manifest)
    if not shader_paths:
        raise ModelCompileError(
            "compiled package does not reference any shader artifacts"
        )
    for relative_path in sorted(shader_paths):
        shader = package_artifact_path(package_dir, relative_path, "shader")
        if not shader.is_file():
            raise ModelCompileError(
                f"compiled package references missing shader {shader}"
            )
        payload = shader.read_bytes()
        if len(payload) < 4 or payload[:4] != b"\x03\x02#\x07":
            raise ModelCompileError(
                f"compiled package shader is not valid SPIR-V: {shader}"
            )
    validate_package_artifact_integrity(package_dir, manifest)
    validate_compiled_spirv_requirements(package_dir, manifest)


def validate_compiled_spirv_requirements(package_dir: Path, manifest: Json) -> None:
    speculative_decoders = manifest.get("speculative_decoders", [])
    executions = list(manifest["component_executions"])
    executions.extend(
        execution
        for decoder in speculative_decoders
        for execution in decoder["component_executions"]
    )
    mandatory_shader_paths = {
        manifest["input_transducer"]["shader_path"],
        manifest["input_transducer"]["batch_shader_path"],
        manifest["output_transducer"]["embedding_norm_shader_path"],
        manifest["output_transducer"]["embedding_norm_batch_shader_path"],
        manifest["output_transducer"]["projection_shader_path"],
        manifest["output_transducer"]["projection_batch_shader_path"],
        *(kernel["shader_path"] for kernel in manifest["sampler"]["kernels"]),
        *(
            decoder["output_transducer"]["norm_shader_path"]
            for decoder in speculative_decoders
        ),
        *(
            decoder["output_transducer"]["projection_shader_path"]
            for decoder in speculative_decoders
        ),
    }
    for execution in executions:
        for kernel in execution["kernels"]:
            mandatory_shader_paths.add(kernel["shader_path"])
            for implementation in kernel["batch_implementations"]:
                actual_features, actual_subgroup_operations = spirv_vulkan_requirements(
                    package_dir,
                    {stage["shader_path"] for stage in implementation["stages"]},
                )
                requirements = implementation["device_requirements"]
                if (
                    requirements["vulkan_features"] != actual_features
                    or requirements["subgroup_operations"] != actual_subgroup_operations
                ):
                    raise ModelCompileError(
                        "compiled batch implementation "
                        f"{execution['component_id']}.{kernel['node_id']} does not declare "
                        "the Vulkan requirements of its SPIR-V artifacts"
                    )

    actual_features, actual_subgroup_operations = spirv_vulkan_requirements(
        package_dir, mandatory_shader_paths
    )
    if (
        manifest["required_vulkan_features"] != actual_features
        or manifest["required_vulkan_subgroup_operations"] != actual_subgroup_operations
    ):
        raise ModelCompileError(
            "compiled package does not declare the Vulkan requirements of its "
            "mandatory SPIR-V artifacts"
        )


def validate_package_artifact_integrity(package_dir: Path, manifest: Json) -> None:
    integrity = manifest.get("artifact_integrity")
    if (
        not isinstance(integrity, dict)
        or integrity.get("schema") != PACKAGE_ARTIFACT_INTEGRITY_SCHEMA
        or integrity.get("algorithm") != "sha256"
        or not isinstance(integrity.get("files"), dict)
        or not integrity["files"]
    ):
        raise ModelCompileError(
            "compiled package artifact integrity contract is invalid"
        )

    actual_files = {
        path.relative_to(package_dir).as_posix()
        for path in package_dir.rglob("*")
        if path.is_file()
        and path.relative_to(package_dir).parts[0] != WEIGHTS_PACKAGE_DIR
        and path.name != "vulkan_resident_package.json"
    }
    if set(integrity["files"]) != actual_files:
        raise ModelCompileError(
            "compiled package artifact integrity contract does not cover every non-weight artifact"
        )
    for relative_path, contract in integrity["files"].items():
        path = package_artifact_path(package_dir, relative_path, "integrity artifact")
        if (
            not isinstance(contract, dict)
            or not isinstance(contract.get("byte_count"), int)
            or isinstance(contract.get("byte_count"), bool)
            or contract["byte_count"] < 0
            or not isinstance(contract.get("sha256"), str)
            or len(contract["sha256"]) != 64
            or any(
                character not in "0123456789abcdef" for character in contract["sha256"]
            )
        ):
            raise ModelCompileError(
                f"compiled package artifact integrity entry for {relative_path!r} is invalid"
            )
        payload = path.read_bytes()
        if (
            len(payload) != contract["byte_count"]
            or sha256(payload).hexdigest() != contract["sha256"]
        ):
            raise ModelCompileError(
                f"compiled package artifact {relative_path!r} does not match its integrity contract"
            )


def package_artifact_path(package_dir: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ModelCompileError(f"compiled package has no {label} path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ModelCompileError(
            f"compiled package {label} path must stay inside the package: {value!r}"
        )
    return package_dir / relative


