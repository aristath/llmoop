fn validate_component_executions(
    package_id: &str,
    component_executions: &[VulkanResidentComponentExecutionSpec],
) -> Result<(), VulkanResidentTokenModelPackageError> {
    if component_executions.is_empty() {
        return Err(VulkanResidentTokenModelPackageError::new(format!(
            "resident model package {:?} does not declare component executions",
            package_id
        )));
    }
    let mut declared_kernels = BTreeSet::new();
    for component in component_executions {
        if component.kernels.is_empty() {
            return Err(VulkanResidentTokenModelPackageError::new(format!(
                "resident model package {:?} declares component {:?} with no executable kernels",
                package_id, component.component_id
            )));
        }
        for kernel in &component.kernels {
            if !declared_kernels.insert((component.component_id.as_str(), kernel.node_id.as_str())) {
                return Err(VulkanResidentTokenModelPackageError::new(format!(
                    "resident model package {:?} declares duplicate component kernel {}.{}",
                    package_id, component.component_id, kernel.node_id
                )));
            }
            if !kernel.execution_domain.supports_decode() {
                return Err(VulkanResidentTokenModelPackageError::new(format!(
                    "resident model package {:?} declares non-decode primary execution domain {:?} for {}.{}",
                    package_id, kernel.execution_domain, component.component_id, kernel.node_id
                )));
            }
            let implementations_are_valid =
                kernel.batch_implementations.iter().all(|implementation| {
                    let extensions = &implementation.device_requirements.vulkan_device_extensions;
                    let features = &implementation.device_requirements.vulkan_features;
                    let feature_names = features
                        .iter()
                        .map(|feature| feature.label())
                        .collect::<Vec<_>>();
                    let subgroup_operations =
                        &implementation.device_requirements.subgroup_operations;
                    let subgroup_operation_names = subgroup_operations
                        .iter()
                        .map(|operation| operation.label())
                        .collect::<Vec<_>>();
                    implementation.lane_tile_width > 0
                        && !implementation.stages.is_empty()
                        && match kernel.batch_mode {
                            VulkanResidentComponentKernelBatchMode::SerialLanes => false,
                            VulkanResidentComponentKernelBatchMode::WeightShared => true,
                            VulkanResidentComponentKernelBatchMode::CausalScan => {
                                implementation.execution_domain.supports_prefill()
                            }
                        }
                        && implementation.stages.iter().all(|stage| {
                            !stage.shader_path.is_empty()
                                && stage.local_size_x > 0
                                && stage.workgroup_count_x > 0
                        })
                        && extensions.iter().all(|extension| !extension.is_empty())
                        && extensions.windows(2).all(|pair| pair[0] < pair[1])
                        && features.iter().collect::<BTreeSet<_>>().len() == features.len()
                        && feature_names.windows(2).all(|pair| pair[0] < pair[1])
                        && subgroup_operations.iter().collect::<BTreeSet<_>>().len()
                            == subgroup_operations.len()
                        && subgroup_operation_names
                            .windows(2)
                            .all(|pair| pair[0] < pair[1])
                        && implementation
                            .device_requirements
                            .cooperative_bfloat16_shape
                            .is_none_or(|shape| shape.into_iter().all(|dimension| dimension > 0))
                        && implementation
                            .device_requirements
                            .subgroup_size
                            .is_none_or(|subgroup_size| subgroup_size > 0)
                });
            let valid_batch_contract = match kernel.batch_mode {
                VulkanResidentComponentKernelBatchMode::SerialLanes => {
                    kernel.batch_implementations.is_empty()
                }
                VulkanResidentComponentKernelBatchMode::WeightShared
                | VulkanResidentComponentKernelBatchMode::CausalScan => {
                    !kernel.batch_implementations.is_empty() && implementations_are_valid
                }
            };
            if !valid_batch_contract {
                return Err(VulkanResidentTokenModelPackageError::new(format!(
                    "resident model package {:?} declares invalid {:?} execution metadata for {}.{}",
                    package_id, kernel.batch_mode, component.component_id, kernel.node_id
                )));
            }
        }
    }

    Ok(())
}

fn validate_generation_execution_contract(
    manifest: &VulkanResidentModelPackageManifest,
    circuit_graph: &VulkanResidentPackageCircuitGraph,
) -> Result<(), VulkanResidentTokenModelPackageError> {
    let required_device_extensions = manifest
        .required_vulkan_device_extensions
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    if required_device_extensions.len() != manifest.required_vulkan_device_extensions.len()
        || required_device_extensions
            .iter()
            .any(|extension| extension.is_empty())
        || !manifest
            .required_vulkan_device_extensions
            .windows(2)
            .all(|pair| pair[0] < pair[1])
    {
        return Err(VulkanResidentTokenModelPackageError::new(format!(
            "resident model package {:?} has invalid required Vulkan device extensions",
            manifest.package_id
        )));
    }
    let required_feature_names = manifest
        .required_vulkan_features
        .iter()
        .map(|feature| feature.label())
        .collect::<Vec<_>>();
    if manifest
        .required_vulkan_features
        .iter()
        .collect::<BTreeSet<_>>()
        .len()
        != manifest.required_vulkan_features.len()
        || !required_feature_names
            .windows(2)
            .all(|pair| pair[0] < pair[1])
    {
        return Err(VulkanResidentTokenModelPackageError::new(format!(
            "resident model package {:?} has invalid required Vulkan features",
            manifest.package_id
        )));
    }
    let required_subgroup_operation_names = manifest
        .required_vulkan_subgroup_operations
        .iter()
        .map(|operation| operation.label())
        .collect::<Vec<_>>();
    if manifest
        .required_vulkan_subgroup_operations
        .iter()
        .collect::<BTreeSet<_>>()
        .len()
        != manifest.required_vulkan_subgroup_operations.len()
        || !required_subgroup_operation_names
            .windows(2)
            .all(|pair| pair[0] < pair[1])
    {
        return Err(VulkanResidentTokenModelPackageError::new(format!(
            "resident model package {:?} has invalid required Vulkan subgroup operations",
            manifest.package_id
        )));
    }
    let components_with_role = |role: crate::stream_circuit::CircuitRuntimeRole| {
        circuit_graph
            .components
            .iter()
            .filter(|component| component.runtime_role == role)
            .collect::<Vec<_>>()
    };
    let inputs = components_with_role(crate::stream_circuit::CircuitRuntimeRole::InputTransducer);
    let outputs = components_with_role(crate::stream_circuit::CircuitRuntimeRole::OutputTransducer);
    let samplers = components_with_role(crate::stream_circuit::CircuitRuntimeRole::Sampler);
    let processors = components_with_role(crate::stream_circuit::CircuitRuntimeRole::SignalProcessor);
    if inputs.len() != 1 || outputs.len() != 1 || samplers.len() != 1 || processors.is_empty() {
        return Err(VulkanResidentTokenModelPackageError::new(format!(
            "resident model package {:?} generation graph requires one input transducer, one output transducer, one sampler, and at least one signal processor",
            manifest.package_id
        )));
    }
    let input = inputs[0];
    let output = outputs[0];
    let sampler = samplers[0];
    let processor_ids = processors
        .iter()
        .map(|component| component.component_id.as_str())
        .collect::<BTreeSet<_>>();
    let forward = circuit_graph
        .edges
        .iter()
        .filter(|edge| edge.connection.is_forward())
        .collect::<Vec<_>>();
    let input_edges = forward
        .iter()
        .copied()
        .filter(|edge| {
            edge.source.component_id == input.component_id
                && processor_ids.contains(edge.destination.component_id.as_str())
        })
        .collect::<Vec<_>>();
    let output_edges = forward
        .iter()
        .copied()
        .filter(|edge| {
            processor_ids.contains(edge.source.component_id.as_str())
                && edge.destination.component_id == output.component_id
        })
        .collect::<Vec<_>>();
    let sampler_edges = forward
        .iter()
        .copied()
        .filter(|edge| {
            edge.source.component_id == output.component_id
                && edge.destination.component_id == sampler.component_id
        })
        .collect::<Vec<_>>();
    let feedback_edges = circuit_graph
        .edges
        .iter()
        .filter(|edge| {
            !edge.connection.is_forward()
                && edge.source.component_id == sampler.component_id
                && edge.destination.component_id == input.component_id
        })
        .collect::<Vec<_>>();
    if [
        input_edges.len(),
        output_edges.len(),
        sampler_edges.len(),
        feedback_edges.len(),
    ]
    .into_iter()
    .any(|count| count != 1)
    {
        return Err(VulkanResidentTokenModelPackageError::new(format!(
            "resident model package {:?} must wire input transducer -> processors -> output transducer -> sampler with one delayed sampler feedback edge",
            manifest.package_id
        )));
    }

    let input_nodes = &input.circuit.nodes;
    let output_nodes = &output.circuit.nodes;
    let sampler_nodes = &sampler.circuit.nodes;
    if input_nodes.len() != 1
        || input_nodes[0].inputs.len() != 1
        || input_nodes[0].outputs.len() != 1
        || output_nodes.len() != 2
        || output_nodes[0].inputs.len() != 1
        || output_nodes[1].outputs.len() != 1
        || sampler_nodes.len() != 1
        || sampler_nodes[0].inputs.len() != 2
        || sampler_nodes[0].outputs.len() != 1
    {
        return Err(VulkanResidentTokenModelPackageError::new(format!(
            "resident model package {:?} generation system components have invalid node boundaries",
            manifest.package_id
        )));
    }
    let input_token_port = input_nodes[0].inputs[0].as_str();
    let input_frame_port = input_nodes[0].outputs[0].as_str();
    let output_frame_port = output_nodes[0].inputs[0].as_str();
    let output_logits_port = output_nodes[1].outputs[0].as_str();
    let sampler_logits_port = sampler_nodes[0].inputs[0].as_str();
    let sampler_random_port = sampler_nodes[0].inputs[1].as_str();
    let sampler_token_port = sampler_nodes[0].outputs[0].as_str();
    if input_edges[0].source.port_id != input_frame_port
        || output_edges[0].destination.port_id != output_frame_port
        || sampler_edges[0].source.port_id != output_logits_port
        || sampler_edges[0].destination.port_id != sampler_logits_port
        || feedback_edges[0].source.port_id != sampler_token_port
        || feedback_edges[0].destination.port_id != input_token_port
    {
        return Err(VulkanResidentTokenModelPackageError::new(format!(
            "resident model package {:?} generation edges do not match system-component ports",
            manifest.package_id
        )));
    }

    let external_input_endpoints = circuit_graph
        .boundary
        .external_inputs
        .iter()
        .map(|port| {
            (
                port.endpoint.component_id.as_str(),
                port.endpoint.port_id.as_str(),
            )
        })
        .collect::<BTreeSet<_>>();
    let public_output_endpoints = circuit_graph
        .boundary
        .public_outputs
        .iter()
        .map(|port| {
            (
                port.endpoint.component_id.as_str(),
                port.endpoint.port_id.as_str(),
            )
        })
        .collect::<BTreeSet<_>>();
    if circuit_graph.boundary.external_inputs.len() != 2
        || external_input_endpoints
            != BTreeSet::from([
                (input.component_id.as_str(), input_token_port),
                (sampler.component_id.as_str(), sampler_random_port),
            ])
        || circuit_graph.boundary.public_outputs.len() != 1
        || public_output_endpoints
            != BTreeSet::from([(sampler.component_id.as_str(), sampler_token_port)])
    {
        return Err(VulkanResidentTokenModelPackageError::new(format!(
            "resident model package {:?} must expose one input-transducer input, one sampler random seed, and one sampler public output",
            manifest.package_id
        )));
    }

    let input_weight = input
        .params
        .refs
        .get("weight")
        .and_then(|param| param.tensor.as_deref());
    if input_nodes.len() != 1
        || input_nodes[0].op != "embedding_lookup"
        || input_weight != Some(manifest.input_transducer.spec.parameter_tensor.as_str())
        || manifest.input_transducer.spec.output_signal_id != input_edges[0].destination.port_id
        || manifest.input_transducer.shader_path.is_empty()
        || manifest.input_transducer.batch_shader_path.is_empty()
    {
        return Err(VulkanResidentTokenModelPackageError::new(format!(
            "resident model package {:?} input-transducer execution does not match its circuit component",
            manifest.package_id
        )));
    }

    let output_node_ids = output_nodes
        .iter()
        .map(|node| node.id.as_str())
        .collect::<Vec<_>>();
    let output_ops = output_nodes
        .iter()
        .map(|node| node.op.as_str())
        .collect::<Vec<_>>();
    let norm_weight = output
        .params
        .refs
        .get("output_norm.weight")
        .and_then(|param| param.tensor.as_deref());
    let projection_weight = output
        .params
        .refs
        .get("output_projection.weight")
        .and_then(|param| param.tensor.as_deref());
    if output_node_ids
        != manifest
            .output_transducer
            .spec
            .node_ids
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>()
        || output_ops != ["rms_norm", "linear_projection"]
        || norm_weight
            != Some(
                manifest
                    .output_transducer
                    .spec
                    .norm_parameter_tensor
                    .as_str(),
            )
        || projection_weight
            != Some(
                manifest
                    .output_transducer
                    .spec
                    .projection_parameter_tensor
                    .as_str(),
            )
        || manifest.output_transducer.spec.input_signal_id != output_edges[0].source.port_id
        || manifest
            .output_transducer
            .embedding_norm_shader_path
            .is_empty()
        || manifest
            .output_transducer
            .embedding_norm_batch_shader_path
            .is_empty()
        || manifest
            .output_transducer
            .embedding_norm_batch_lane_tile_width
            == 0
        || manifest.output_transducer.projection_shader_path.is_empty()
        || manifest
            .output_transducer
            .projection_batch_shader_path
            .is_empty()
        || manifest.output_transducer.projection_batch_lane_tile_width == 0
    {
        return Err(VulkanResidentTokenModelPackageError::new(format!(
            "resident model package {:?} output-transducer execution does not match its circuit component",
            manifest.package_id
        )));
    }

    let sampler_attrs = sampler_nodes.first().map(|node| &node.attrs);
    let sampler_spec = &manifest.sampler.spec;
    let sampler_matches = sampler_nodes.len() == 1
        && sampler_nodes[0].op == "sample_token"
        && sampler_attrs
            .and_then(|attrs| attrs.get("randomness"))
            .and_then(Value::as_str)
            == Some("seed_and_stream_tick")
        && sampler_attrs
            .and_then(|attrs| attrs.get("method"))
            .and_then(Value::as_str)
            == Some(sampler_spec.method.as_str())
        && sampler_attrs
            .and_then(|attrs| attrs.get("temperature"))
            .and_then(Value::as_f64)
            .map(|value| value as f32)
            == Some(sampler_spec.temperature)
        && sampler_attrs
            .and_then(|attrs| attrs.get("top_k"))
            .and_then(Value::as_u64)
            == Some(u64::from(sampler_spec.top_k))
        && sampler_attrs
            .and_then(|attrs| attrs.get("top_p"))
            .and_then(Value::as_f64)
            .map(|value| value as f32)
            == Some(sampler_spec.top_p)
        && sampler_attrs
            .and_then(|attrs| attrs.get("min_p"))
            .and_then(Value::as_f64)
            .map(|value| value as f32)
            == Some(sampler_spec.min_p)
        && sampler_attrs
            .and_then(|attrs| attrs.get("presence_penalty"))
            .and_then(Value::as_f64)
            .map(|value| value as f32)
            == Some(sampler_spec.presence_penalty)
        && sampler_attrs
            .and_then(|attrs| attrs.get("repetition_penalty"))
            .and_then(Value::as_f64)
            .map(|value| value as f32)
            == Some(sampler_spec.repetition_penalty)
        && !manifest.sampler.kernels.is_empty();
    if !sampler_matches {
        return Err(VulkanResidentTokenModelPackageError::new(format!(
            "resident model package {:?} sampler execution does not match its circuit component",
            manifest.package_id
        )));
    }
    Ok(())
}

fn validate_component_executions_against_graph(
    package_id: &str,
    component_executions: &[VulkanResidentComponentExecutionSpec],
    graph: &ResolvedLoweredExecutionGraph,
) -> Result<(), VulkanResidentTokenModelPackageError> {
    validate_component_executions(package_id, component_executions)?;
    let execution_by_component = component_executions
        .iter()
        .map(|execution| (execution.component_id.as_str(), execution))
        .collect::<BTreeMap<_, _>>();
    let graph_components = graph
        .circuits
        .iter()
        .filter(|artifact| artifact.circuit.runtime_role.is_signal_processor())
        .map(|artifact| artifact.component.id.as_str())
        .collect::<BTreeSet<_>>();
    if execution_by_component.keys().copied().collect::<BTreeSet<_>>() != graph_components {
        return Err(VulkanResidentTokenModelPackageError::new(format!(
            "resident model package {:?} component executions do not match its circuit graph",
            package_id
        )));
    }
    for artifact in graph
        .circuits
        .iter()
        .filter(|artifact| artifact.circuit.runtime_role.is_signal_processor())
    {
        let execution = execution_by_component[artifact.component.id.as_str()];
        if execution.operator_type != artifact.component.operator_type
            || execution.implementation != artifact.component.implementation
        {
            return Err(VulkanResidentTokenModelPackageError::new(format!(
                "resident model package {:?} execution identity for component {} does not match its circuit",
                package_id, artifact.component.id
            )));
        }
        if execution.kernels.len() != artifact.circuit.nodes.len() {
            return Err(VulkanResidentTokenModelPackageError::new(format!(
                "resident model package {:?} component {} execution does not cover every circuit node",
                package_id, artifact.component.id
            )));
        }
        for (expected_index, (kernel, node)) in execution
            .kernels
            .iter()
            .zip(&artifact.circuit.nodes)
            .enumerate()
        {
            let source_node_ids = semantic_source_node_ids(node);
            let semantic_module_ids = artifact
                .circuit
                .semantic_module_tree
                .as_ref()
                .map(|tree| {
                    tree.modules
                        .iter()
                        .filter(|module| {
                            module
                                .source_node_ids
                                .iter()
                                .any(|node_id| source_node_ids.contains(node_id))
                        })
                        .map(|module| module.id.clone())
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            if kernel.execution_index != expected_index
                || kernel.node_id != node.id
                || kernel.op != node.op
                || kernel.source_node_ids != source_node_ids
                || kernel.semantic_module_ids != semantic_module_ids
                || kernel.shader_path.is_empty()
            {
                return Err(VulkanResidentTokenModelPackageError::new(format!(
                    "resident model package {:?} component {} kernel {} does not match its circuit node",
                    package_id, artifact.component.id, expected_index
                )));
            }
        }
    }
    Ok(())
}

fn semantic_source_node_ids(node: &CircuitNode) -> Vec<String> {
    for attr in ["semantic_source_node_ids", "compiled_from"] {
        let source_node_ids = node
            .attrs
            .get(attr)
            .and_then(Value::as_array)
            .map(|sources| {
                sources
                    .iter()
                    .filter_map(Value::as_str)
                    .map(str::to_string)
                    .collect::<Vec<_>>()
            })
            .filter(|sources| !sources.is_empty());
        if let Some(source_node_ids) = source_node_ids {
            return source_node_ids;
        }
    }
    vec![node.id.clone()]
}

fn validate_component_executions_against_mounted_dispatches(
    package_id: &str,
    component_executions: &[VulkanResidentComponentExecutionSpec],
    mounted_bound: &VulkanMountedPlacedBoundDispatchPlan,
) -> Result<(), VulkanResidentTokenModelPackageError> {
    let declared_components = component_executions
        .iter()
        .map(|component| component.component_id.as_str())
        .collect::<BTreeSet<_>>();
    let mounted_components = mounted_bound
        .dispatches
        .iter()
        .map(|dispatch| dispatch.component_id.as_str())
        .collect::<BTreeSet<_>>();

    let missing_components = mounted_components
        .difference(&declared_components)
        .copied()
        .collect::<Vec<_>>();
    if !missing_components.is_empty() {
        return Err(VulkanResidentTokenModelPackageError::new(format!(
            "resident model package {:?} is missing component executions for mounted components: {}",
            package_id,
            missing_components.join(", ")
        )));
    }

    let unknown_components = declared_components
        .difference(&mounted_components)
        .copied()
        .collect::<Vec<_>>();
    if !unknown_components.is_empty() {
        return Err(VulkanResidentTokenModelPackageError::new(format!(
            "resident model package {:?} declares component executions for unknown mounted components: {}",
            package_id,
            unknown_components.join(", ")
        )));
    }

    for component in component_executions {
        let mounted_dispatches = mounted_bound
            .dispatches
            .iter()
            .filter(|dispatch| dispatch.component_id == component.component_id)
            .collect::<Vec<_>>();
        if component.kernels.len() != mounted_dispatches.len() {
            return Err(VulkanResidentTokenModelPackageError::new(format!(
                "resident model package {:?} declares {} kernels for component {}, but mounted dispatch plan has {}",
                package_id,
                component.kernels.len(),
                component.component_id,
                mounted_dispatches.len()
            )));
        }

        for (expected_index, (kernel, dispatch)) in
            component.kernels.iter().zip(mounted_dispatches).enumerate()
        {
            if kernel.execution_index != expected_index {
                return Err(VulkanResidentTokenModelPackageError::new(format!(
                    "resident model package {:?} declares component {} kernel {} with execution_index {}, expected {}",
                    package_id,
                    component.component_id,
                    kernel.node_id,
                    kernel.execution_index,
                    expected_index
                )));
            }
            if kernel.node_id != dispatch.node_id {
                return Err(VulkanResidentTokenModelPackageError::new(format!(
                    "resident model package {:?} declares component {} execution_index {} as node {}, but mounted dispatch plan has node {}",
                    package_id, component.component_id, expected_index, kernel.node_id, dispatch.node_id
                )));
            }
            if kernel.op != dispatch.op {
                return Err(VulkanResidentTokenModelPackageError::new(format!(
                    "resident model package {:?} declares component {} node {} op {}, but mounted dispatch plan has op {}",
                    package_id, component.component_id, kernel.node_id, kernel.op, dispatch.op
                )));
            }
            if kernel.execution_index != dispatch.node_index {
                return Err(VulkanResidentTokenModelPackageError::new(format!(
                    "resident model package {:?} declares component {} node {} execution_index {}, but mounted dispatch plan has node_index {}",
                    package_id,
                    component.component_id,
                    kernel.node_id,
                    kernel.execution_index,
                    dispatch.node_index
                )));
            }
        }
    }

    Ok(())
}

fn validate_component_executions_cover_prepared_dispatches(
    package_id: &str,
    component_executions: &[VulkanResidentComponentExecutionSpec],
    prepared_plan: &VulkanPreparedDispatchPlan,
) -> Result<(), VulkanResidentTokenModelPackageError> {
    let declared_components = component_executions
        .iter()
        .map(|component| component.component_id.as_str())
        .collect::<BTreeSet<_>>();
    let mounted_components = prepared_plan
        .dispatches
        .iter()
        .map(|dispatch| dispatch.component_id.as_str())
        .collect::<BTreeSet<_>>();

    let missing_components = mounted_components
        .difference(&declared_components)
        .copied()
        .collect::<Vec<_>>();
    if !missing_components.is_empty() {
        return Err(VulkanResidentTokenModelPackageError::new(format!(
            "resident model package {:?} is missing component executions for mounted components: {}",
            package_id,
            missing_components.join(", ")
        )));
    }

    for component in component_executions
        .iter()
        .filter(|component| mounted_components.contains(component.component_id.as_str()))
    {
        let mounted_dispatches = prepared_plan
            .dispatches
            .iter()
            .filter(|dispatch| dispatch.component_id == component.component_id)
            .collect::<Vec<_>>();
        if component.kernels.len() != mounted_dispatches.len() {
            return Err(VulkanResidentTokenModelPackageError::new(format!(
                "resident model package {:?} declares {} kernels for mounted component {}, but mounted dispatch plan has {}",
                package_id,
                component.kernels.len(),
                component.component_id,
                mounted_dispatches.len()
            )));
        }

        for (expected_index, (kernel, dispatch)) in
            component.kernels.iter().zip(mounted_dispatches).enumerate()
        {
            if kernel.execution_index != expected_index {
                return Err(VulkanResidentTokenModelPackageError::new(format!(
                    "resident model package {:?} declares component {} kernel {} with execution_index {}, expected {}",
                    package_id,
                    component.component_id,
                    kernel.node_id,
                    kernel.execution_index,
                    expected_index
                )));
            }
            if kernel.node_id != dispatch.node_id {
                return Err(VulkanResidentTokenModelPackageError::new(format!(
                    "resident model package {:?} declares mounted component {} execution_index {} as node {}, but mounted dispatch plan has node {}",
                    package_id, component.component_id, expected_index, kernel.node_id, dispatch.node_id
                )));
            }
            if kernel.op != dispatch.op {
                return Err(VulkanResidentTokenModelPackageError::new(format!(
                    "resident model package {:?} declares mounted component {} node {} op {}, but mounted dispatch plan has op {}",
                    package_id, component.component_id, kernel.node_id, kernel.op, dispatch.op
                )));
            }
            if kernel.execution_index != dispatch.node_index {
                return Err(VulkanResidentTokenModelPackageError::new(format!(
                    "resident model package {:?} declares mounted component {} node {} execution_index {}, but mounted dispatch plan has node_index {}",
                    package_id,
                    component.component_id,
                    kernel.node_id,
                    kernel.execution_index,
                    dispatch.node_index
                )));
            }
        }
    }

    Ok(())
}
