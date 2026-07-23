fn inspect_runtime_topology(
    args: &Args,
    package_manifest: &Path,
    manifest_dir: &Path,
    manifest: VulkanResidentModelPackageManifest,
) -> Result<(), Box<dyn Error>> {
    let default_device_id = args
        .default_device_id
        .as_deref()
        .unwrap_or(RUNTIME_DEFAULT_LOGICAL_DEVICE_ID);
    let available_devices = inspect_available_devices(
        default_device_id,
        runtime_report_default_vulkan_physical_device_index(args),
    );
    let source_graph = manifest.resolved_source_graph(manifest_dir.to_path_buf())?;
    let runtime_graph = manifest.runtime_graph_from_controls(
        args.default_device_id.as_deref(),
        &args.node_devices,
        &args.duplicate_after,
        args.source_chain.as_deref(),
    )?;
    let effective_graph = source_graph.instantiate_runtime_graph(&runtime_graph)?;
    let placement = effective_graph.placement_plan(&runtime_graph.placement_spec())?;
    let placement_device_ids = placement_device_ids(&placement.components);
    let runtime_routes = runtime_edge_routes_report(args, &placement.edges);
    let device_bindings = runtime_device_bindings_report(args, &placement_device_ids);
    let source_components = source_components_report(&manifest);
    let payload = RuntimeTopologyReport {
        ok: true,
        schema: RUNTIME_TOPOLOGY_SCHEMA.to_string(),
        package_manifest: package_manifest.to_path_buf(),
        package_root: manifest_dir.to_path_buf(),
        package_id: manifest.package_id.clone(),
        compiled_schema: manifest.schema.clone(),
        config_path: manifest.config_path.clone(),
        tokenizer: serde_json::to_value(&manifest.tokenizer)?,
        available_devices,
        compiled: RuntimeCompiledExecutionGraphSummary {
            topology: manifest.circuit_graph.topology.clone(),
            source_component_count: source_components.len(),
            source_components,
            max_context_activations: manifest.max_context_activations,
        },
        runtime_graph_controls: runtime_graph_report(args),
        runtime_graph: runtime_graph,
        effective: RuntimeEffectiveExecutionGraphTopology {
            topology: placement.topology,
            component_count: placement.components.len(),
            edge_count: placement.edges.len(),
            local_edge_count: placement.local_edge_count,
            cross_device_edge_count: placement.cross_device_edge_count,
            device_count: placement_device_ids.len(),
            device_ids: placement_device_ids,
            device_bindings,
            edge_routes: runtime_routes,
            components: placement.components,
            edges: placement.edges,
        },
    };

    if args.json {
        println!("{}", serde_json::to_string_pretty(&payload)?);
    } else {
        println!("package_id={}", payload.package_id);
        println!("source_component_count={}", payload.compiled.source_component_count);
        println!("effective_node_count={}", payload.effective.component_count);
        println!("device_count={}", payload.effective.device_count);
        println!(
            "cross_device_edge_count={}",
            payload.effective.cross_device_edge_count
        );
        println!(
            "same_physical_target_edge_count={}",
            payload
                .effective
                .edge_routes
                .same_physical_target_edge_count
        );
        println!(
            "cross_physical_target_edge_count={}",
            payload
                .effective
                .edge_routes
                .cross_physical_target_edge_count
        );
    }

    Ok(())
}

fn inspect_package(
    args: &Args,
    package_manifest: &Path,
    manifest_dir: &Path,
    manifest: VulkanResidentModelPackageManifest,
) -> Result<(), Box<dyn Error>> {
    let default_device_id = args
        .default_device_id
        .as_deref()
        .unwrap_or(RUNTIME_DEFAULT_LOGICAL_DEVICE_ID);
    let available_devices = inspect_available_devices(
        default_device_id,
        runtime_report_default_vulkan_physical_device_index(args),
    );
    let source_components = source_components_report(&manifest);
    let source_component_count = source_components.len();
    let payload = RuntimePackageInspectionReport {
        ok: true,
        package_manifest: package_manifest.to_path_buf(),
        package_root: manifest_dir.to_path_buf(),
        schema: manifest.schema.clone(),
        package_id: manifest.package_id.clone(),
        config_path: manifest.config_path.clone(),
        tokenizer: serde_json::to_value(&manifest.tokenizer)?,
        compiled_topology: manifest.circuit_graph.topology.clone(),
        runtime_graph: runtime_graph_report(args),
        device_bindings: runtime_device_bindings_report(args, &[]),
        max_context_activations: manifest.max_context_activations,
        source_component_count,
        source_components,
        available_devices,
    };

    if args.json {
        println!("{}", serde_json::to_string_pretty(&payload)?);
    } else {
        println!("package_id={}", payload.package_id);
        println!("source_component_count={}", payload.source_component_count);
        println!("compiled_topology={}", payload.compiled_topology);
        for component in &payload.source_components {
            println!(
                "{} {} kernels={} state_ports={}",
                component.component_id, component.operator_type, component.kernel_count, component.state_port_count
            );
        }
    }

    Ok(())
}

fn source_components_report(manifest: &VulkanResidentModelPackageManifest) -> Vec<RuntimeSourceComponent> {
    let execution_by_component = manifest
        .component_executions
        .iter()
        .map(|execution| (execution.component_id.as_str(), execution))
        .collect::<BTreeMap<_, _>>();

    manifest
        .circuit_graph
        .components
        .iter()
        .enumerate()
        .map(|(component_index, component)| {
            let execution = execution_by_component.get(component.component_id.as_str());
            RuntimeSourceComponent {
                component_index,
                component_id: component.component_id.clone(),
                operator_type: component.operator_type.clone(),
                runtime_role: component.circuit.runtime_role,
                implementation: component.implementation.clone(),
                behavioral_role: component.behavioral_role.clone(),
                source_layer_index: component.circuit.source.source_layer_index,
                circuit_id: component.circuit.id.clone(),
                input_ports: component
                    .circuit
                    .boundary
                    .inputs
                    .iter()
                    .map(package_port_report)
                    .collect::<Vec<_>>(),
                output_ports: component
                    .circuit
                    .boundary
                    .outputs
                    .iter()
                    .map(package_port_report)
                    .collect::<Vec<_>>(),
                state_port_count: component.circuit.state_ports.len(),
                parameter_ref_count: component.params.refs.len(),
                node_count: component.circuit.nodes.len(),
                kernel_count: match component.runtime_role {
                    nerve_runtime::CircuitRuntimeRole::SignalProcessor => execution
                        .map(|execution| execution.kernels.len())
                        .unwrap_or(0),
                    nerve_runtime::CircuitRuntimeRole::InputTransducer => 1,
                    nerve_runtime::CircuitRuntimeRole::OutputTransducer => 2,
                    nerve_runtime::CircuitRuntimeRole::Sampler => manifest.sampler.kernels.len(),
                    nerve_runtime::CircuitRuntimeRole::DraftProcessor
                    | nerve_runtime::CircuitRuntimeRole::DraftInputAdapter
                    | nerve_runtime::CircuitRuntimeRole::DraftOutputTransducer => 0,
                },
            }
        })
        .collect::<Vec<_>>()
}

fn inspect_available_devices(
    default_device_id: &str,
    selected_vulkan_device_index: Option<usize>,
) -> Vec<RuntimeAvailableDevice> {
    discover_runtime_devices(default_device_id, selected_vulkan_device_index)
}

fn inspect_graph(
    args: &Args,
    package_manifest: &Path,
    manifest_dir: &Path,
    manifest: VulkanResidentModelPackageManifest,
) -> Result<(), Box<dyn Error>> {
    let source_graph = manifest.resolved_source_graph(manifest_dir.to_path_buf())?;
    let runtime_graph = manifest.runtime_graph_from_controls(
        args.default_device_id.as_deref(),
        &args.node_devices,
        &args.duplicate_after,
        args.source_chain.as_deref(),
    )?;
    let effective_graph = source_graph.instantiate_runtime_graph(&runtime_graph)?;
    let placement = effective_graph.placement_plan(&runtime_graph.placement_spec())?;
    let placement_device_ids = placement_device_ids(&placement.components);
    let instance_count = runtime_graph.instances.len();
    let edge_count = placement.edges.len();
    let payload = RuntimeGraphInspectionReport {
        ok: true,
        package_manifest: package_manifest.to_path_buf(),
        package_root: manifest_dir.to_path_buf(),
        package_id: manifest.package_id.clone(),
        compiled_source_component_count: source_graph.circuits.len(),
        runtime_graph_controls: runtime_graph_report(args),
        runtime_graph: runtime_graph,
        device_bindings: runtime_device_bindings_report(args, &placement_device_ids),
        effective_component_count: instance_count,
        effective_edge_count: edge_count,
        placement: RuntimeGraphPlacementReport {
            schema: placement.schema,
            topology: placement.topology,
            local_edge_count: placement.local_edge_count,
            cross_device_edge_count: placement.cross_device_edge_count,
            runtime_routes: runtime_edge_routes_report(args, &placement.edges),
            components: placement.components,
            edges: placement.edges,
        },
    };

    if args.json {
        println!("{}", serde_json::to_string_pretty(&payload)?);
    } else {
        println!("package_id={}", payload.package_id);
        println!("effective_node_count={}", payload.effective_component_count);
        println!("effective_edge_count={}", payload.effective_edge_count);
        println!(
            "cross_device_edge_count={}",
            payload.placement.cross_device_edge_count
        );
        for component in &payload.placement.components {
            println!(
                "{} circuit={} device={}",
                component.component_id, component.circuit_id, component.device_id
            );
        }
    }

    Ok(())
}

fn package_port_report(port: &CircuitPort) -> RuntimeComponentPortSummary {
    RuntimeComponentPortSummary {
        id: port.id.clone(),
        signal: port.signal.clone(),
        shape: port.shape.clone(),
        source: port.source.clone(),
        component_port: port.component_port.clone(),
    }
}

fn inspect_device_slice(
    args: &Args,
    package_manifest: &Path,
    manifest_dir: &Path,
    runtime_model: VulkanResidentRuntimeModel,
    device_id: &str,
) -> Result<(), Box<dyn Error>> {
    let capacity = choose_runtime_context_size(package_manifest, args.context_size, 1)?;
    let logical_device_ids = vec![device_id.to_string()];
    let bound_devices = runtime_bound_vulkan_devices(args, &logical_device_ids)?;
    let device = bound_devices.devices.get(device_id).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("logical device {device_id:?} was not mounted"),
        )
    })?;
    let payload = inspect_device_slice_payload(
        device,
        package_manifest,
        manifest_dir,
        runtime_model,
        device_id,
        capacity,
    )?;

    if args.json {
        println!("{}", serde_json::to_string_pretty(&payload)?);
    } else {
        println!("device_id={}", payload.device_id);
        println!("hosted_node_count={}", payload.hosted_component_count);
        println!("incoming_edge_count={}", payload.incoming_edge_count);
        println!("outgoing_edge_count={}", payload.outgoing_edge_count);
        println!("dispatch_count={}", payload.dispatch_count);
        println!("descriptor_count={}", payload.descriptor_count);
        println!("tick_stage_count={}", payload.tick_plan.stage_count);
    }

    Ok(())
}

fn inspect_placement(
    args: &Args,
    package_manifest: &Path,
    manifest_dir: &Path,
    runtime_model: VulkanResidentRuntimeModel,
) -> Result<(), Box<dyn Error>> {
    let capacity = choose_runtime_context_size(package_manifest, args.context_size, 1)?;
    let device_ids = runtime_model.placement_device_ids();
    let placement = runtime_model_placement(manifest_dir, &runtime_model)?;
    let bound_devices = runtime_bound_vulkan_devices(args, &device_ids)?;
    let slices = device_ids
        .iter()
        .map(|device_id| {
            let device = bound_devices.devices.get(device_id).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("logical device {device_id:?} was not mounted"),
                )
            })?;
            inspect_device_slice_payload(
                device,
                package_manifest,
                manifest_dir,
                runtime_model.clone(),
                device_id,
                capacity,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let payload = RuntimePlacementReport {
        ok: true,
        package_manifest: package_manifest.to_path_buf(),
        context_window_activations: capacity,
        runtime_graph: runtime_graph_report(args),
        device_bindings: runtime_device_bindings_report(args, &device_ids),
        bound_devices: bound_devices_report(&bound_devices),
        edge_routes: bound_edge_routes_report(&bound_devices, &placement.edges),
        device_count: device_ids.len(),
        device_ids,
        devices: slices,
    };

    if args.json {
        println!("{}", serde_json::to_string_pretty(&payload)?);
    } else {
        println!("device_count={}", payload.device_count);
        for device in &payload.devices {
            println!(
                "{} nodes={} incoming={} outgoing={} dispatches={}",
                device.device_id,
                device.hosted_component_count,
                device.incoming_edge_count,
                device.outgoing_edge_count,
                device.dispatch_count
            );
        }
    }

    Ok(())
}

fn inspect_device_slice_payload(
    device: &VulkanComputeDevice,
    package_manifest: &Path,
    manifest_dir: &Path,
    runtime_model: VulkanResidentRuntimeModel,
    device_id: &str,
    capacity: usize,
) -> Result<RuntimeDeviceSliceReport, Box<dyn Error>> {
    let slice = VulkanResidentModelPackageDeviceSlice::from_runtime_model_for_device(
        device,
        manifest_dir,
        runtime_model,
        device_id,
        Some(capacity),
    )?;
    let mounted = slice.create_mounted_stream_circuit(device)?;
    let reusable_manifest = VulkanReusableKernelArtifactManifest::new(
        slice
            .loaded_manifest()
            .artifacts
            .iter()
            .map(|artifact| artifact.artifact.clone())
            .collect(),
    );
    let mounted_bound = mounted.mounted_placed_bound_dispatch_plan(&reusable_manifest)?;
    let tick_plan = mounted.stream_tick_plan(&reusable_manifest)?;
    let resident_plan = &mounted.placed_plan.placed_resident_plan;
    let loaded_kernel_artifact_count = slice.loaded_manifest().artifacts.len();

    Ok(RuntimeDeviceSliceReport {
        ok: true,
        package_manifest: package_manifest.to_path_buf(),
        device_name: device.device_name().to_string(),
        device_id: slice.device_id,
        context_window_activations: capacity,
        hosted_components: resident_plan.hosted_component_ids.clone(),
        local_edges: resident_plan
            .local_edges
            .iter()
            .map(|edge| RuntimeLocalEdgeBufferReport {
                edge_index: edge.edge_index,
                signal: edge.signal.clone(),
                source_component_id: edge.source_component_id.clone(),
                destination_component_id: edge.destination_component_id.clone(),
                device_id: edge.source_device_id.clone(),
                byte_capacity: mounted
                    .edge_io
                    .local_edge_buffer(edge.edge_index)
                    .map(|buffer| buffer.byte_capacity),
            })
            .collect::<Vec<_>>(),
        incoming_edges: resident_plan
            .incoming_edges
            .iter()
            .map(|edge| RuntimeRemoteEdgeBufferReport {
                edge_index: edge.edge_index,
                signal: edge.signal.clone(),
                source_device_id: edge.source_device_id.clone(),
                source_component_id: edge.source_component_id.clone(),
                destination_device_id: edge.destination_device_id.clone(),
                destination_component_id: edge.destination_component_id.clone(),
                byte_capacity: mounted
                    .edge_io
                    .incoming_buffer(edge.edge_index)
                    .map(|buffer| buffer.byte_capacity),
            })
            .collect::<Vec<_>>(),
        outgoing_edges: resident_plan
            .outgoing_edges
            .iter()
            .map(|edge| RuntimeRemoteEdgeBufferReport {
                edge_index: edge.edge_index,
                signal: edge.signal.clone(),
                source_device_id: edge.source_device_id.clone(),
                source_component_id: edge.source_component_id.clone(),
                destination_device_id: edge.destination_device_id.clone(),
                destination_component_id: edge.destination_component_id.clone(),
                byte_capacity: mounted
                    .edge_io
                    .outgoing_buffer(edge.edge_index)
                    .map(|buffer| buffer.byte_capacity),
            })
            .collect::<Vec<_>>(),
        hosted_component_count: slice.hosted_component_count,
        incoming_edge_count: slice.incoming_edge_count,
        outgoing_edge_count: slice.outgoing_edge_count,
        permanent_parameter_count: slice.permanent_parameter_count,
        permanent_parameter_bytes: slice.permanent_parameter_bytes,
        reusable_kernel_word_count: slice.reusable_kernel_word_count,
        loaded_kernel_artifact_count,
        dispatch_count: mounted_bound.dispatches.len(),
        descriptor_count: mounted_bound.total_descriptor_count,
        model_boundary_descriptor_count: mounted_bound.model_boundary_descriptor_count,
        incoming_edge_descriptor_count: mounted_bound.incoming_edge_descriptor_count,
        outgoing_edge_descriptor_count: mounted_bound.outgoing_edge_descriptor_count,
        tick_plan: RuntimeDeviceTickPlanReport {
            stage_count: tick_plan.stage_count,
            receive_stage_count: tick_plan.receive_stage_count,
            dispatch_stage_count: tick_plan.dispatch_stage_count,
            publish_stage_count: tick_plan.publish_stage_count,
            local_edge_read_count: tick_plan.local_edge_read_count,
            local_edge_write_count: tick_plan.local_edge_write_count,
            incoming_edge_read_count: tick_plan.incoming_edge_read_count,
            outgoing_edge_write_count: tick_plan.outgoing_edge_write_count,
            model_input_read_count: tick_plan.model_input_read_count,
            model_output_write_count: tick_plan.model_output_write_count,
            can_execute: tick_plan.can_execute,
        },
    })
}
