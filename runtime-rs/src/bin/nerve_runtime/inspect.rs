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
    let patch = manifest.runtime_patch_from_controls(
        args.default_device_id.as_deref(),
        &args.pedal_devices,
        &args.duplicate_after,
        args.source_chain.as_deref(),
    )?;
    let effective_graph = source_graph.instantiate_runtime_patch(&patch)?;
    let placement = effective_graph.placement_plan(&patch.placement_spec())?;
    let placement_device_ids = placement_device_ids(&placement.pedals);
    let runtime_routes = runtime_cable_routes_report(args, &placement.cables);
    let device_bindings = runtime_device_bindings_report(args, &placement_device_ids);
    let source_pedals = source_pedals_report(&manifest);
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
        compiled: RuntimeCompiledPedalboardSummary {
            wiring: manifest.circuit_graph.wiring.clone(),
            source_pedal_count: source_pedals.len(),
            source_pedals,
            max_context_activations: manifest.max_context_activations,
        },
        runtime_patch_controls: runtime_patch_report(args),
        runtime_patch: patch,
        effective: RuntimeEffectivePedalboardTopology {
            wiring: placement.wiring,
            pedal_count: placement.pedals.len(),
            cable_count: placement.cables.len(),
            local_cable_count: placement.local_cable_count,
            cross_device_cable_count: placement.cross_device_cable_count,
            device_count: placement_device_ids.len(),
            device_ids: placement_device_ids,
            device_bindings,
            cable_routes: runtime_routes,
            pedals: placement.pedals,
            cables: placement.cables,
        },
    };

    if args.json {
        println!("{}", serde_json::to_string_pretty(&payload)?);
    } else {
        println!("package_id={}", payload.package_id);
        println!("source_pedal_count={}", payload.compiled.source_pedal_count);
        println!("effective_pedal_count={}", payload.effective.pedal_count);
        println!("device_count={}", payload.effective.device_count);
        println!(
            "cross_device_cable_count={}",
            payload.effective.cross_device_cable_count
        );
        println!(
            "same_physical_target_cable_count={}",
            payload
                .effective
                .cable_routes
                .same_physical_target_cable_count
        );
        println!(
            "cross_physical_target_cable_count={}",
            payload
                .effective
                .cable_routes
                .cross_physical_target_cable_count
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
    let source_pedals = source_pedals_report(&manifest);
    let source_pedal_count = source_pedals.len();
    let payload = RuntimePackageInspectionReport {
        ok: true,
        package_manifest: package_manifest.to_path_buf(),
        package_root: manifest_dir.to_path_buf(),
        schema: manifest.schema.clone(),
        package_id: manifest.package_id.clone(),
        config_path: manifest.config_path.clone(),
        tokenizer: serde_json::to_value(&manifest.tokenizer)?,
        compiled_wiring: manifest.circuit_graph.wiring.clone(),
        runtime_patch: runtime_patch_report(args),
        device_bindings: runtime_device_bindings_report(args, &[]),
        max_context_activations: manifest.max_context_activations,
        source_pedal_count,
        source_pedals,
        available_devices,
    };

    if args.json {
        println!("{}", serde_json::to_string_pretty(&payload)?);
    } else {
        println!("package_id={}", payload.package_id);
        println!("source_pedal_count={}", payload.source_pedal_count);
        println!("compiled_wiring={}", payload.compiled_wiring);
        for pedal in &payload.source_pedals {
            println!(
                "{} {} kernels={} state_ports={}",
                pedal.pedal_id, pedal.operator_type, pedal.kernel_count, pedal.state_port_count
            );
        }
    }

    Ok(())
}

fn source_pedals_report(manifest: &VulkanResidentModelPackageManifest) -> Vec<RuntimeSourcePedal> {
    let execution_by_pedal = manifest
        .pedal_executions
        .iter()
        .map(|execution| (execution.pedal_id.as_str(), execution))
        .collect::<BTreeMap<_, _>>();

    manifest
        .circuit_graph
        .pedals
        .iter()
        .enumerate()
        .map(|(pedal_index, pedal)| {
            let execution = execution_by_pedal.get(pedal.pedal_id.as_str());
            RuntimeSourcePedal {
                pedal_index,
                pedal_id: pedal.pedal_id.clone(),
                operator_type: pedal.operator_type.clone(),
                runtime_role: pedal.circuit.runtime_role,
                implementation: pedal.implementation.clone(),
                behavioral_role: pedal.behavioral_role.clone(),
                source_layer_index: pedal.circuit.source.source_layer_index,
                circuit_id: pedal.circuit.id.clone(),
                input_ports: pedal
                    .circuit
                    .boundary
                    .inputs
                    .iter()
                    .map(package_port_report)
                    .collect::<Vec<_>>(),
                output_ports: pedal
                    .circuit
                    .boundary
                    .outputs
                    .iter()
                    .map(package_port_report)
                    .collect::<Vec<_>>(),
                state_port_count: pedal.circuit.state_ports.len(),
                parameter_ref_count: pedal.params.refs.len(),
                node_count: pedal.circuit.nodes.len(),
                kernel_count: match pedal.runtime_role {
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

fn inspect_patch(
    args: &Args,
    package_manifest: &Path,
    manifest_dir: &Path,
    manifest: VulkanResidentModelPackageManifest,
) -> Result<(), Box<dyn Error>> {
    let source_graph = manifest.resolved_source_graph(manifest_dir.to_path_buf())?;
    let patch = manifest.runtime_patch_from_controls(
        args.default_device_id.as_deref(),
        &args.pedal_devices,
        &args.duplicate_after,
        args.source_chain.as_deref(),
    )?;
    let effective_graph = source_graph.instantiate_runtime_patch(&patch)?;
    let placement = effective_graph.placement_plan(&patch.placement_spec())?;
    let placement_device_ids = placement_device_ids(&placement.pedals);
    let instance_count = patch.instances.len();
    let cable_count = placement.cables.len();
    let payload = RuntimePatchInspectionReport {
        ok: true,
        package_manifest: package_manifest.to_path_buf(),
        package_root: manifest_dir.to_path_buf(),
        package_id: manifest.package_id.clone(),
        compiled_source_pedal_count: source_graph.circuits.len(),
        runtime_patch_controls: runtime_patch_report(args),
        runtime_patch: patch,
        device_bindings: runtime_device_bindings_report(args, &placement_device_ids),
        effective_pedal_count: instance_count,
        effective_cable_count: cable_count,
        placement: RuntimePatchPlacementReport {
            schema: placement.schema,
            wiring: placement.wiring,
            local_cable_count: placement.local_cable_count,
            cross_device_cable_count: placement.cross_device_cable_count,
            runtime_routes: runtime_cable_routes_report(args, &placement.cables),
            pedals: placement.pedals,
            cables: placement.cables,
        },
    };

    if args.json {
        println!("{}", serde_json::to_string_pretty(&payload)?);
    } else {
        println!("package_id={}", payload.package_id);
        println!("effective_pedal_count={}", payload.effective_pedal_count);
        println!("effective_cable_count={}", payload.effective_cable_count);
        println!(
            "cross_device_cable_count={}",
            payload.placement.cross_device_cable_count
        );
        for pedal in &payload.placement.pedals {
            println!(
                "{} circuit={} device={}",
                pedal.pedal_id, pedal.circuit_id, pedal.device_id
            );
        }
    }

    Ok(())
}

fn package_port_report(port: &CircuitPort) -> RuntimePedalPortSummary {
    RuntimePedalPortSummary {
        id: port.id.clone(),
        signal: port.signal.clone(),
        shape: port.shape.clone(),
        source: port.source.clone(),
        pedal_port: port.pedal_port.clone(),
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
        println!("hosted_pedal_count={}", payload.hosted_pedal_count);
        println!("incoming_cable_count={}", payload.incoming_cable_count);
        println!("outgoing_cable_count={}", payload.outgoing_cable_count);
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
        runtime_patch: runtime_patch_report(args),
        device_bindings: runtime_device_bindings_report(args, &device_ids),
        bound_devices: bound_devices_report(&bound_devices),
        cable_routes: bound_cable_routes_report(&bound_devices, &placement.cables),
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
                "{} pedals={} incoming={} outgoing={} dispatches={}",
                device.device_id,
                device.hosted_pedal_count,
                device.incoming_cable_count,
                device.outgoing_cable_count,
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
        hosted_pedals: resident_plan.hosted_pedal_ids.clone(),
        local_cables: resident_plan
            .local_cables
            .iter()
            .map(|cable| RuntimeLocalCableBufferReport {
                cable_index: cable.cable_index,
                signal: cable.signal.clone(),
                source_pedal_id: cable.source_pedal_id.clone(),
                destination_pedal_id: cable.destination_pedal_id.clone(),
                device_id: cable.source_device_id.clone(),
                byte_capacity: mounted
                    .cable_io
                    .local_cable_buffer(cable.cable_index)
                    .map(|buffer| buffer.byte_capacity),
            })
            .collect::<Vec<_>>(),
        incoming_cables: resident_plan
            .incoming_cables
            .iter()
            .map(|cable| RuntimeRemoteCableBufferReport {
                cable_index: cable.cable_index,
                signal: cable.signal.clone(),
                source_device_id: cable.source_device_id.clone(),
                source_pedal_id: cable.source_pedal_id.clone(),
                destination_device_id: cable.destination_device_id.clone(),
                destination_pedal_id: cable.destination_pedal_id.clone(),
                byte_capacity: mounted
                    .cable_io
                    .incoming_buffer(cable.cable_index)
                    .map(|buffer| buffer.byte_capacity),
            })
            .collect::<Vec<_>>(),
        outgoing_cables: resident_plan
            .outgoing_cables
            .iter()
            .map(|cable| RuntimeRemoteCableBufferReport {
                cable_index: cable.cable_index,
                signal: cable.signal.clone(),
                source_device_id: cable.source_device_id.clone(),
                source_pedal_id: cable.source_pedal_id.clone(),
                destination_device_id: cable.destination_device_id.clone(),
                destination_pedal_id: cable.destination_pedal_id.clone(),
                byte_capacity: mounted
                    .cable_io
                    .outgoing_buffer(cable.cable_index)
                    .map(|buffer| buffer.byte_capacity),
            })
            .collect::<Vec<_>>(),
        hosted_pedal_count: slice.hosted_pedal_count,
        incoming_cable_count: slice.incoming_cable_count,
        outgoing_cable_count: slice.outgoing_cable_count,
        permanent_parameter_count: slice.permanent_parameter_count,
        permanent_parameter_bytes: slice.permanent_parameter_bytes,
        reusable_kernel_word_count: slice.reusable_kernel_word_count,
        loaded_kernel_artifact_count,
        dispatch_count: mounted_bound.dispatches.len(),
        descriptor_count: mounted_bound.total_descriptor_count,
        model_boundary_descriptor_count: mounted_bound.model_boundary_descriptor_count,
        incoming_cable_descriptor_count: mounted_bound.incoming_cable_descriptor_count,
        outgoing_cable_descriptor_count: mounted_bound.outgoing_cable_descriptor_count,
        tick_plan: RuntimeDeviceTickPlanReport {
            stage_count: tick_plan.stage_count,
            receive_stage_count: tick_plan.receive_stage_count,
            dispatch_stage_count: tick_plan.dispatch_stage_count,
            publish_stage_count: tick_plan.publish_stage_count,
            local_cable_read_count: tick_plan.local_cable_read_count,
            local_cable_write_count: tick_plan.local_cable_write_count,
            incoming_cable_read_count: tick_plan.incoming_cable_read_count,
            outgoing_cable_write_count: tick_plan.outgoing_cable_write_count,
            model_input_read_count: tick_plan.model_input_read_count,
            model_output_write_count: tick_plan.model_output_write_count,
            can_execute: tick_plan.can_execute,
        },
    })
}

