use std::collections::BTreeMap;
use std::error::Error;
use std::io;
use std::path::{Path, PathBuf};

use llmoop_runtime::{
    CircuitPort, VulkanComputeDevice, VulkanResidentGreedyInProcessPlacedModelPackage,
    VulkanResidentGreedyModelPackage, VulkanResidentGreedyModelPackageDeviceSlice,
    VulkanResidentGreedyModelPackageManifest, VulkanResidentHfTokenizerTextCodec,
    VulkanResidentTokenEngine, VulkanResidentTokenEngineRunBudget,
    VulkanResidentTokenEngineRunStopCondition, VulkanResidentTokenTextCodec,
    VulkanReusableKernelArtifactManifest,
};
use serde_json::{Value, json};

#[derive(Clone, Debug, PartialEq, Eq)]
struct Args {
    package_manifest: Option<PathBuf>,
    prompt: Option<String>,
    inspect_package: bool,
    inspect_patch: bool,
    inspect_placement: bool,
    inspect_device_slice: Option<String>,
    default_device_id: Option<String>,
    pedal_devices: BTreeMap<String, String>,
    duplicate_after: Vec<(String, String)>,
    source_chain: Option<Vec<(String, String)>>,
    max_new_tokens: usize,
    capacity: Option<usize>,
    cycle_ticks: usize,
    max_scheduler_turns: usize,
    add_special_tokens: bool,
    skip_special_tokens: bool,
    generated_only: bool,
    json: bool,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            package_manifest: None,
            prompt: None,
            inspect_package: false,
            inspect_patch: false,
            inspect_placement: false,
            inspect_device_slice: None,
            default_device_id: None,
            pedal_devices: BTreeMap::new(),
            duplicate_after: Vec::new(),
            source_chain: None,
            max_new_tokens: 4,
            capacity: None,
            cycle_ticks: 4,
            max_scheduler_turns: 1_024,
            add_special_tokens: true,
            skip_special_tokens: true,
            generated_only: false,
            json: false,
        }
    }
}

fn main() {
    if let Err(error) = run() {
        eprintln!("llmoop-runtime error: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    if std::env::args()
        .skip(1)
        .any(|arg| arg == "--help" || arg == "-h")
    {
        print_usage();
        return Ok(());
    }

    let args = parse_args().map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error))?;
    let package_manifest = args.package_manifest.as_ref().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "--package is required; run `python -m llmoop --compile-model <MODEL_DIR>` first",
        )
    })?;
    let manifest_dir = package_manifest
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    if args.inspect_package {
        let manifest = VulkanResidentGreedyModelPackageManifest::from_json_file(package_manifest)?;
        return inspect_package(&args, package_manifest, &manifest_dir, manifest);
    }
    if args.inspect_patch {
        let manifest = VulkanResidentGreedyModelPackageManifest::from_json_file(package_manifest)?;
        return inspect_patch(&args, package_manifest, &manifest_dir, manifest);
    }
    let manifest = runtime_manifest(&args, package_manifest)?;
    if args.inspect_placement {
        return inspect_placement(&args, package_manifest, &manifest_dir, manifest);
    }
    if let Some(device_id) = args.inspect_device_slice.as_deref() {
        return inspect_device_slice(&args, package_manifest, &manifest_dir, manifest, device_id);
    }
    let prompt = args
        .prompt
        .as_ref()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "--prompt is required"))?;
    let tokenizer_dir = tokenizer_dir_from_package(package_manifest)?;
    let codec = VulkanResidentHfTokenizerTextCodec::from_model_dir(&tokenizer_dir)?
        .with_add_special_tokens(args.add_special_tokens)
        .with_skip_special_tokens(args.skip_special_tokens);
    let prompt_ids = codec.encode_text(prompt)?;
    let needed_capacity = prompt_ids
        .len()
        .checked_add(args.max_new_tokens)
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "prompt token count plus --max-new-tokens overflowed usize",
            )
        })?;
    let capacity = choose_runtime_capacity(package_manifest, args.capacity, needed_capacity)?;

    let device = VulkanComputeDevice::new()?;
    if manifest.placement_device_ids().len() > 1 {
        return run_placed_prompt(
            &args,
            package_manifest,
            &manifest_dir,
            &tokenizer_dir,
            prompt,
            prompt_ids,
            capacity,
            device,
            manifest,
            &codec,
        );
    }

    let model = VulkanResidentGreedyModelPackage::from_manifest(
        &device,
        &manifest_dir,
        manifest,
        Some(capacity),
    )?;
    let mut engine = VulkanResidentTokenEngine::new(device);
    engine.add_model_package("compiled_model", model)?;
    engine.create_stream_from_model("compiled_model", "main")?;

    let turn = engine.submit_live_text_turn_until_idle(
        "main",
        "prompt",
        prompt.clone(),
        args.max_new_tokens,
        "cli",
        VulkanResidentTokenEngineRunBudget::new(args.max_scheduler_turns, 1, args.cycle_ticks),
        &codec,
    )?;
    let stream = engine
        .stream("main")
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "runtime stream disappeared"))?;
    let snapshot = engine.snapshot();

    if args.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "ok": true,
                "execution_mode": "single_device_resident",
                "package_manifest": package_manifest,
                "tokenizer_dir": tokenizer_dir,
                "device_name": snapshot.device_name,
                "device_id": stream.device_id,
                "runtime_patch": runtime_patch_report(&args),
                "pedal_count": stream.pedal_count,
                "dispatches_per_tick": stream.per_tick_dispatch_count,
                "descriptors_per_tick": stream.per_tick_descriptor_count,
                "push_constant_bytes_per_tick": stream.per_tick_push_constant_byte_count,
                "resident_capacity_activations": stream.dynamic_state_capacity_activations,
                "needed_capacity_activations": needed_capacity,
                "tokenizer": {
                    "add_special_tokens": args.add_special_tokens,
                    "skip_special_tokens": args.skip_special_tokens,
                },
                "prompt_text": prompt,
                "prompt_ids": turn.queued_input_event.encoded_token_ids,
                "generated_ids": turn.generated_token_ids,
                "generated_text": turn.generated_text,
                "output_text": turn.output_text,
                "stop_reason": engine_stop_label(turn.stop_condition),
                "scheduler_turns": turn.scheduler_turn_count(),
                "runtime_cycles": turn.runtime_cycle_count,
            }))?
        );
    } else if args.generated_only {
        print_text(&turn.generated_text);
    } else {
        print_text(&turn.output_text);
    }

    Ok(())
}

fn run_placed_prompt(
    args: &Args,
    package_manifest: &Path,
    manifest_dir: &Path,
    tokenizer_dir: &Path,
    prompt: &str,
    prompt_ids: Vec<u32>,
    capacity: usize,
    device: VulkanComputeDevice,
    manifest: VulkanResidentGreedyModelPackageManifest,
    codec: &VulkanResidentHfTokenizerTextCodec,
) -> Result<(), Box<dyn Error>> {
    let package = VulkanResidentGreedyInProcessPlacedModelPackage::from_manifest_for_devices(
        &device,
        manifest_dir,
        manifest,
        Some(capacity),
    )?;
    let run = package.run_prompt_event_bounded_in_process(
        &device,
        &prompt_ids,
        0,
        args.max_new_tokens,
        None,
        args.max_scheduler_turns,
    )?;
    let generated_text = codec.decode_tokens(&run.generated_token_ids)?;
    let output_text = codec.decode_tokens(&run.output_token_ids)?;
    let total_scheduler_turns = run
        .tick_runs
        .iter()
        .map(|tick| tick.tick_run.placed_run.scheduler_turn_count)
        .sum::<usize>();
    let completed_stage_deltas = run
        .tick_runs
        .iter()
        .map(|tick| tick.tick_run.placed_run.completed_stage_delta)
        .collect::<Vec<_>>();

    if args.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "ok": true,
                "execution_mode": "placed_in_process",
                "package_manifest": package_manifest,
                "tokenizer_dir": tokenizer_dir,
                "device_name": device.device_name(),
                "boundary_device_id": package.boundary_device_id,
                "device_count": package.device_count,
                "device_ids": package.device_ids,
                "runtime_patch": runtime_patch_report(args),
                "hosted_pedal_count": package.hosted_pedal_count,
                "resident_capacity_activations": package.dynamic_state_capacity_activations,
                "needed_capacity_activations": prompt_ids.len() + args.max_new_tokens,
                "tokenizer": {
                    "add_special_tokens": args.add_special_tokens,
                    "skip_special_tokens": args.skip_special_tokens,
                },
                "prompt_text": prompt,
                "prompt_ids": run.prompt_token_ids,
                "generated_ids": run.generated_token_ids,
                "generated_text": generated_text,
                "output_text": output_text,
                "stop_reason": run.stop_reason,
                "tick_count": run.tick_runs.len(),
                "scheduler_turns": total_scheduler_turns,
                "max_scheduler_turns_per_tick": args.max_scheduler_turns,
                "completed_stage_deltas": completed_stage_deltas,
            }))?
        );
    } else if args.generated_only {
        print_text(&generated_text);
    } else {
        print_text(&output_text);
    }

    Ok(())
}

fn inspect_package(
    args: &Args,
    package_manifest: &Path,
    manifest_dir: &Path,
    manifest: VulkanResidentGreedyModelPackageManifest,
) -> Result<(), Box<dyn Error>> {
    let default_device_id = args
        .default_device_id
        .as_deref()
        .unwrap_or(&manifest.placement.default_device_id);
    let available_devices = inspect_available_devices(default_device_id);
    let execution_by_pedal = manifest
        .pedal_executions
        .iter()
        .map(|execution| (execution.pedal_id.as_str(), execution))
        .collect::<BTreeMap<_, _>>();
    let source_pedals = manifest
        .circuit_graph
        .pedals
        .iter()
        .enumerate()
        .map(|(pedal_index, pedal)| {
            let execution = execution_by_pedal.get(pedal.pedal_id.as_str());
            json!({
                "pedal_index": pedal_index,
                "pedal_id": pedal.pedal_id,
                "operator_type": pedal.operator_type,
                "implementation": pedal.implementation,
                "behavioral_role": pedal.behavioral_role,
                "source_layer_index": pedal.circuit.source.source_layer_index,
                "circuit_id": pedal.circuit.id,
                "input_ports": pedal.circuit.boundary.inputs.iter().map(package_port_report).collect::<Vec<_>>(),
                "output_ports": pedal.circuit.boundary.outputs.iter().map(package_port_report).collect::<Vec<_>>(),
                "state_port_count": pedal.circuit.state_ports.len(),
                "parameter_ref_count": pedal.params.refs.len(),
                "node_count": pedal.circuit.nodes.len(),
                "kernel_count": execution.map(|execution| execution.kernels.len()).unwrap_or(0),
            })
        })
        .collect::<Vec<_>>();
    let payload = json!({
        "ok": true,
        "package_manifest": package_manifest,
        "package_root": manifest_dir,
        "schema": manifest.schema,
        "package_id": manifest.package_id,
        "config_path": manifest.config_path,
        "tokenizer": manifest.tokenizer,
        "compiled_wiring": manifest.circuit_graph.wiring,
        "compiled_default_device_id": manifest.placement.default_device_id,
        "compiled_pedal_devices": manifest.placement.pedal_devices,
        "runtime_patch": runtime_patch_report(args),
        "dynamic_state_capacity_activations": manifest.dynamic_state_capacity_activations,
        "capacity_profiles": manifest.capacity_profiles.iter().map(|profile| json!({
            "min_dynamic_state_capacity_activations": profile.min_dynamic_state_capacity_activations,
            "max_dynamic_state_capacity_activations": profile.max_dynamic_state_capacity_activations,
            "shader_override_count": profile.pedal_execution_shader_overrides.len(),
        })).collect::<Vec<_>>(),
        "source_pedal_count": source_pedals.len(),
        "source_pedals": source_pedals,
        "available_devices": available_devices,
    });

    if args.json {
        println!("{}", serde_json::to_string_pretty(&payload)?);
    } else {
        println!("package_id={}", payload["package_id"]);
        println!("source_pedal_count={}", payload["source_pedal_count"]);
        println!("compiled_wiring={}", payload["compiled_wiring"]);
        println!(
            "default_device_id={}",
            payload["compiled_default_device_id"]
        );
        for pedal in payload["source_pedals"].as_array().into_iter().flatten() {
            println!(
                "{} {} kernels={} state_ports={}",
                pedal["pedal_id"],
                pedal["operator_type"],
                pedal["kernel_count"],
                pedal["state_port_count"]
            );
        }
    }

    Ok(())
}

fn inspect_available_devices(default_device_id: &str) -> Vec<Value> {
    match VulkanComputeDevice::available_compute_devices() {
        Ok(devices) if devices.is_empty() => vec![json!({
            "device_id": default_device_id,
            "backend": "vulkan_compute",
            "available": false,
            "notes": ["no compute-capable Vulkan physical devices were found"],
        })],
        Ok(devices) => devices
            .iter()
            .map(|device| {
                let runtime_device_id = device
                    .selected_by_default
                    .then(|| default_device_id.to_string());
                let device_id = runtime_device_id
                    .clone()
                    .unwrap_or_else(|| device.physical_device_id.clone());
                json!({
                    "device_id": device_id,
                    "runtime_device_id": runtime_device_id,
                    "physical_device_id": device.physical_device_id.clone(),
                    "physical_device_index": device.physical_device_index,
                    "backend": "vulkan_compute",
                    "device_name": device.device_name.clone(),
                    "device_type": device.device_type.clone(),
                    "vendor_id": device.vendor_id,
                    "raw_device_id": device.device_id,
                    "api_version": device.api_version,
                    "driver_version": device.driver_version,
                    "compute_queue_family_indices": device.compute_queue_family_indices.clone(),
                    "memory_heaps": device.memory_heaps.iter().map(|heap| json!({
                        "heap_index": heap.heap_index,
                        "size_bytes": heap.size_bytes,
                        "device_local": heap.device_local,
                    })).collect::<Vec<_>>(),
                    "available": true,
                    "selected_by_default": device.selected_by_default,
                    "runtime_binding": if device.selected_by_default {
                        "selected_local_vulkan_device"
                    } else {
                        "inventory_only"
                    },
                    "can_host_runtime_pedals_on_physical_device": device.selected_by_default,
                    "notes": if device.selected_by_default {
                        vec!["currently selected by VulkanComputeDevice::new()"]
                    } else {
                        vec!["detected by Vulkan inventory; explicit physical-device binding is not implemented yet"]
                    },
                })
            })
            .collect(),
        Err(error) => vec![json!({
            "device_id": default_device_id,
            "backend": "vulkan_compute",
            "available": false,
            "error": error.to_string(),
        })],
    }
}

fn inspect_patch(
    args: &Args,
    package_manifest: &Path,
    manifest_dir: &Path,
    manifest: VulkanResidentGreedyModelPackageManifest,
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
    let instance_count = patch.instances.len();
    let cable_count = placement.cables.len();
    let payload = json!({
        "ok": true,
        "package_manifest": package_manifest,
        "package_root": manifest_dir,
        "package_id": manifest.package_id,
        "compiled_source_pedal_count": source_graph.circuits.len(),
        "runtime_patch_controls": runtime_patch_report(args),
        "runtime_patch": patch,
        "effective_pedal_count": instance_count,
        "effective_cable_count": cable_count,
        "placement": {
            "schema": placement.schema,
            "wiring": placement.wiring,
            "local_cable_count": placement.local_cable_count,
            "cross_device_cable_count": placement.cross_device_cable_count,
            "pedals": placement.pedals,
            "cables": placement.cables,
        },
    });

    if args.json {
        println!("{}", serde_json::to_string_pretty(&payload)?);
    } else {
        println!("package_id={}", payload["package_id"]);
        println!("effective_pedal_count={}", payload["effective_pedal_count"]);
        println!("effective_cable_count={}", payload["effective_cable_count"]);
        println!(
            "cross_device_cable_count={}",
            payload["placement"]["cross_device_cable_count"]
        );
        for pedal in payload["placement"]["pedals"]
            .as_array()
            .into_iter()
            .flatten()
        {
            println!(
                "{} circuit={} device={}",
                pedal["pedal_id"], pedal["circuit_id"], pedal["device_id"]
            );
        }
    }

    Ok(())
}

fn package_port_report(port: &CircuitPort) -> Value {
    json!({
        "id": port.id,
        "signal": port.signal,
        "shape": port.shape,
        "source": port.source,
        "pedal_port": port.pedal_port,
    })
}

fn inspect_device_slice(
    args: &Args,
    package_manifest: &Path,
    manifest_dir: &Path,
    manifest: VulkanResidentGreedyModelPackageManifest,
    device_id: &str,
) -> Result<(), Box<dyn Error>> {
    let capacity = choose_runtime_capacity(package_manifest, args.capacity, 1)?;
    let device = VulkanComputeDevice::new()?;
    let payload = inspect_device_slice_payload(
        &device,
        package_manifest,
        manifest_dir,
        manifest,
        device_id,
        capacity,
    )?;

    if args.json {
        println!("{}", serde_json::to_string_pretty(&payload)?);
    } else {
        println!("device_id={}", payload["device_id"]);
        println!("hosted_pedal_count={}", payload["hosted_pedal_count"]);
        println!("incoming_cable_count={}", payload["incoming_cable_count"]);
        println!("outgoing_cable_count={}", payload["outgoing_cable_count"]);
        println!("dispatch_count={}", payload["dispatch_count"]);
        println!("descriptor_count={}", payload["descriptor_count"]);
        println!("tick_stage_count={}", payload["tick_plan"]["stage_count"]);
    }

    Ok(())
}

fn inspect_placement(
    args: &Args,
    package_manifest: &Path,
    manifest_dir: &Path,
    manifest: VulkanResidentGreedyModelPackageManifest,
) -> Result<(), Box<dyn Error>> {
    let capacity = choose_runtime_capacity(package_manifest, args.capacity, 1)?;
    let device_ids = manifest.placement_device_ids();
    let device = VulkanComputeDevice::new()?;
    let slices = device_ids
        .iter()
        .map(|device_id| {
            inspect_device_slice_payload(
                &device,
                package_manifest,
                manifest_dir,
                manifest.clone(),
                device_id,
                capacity,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let payload = json!({
        "ok": true,
        "package_manifest": package_manifest,
        "device_name": device.device_name(),
        "resident_capacity_activations": capacity,
        "runtime_patch": runtime_patch_report(args),
        "device_count": device_ids.len(),
        "device_ids": device_ids,
        "devices": slices,
    });

    if args.json {
        println!("{}", serde_json::to_string_pretty(&payload)?);
    } else {
        println!("device_count={}", payload["device_count"]);
        for device in payload["devices"].as_array().into_iter().flatten() {
            println!(
                "{} pedals={} incoming={} outgoing={} dispatches={}",
                device["device_id"],
                device["hosted_pedal_count"],
                device["incoming_cable_count"],
                device["outgoing_cable_count"],
                device["dispatch_count"]
            );
        }
    }

    Ok(())
}

fn inspect_device_slice_payload(
    device: &VulkanComputeDevice,
    package_manifest: &Path,
    manifest_dir: &Path,
    manifest: VulkanResidentGreedyModelPackageManifest,
    device_id: &str,
    capacity: usize,
) -> Result<Value, Box<dyn Error>> {
    let slice = VulkanResidentGreedyModelPackageDeviceSlice::from_manifest_for_device(
        device,
        manifest_dir,
        manifest,
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

    Ok(json!({
        "ok": true,
        "package_manifest": package_manifest,
        "device_name": device.device_name(),
        "device_id": slice.device_id,
        "resident_capacity_activations": capacity,
        "hosted_pedals": resident_plan.hosted_pedal_ids,
        "local_cables": resident_plan.local_cables.iter().map(|cable| json!({
            "cable_index": cable.cable_index,
            "signal": cable.signal,
            "source_pedal_id": cable.source_pedal_id,
            "destination_pedal_id": cable.destination_pedal_id,
            "device_id": cable.source_device_id,
            "byte_capacity": mounted.cable_io.local_cable_buffer(cable.cable_index).map(|buffer| buffer.byte_capacity),
        })).collect::<Vec<_>>(),
        "incoming_cables": resident_plan.incoming_cables.iter().map(|cable| json!({
            "cable_index": cable.cable_index,
            "signal": cable.signal,
            "source_device_id": cable.source_device_id,
            "source_pedal_id": cable.source_pedal_id,
            "destination_device_id": cable.destination_device_id,
            "destination_pedal_id": cable.destination_pedal_id,
            "byte_capacity": mounted.cable_io.incoming_buffer(cable.cable_index).map(|buffer| buffer.byte_capacity),
        })).collect::<Vec<_>>(),
        "outgoing_cables": resident_plan.outgoing_cables.iter().map(|cable| json!({
            "cable_index": cable.cable_index,
            "signal": cable.signal,
            "source_device_id": cable.source_device_id,
            "source_pedal_id": cable.source_pedal_id,
            "destination_device_id": cable.destination_device_id,
            "destination_pedal_id": cable.destination_pedal_id,
            "byte_capacity": mounted.cable_io.outgoing_buffer(cable.cable_index).map(|buffer| buffer.byte_capacity),
        })).collect::<Vec<_>>(),
        "hosted_pedal_count": slice.hosted_pedal_count,
        "incoming_cable_count": slice.incoming_cable_count,
        "outgoing_cable_count": slice.outgoing_cable_count,
        "permanent_parameter_count": slice.permanent_parameter_count,
        "permanent_parameter_bytes": slice.permanent_parameter_bytes,
        "reusable_kernel_word_count": slice.reusable_kernel_word_count,
        "loaded_kernel_artifact_count": slice.loaded_manifest().artifacts.len(),
        "dispatch_count": mounted_bound.dispatches.len(),
        "descriptor_count": mounted_bound.total_descriptor_count,
        "model_boundary_descriptor_count": mounted_bound.model_boundary_descriptor_count,
        "incoming_cable_descriptor_count": mounted_bound.incoming_cable_descriptor_count,
        "outgoing_cable_descriptor_count": mounted_bound.outgoing_cable_descriptor_count,
        "tick_plan": {
            "stage_count": tick_plan.stage_count,
            "receive_stage_count": tick_plan.receive_stage_count,
            "dispatch_stage_count": tick_plan.dispatch_stage_count,
            "publish_stage_count": tick_plan.publish_stage_count,
            "local_cable_read_count": tick_plan.local_cable_read_count,
            "local_cable_write_count": tick_plan.local_cable_write_count,
            "incoming_cable_read_count": tick_plan.incoming_cable_read_count,
            "outgoing_cable_write_count": tick_plan.outgoing_cable_write_count,
            "model_input_read_count": tick_plan.model_input_read_count,
            "model_output_write_count": tick_plan.model_output_write_count,
            "can_execute": tick_plan.can_execute,
        },
    }))
}

fn tokenizer_dir_from_package(package_manifest: &Path) -> Result<PathBuf, Box<dyn Error>> {
    let manifest = VulkanResidentGreedyModelPackageManifest::from_json_file(package_manifest)?;
    let manifest_dir = package_manifest
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let tokenizer_dir = resolve_package_path(&manifest_dir, &manifest.tokenizer.path);
    if !tokenizer_dir.join("tokenizer.json").is_file() {
        return Err(Box::new(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "compiled package declares tokenizer at {}, but tokenizer.json is missing",
                tokenizer_dir.display()
            ),
        )));
    }
    Ok(tokenizer_dir)
}

fn resolve_package_path(manifest_dir: &Path, raw_path: &str) -> PathBuf {
    let path = Path::new(raw_path);
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        manifest_dir.join(path)
    }
}

fn runtime_manifest(
    args: &Args,
    package_manifest: &Path,
) -> Result<VulkanResidentGreedyModelPackageManifest, Box<dyn Error>> {
    let manifest = VulkanResidentGreedyModelPackageManifest::from_json_file(package_manifest)?;
    Ok(manifest.with_runtime_patch_controls(
        args.default_device_id.as_deref(),
        &args.pedal_devices,
        &args.duplicate_after,
        args.source_chain.as_deref(),
    )?)
}

fn runtime_patch_report(args: &Args) -> Value {
    json!({
        "default_device_id": args.default_device_id.clone(),
        "pedal_devices": args.pedal_devices.clone(),
        "source_chain": args.source_chain.as_ref().map(|source_chain| {
            source_chain.iter().map(|(instance_id, source_pedal_id)| {
                json!({
                    "instance_id": instance_id,
                    "source_pedal_id": source_pedal_id,
                })
            }).collect::<Vec<_>>()
        }),
        "duplicate_after": args.duplicate_after.iter().map(|(after_instance_id, new_instance_id)| {
            json!({
                "after_instance_id": after_instance_id,
                "new_instance_id": new_instance_id,
            })
        }).collect::<Vec<_>>(),
    })
}

fn choose_runtime_capacity(
    package_manifest: &Path,
    requested_capacity: Option<usize>,
    needed_capacity: usize,
) -> Result<usize, Box<dyn Error>> {
    let manifest = VulkanResidentGreedyModelPackageManifest::from_json_file(package_manifest)?;
    let default_capacity = manifest.dynamic_state_capacity_activations;
    let max_supported_capacity = manifest
        .capacity_profiles
        .iter()
        .map(|profile| profile.max_dynamic_state_capacity_activations)
        .chain(std::iter::once(default_capacity))
        .max()
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "compiled package does not declare any supported dynamic-state capacity",
            )
        })?;

    if let Some(capacity) = requested_capacity {
        if capacity < needed_capacity {
            return Err(Box::new(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "requested capacity {capacity} is too small: prompt plus generation needs {needed_capacity} activations"
                ),
            )));
        }
        if capacity > max_supported_capacity {
            return Err(Box::new(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "requested capacity {capacity} exceeds compiled package support ({max_supported_capacity}); recompile with a larger capacity"
                ),
            )));
        }
        let supported = capacity == default_capacity
            || manifest.capacity_profiles.iter().any(|profile| {
                profile.min_dynamic_state_capacity_activations <= capacity
                    && capacity <= profile.max_dynamic_state_capacity_activations
            });
        if !supported {
            return Err(Box::new(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "requested capacity {capacity} is not supported by this compiled package; recompile with a matching capacity profile"
                ),
            )));
        }
        return Ok(capacity);
    }

    if default_capacity >= needed_capacity {
        return Ok(default_capacity);
    }

    let mut profiles = manifest.capacity_profiles;
    profiles.sort_by_key(|profile| {
        (
            profile.max_dynamic_state_capacity_activations,
            profile.min_dynamic_state_capacity_activations,
        )
    });
    if let Some(profile) = profiles
        .into_iter()
        .find(|profile| needed_capacity <= profile.max_dynamic_state_capacity_activations)
    {
        return Ok(needed_capacity.max(profile.min_dynamic_state_capacity_activations));
    }

    Err(Box::new(io::Error::new(
        io::ErrorKind::InvalidInput,
        format!(
            "prompt plus generation needs {needed_capacity} activations, but compiled package supports up to {max_supported_capacity}; recompile with a larger capacity"
        ),
    )))
}

fn parse_args() -> Result<Args, String> {
    let mut parsed = Args::default();
    let mut raw = std::env::args().skip(1);

    while let Some(arg) = raw.next() {
        match arg.as_str() {
            "--package" | "--package-manifest" => {
                parsed.package_manifest = Some(PathBuf::from(next_value(&mut raw, &arg)?));
            }
            "--prompt" => {
                parsed.prompt = Some(next_value(&mut raw, "--prompt")?);
            }
            "--inspect-package" | "--inspect-pedals" => {
                parsed.inspect_package = true;
            }
            "--inspect-patch" => {
                parsed.inspect_patch = true;
            }
            "--inspect-placement" => {
                parsed.inspect_placement = true;
            }
            "--inspect-device-slice" => {
                parsed.inspect_device_slice = Some(next_value(&mut raw, "--inspect-device-slice")?);
            }
            "--device" | "--default-device-id" => {
                parsed.default_device_id = Some(next_value(&mut raw, &arg)?);
            }
            "--place-pedal" | "--place" => {
                let assignment = next_value(&mut raw, &arg)?;
                let (pedal_id, device_id) = parse_pedal_device_assignment(&assignment)?;
                if parsed
                    .pedal_devices
                    .insert(pedal_id.clone(), device_id)
                    .is_some()
                {
                    return Err(format!(
                        "duplicate runtime placement for pedal {pedal_id:?}"
                    ));
                }
            }
            "--duplicate-after" => {
                let assignment = next_value(&mut raw, "--duplicate-after")?;
                parsed
                    .duplicate_after
                    .push(parse_duplicate_after_assignment(&assignment)?);
            }
            "--chain" | "--source-chain" => {
                let chain = parse_source_chain(&next_value(&mut raw, &arg)?)?;
                if parsed.source_chain.replace(chain).is_some() {
                    return Err("--chain may only be supplied once".to_string());
                }
            }
            "--max-new-tokens" => {
                parsed.max_new_tokens = parse_next(&mut raw, "--max-new-tokens")?;
            }
            "--capacity" => {
                parsed.capacity = Some(parse_next(&mut raw, "--capacity")?);
            }
            "--cycle-ticks" => {
                parsed.cycle_ticks = parse_next(&mut raw, "--cycle-ticks")?;
            }
            "--max-scheduler-turns" => {
                parsed.max_scheduler_turns = parse_next(&mut raw, "--max-scheduler-turns")?;
            }
            "--no-special-tokens" => {
                parsed.add_special_tokens = false;
            }
            "--keep-special-tokens" => {
                parsed.skip_special_tokens = false;
            }
            "--generated-only" => {
                parsed.generated_only = true;
            }
            "--json" => {
                parsed.json = true;
            }
            _ => {
                return Err(format!("unknown argument {arg:?}\n\n{}", usage()));
            }
        }
    }

    if matches!(parsed.prompt.as_deref(), Some("")) {
        return Err("--prompt must not be empty".to_string());
    }
    let inspect_mode_count = usize::from(parsed.inspect_package)
        + usize::from(parsed.inspect_patch)
        + usize::from(parsed.inspect_placement)
        + usize::from(parsed.inspect_device_slice.is_some());
    if inspect_mode_count > 1 {
        return Err(
            "--inspect-package, --inspect-patch, --inspect-placement, and --inspect-device-slice are mutually exclusive"
                .to_string(),
        );
    }
    if matches!(parsed.inspect_device_slice.as_deref(), Some("")) {
        return Err("--inspect-device-slice must not be empty".to_string());
    }
    if matches!(parsed.default_device_id.as_deref(), Some("")) {
        return Err("--device must not be empty".to_string());
    }
    if parsed.max_new_tokens == 0 {
        return Err("--max-new-tokens must be at least 1".to_string());
    }
    if matches!(parsed.capacity, Some(0)) {
        return Err("--capacity must be at least 1".to_string());
    }
    if parsed.cycle_ticks == 0 {
        return Err("--cycle-ticks must be at least 1".to_string());
    }
    if parsed.max_scheduler_turns == 0 {
        return Err("--max-scheduler-turns must be at least 1".to_string());
    }

    Ok(parsed)
}

fn parse_pedal_device_assignment(raw: &str) -> Result<(String, String), String> {
    let (pedal_id, device_id) = raw
        .split_once('=')
        .ok_or_else(|| format!("invalid runtime placement {raw:?}; expected PEDAL_ID=DEVICE_ID"))?;
    let pedal_id = pedal_id.trim();
    let device_id = device_id.trim();
    if pedal_id.is_empty() {
        return Err(format!(
            "invalid runtime placement {raw:?}; pedal id must not be empty"
        ));
    }
    if device_id.is_empty() {
        return Err(format!(
            "invalid runtime placement {raw:?}; device id must not be empty"
        ));
    }
    Ok((pedal_id.to_string(), device_id.to_string()))
}

fn parse_duplicate_after_assignment(raw: &str) -> Result<(String, String), String> {
    let (after_instance_id, new_instance_id) = raw.split_once('=').ok_or_else(|| {
        format!("invalid runtime duplicate {raw:?}; expected AFTER_INSTANCE_ID=NEW_INSTANCE_ID")
    })?;
    let after_instance_id = after_instance_id.trim();
    let new_instance_id = new_instance_id.trim();
    if after_instance_id.is_empty() {
        return Err(format!(
            "invalid runtime duplicate {raw:?}; source instance id must not be empty"
        ));
    }
    if new_instance_id.is_empty() {
        return Err(format!(
            "invalid runtime duplicate {raw:?}; new instance id must not be empty"
        ));
    }
    Ok((after_instance_id.to_string(), new_instance_id.to_string()))
}

fn parse_source_chain(raw: &str) -> Result<Vec<(String, String)>, String> {
    let separator = if raw.contains("->") { "->" } else { "," };
    let mut chain = Vec::new();
    let mut instance_ids = std::collections::BTreeSet::new();

    for raw_item in raw.split(separator) {
        let raw_item = raw_item.trim();
        if raw_item.is_empty() {
            return Err(format!(
                "invalid runtime chain {raw:?}; chain items must not be empty"
            ));
        }
        let (instance_id, source_pedal_id) =
            if let Some((instance_id, source_pedal_id)) = raw_item.split_once('=') {
                (instance_id.trim(), source_pedal_id.trim())
            } else {
                (raw_item, raw_item)
            };
        if instance_id.is_empty() {
            return Err(format!(
                "invalid runtime chain item {raw_item:?}; instance id must not be empty"
            ));
        }
        if source_pedal_id.is_empty() {
            return Err(format!(
                "invalid runtime chain item {raw_item:?}; source pedal id must not be empty"
            ));
        }
        if !instance_ids.insert(instance_id.to_string()) {
            return Err(format!(
                "invalid runtime chain {raw:?}; duplicate instance id {instance_id:?}"
            ));
        }
        chain.push((instance_id.to_string(), source_pedal_id.to_string()));
    }

    if chain.is_empty() {
        return Err("runtime chain must contain at least one pedal".to_string());
    }

    Ok(chain)
}

fn parse_next<T: std::str::FromStr>(
    raw: &mut impl Iterator<Item = String>,
    flag: &str,
) -> Result<T, String>
where
    T::Err: std::fmt::Display,
{
    let value = next_value(raw, flag)?;
    value
        .parse::<T>()
        .map_err(|error| format!("invalid value {value:?} for {flag}: {error}"))
}

fn next_value(raw: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    raw.next()
        .ok_or_else(|| format!("{flag} requires a value\n\n{}", usage()))
}

fn print_text(text: &str) {
    print!("{text}");
    if !text.ends_with('\n') {
        println!();
    }
}

fn engine_stop_label(stop: VulkanResidentTokenEngineRunStopCondition) -> &'static str {
    match stop {
        VulkanResidentTokenEngineRunStopCondition::Idle => "idle",
        VulkanResidentTokenEngineRunStopCondition::SchedulerTurnBudget => "scheduler_turn_budget",
    }
}

fn print_usage() {
    println!("{}", usage());
}

fn usage() -> &'static str {
    "Usage: llmoop-runtime --package <COMPILED_PACKAGE.json> --prompt <TEXT> [OPTIONS]

Options:
  --package <PATH>           Compiled resident model package manifest. Required.
  --package-manifest <PATH>  Alias for --package.
  --prompt <TEXT>            External text event to inject into the resident stream. Required.
  --device <DEVICE_ID>       Default logical device for this runtime patch.
  --default-device-id <ID>   Alias for --device.
  --place-pedal <PEDAL=DEV>  Assign one runtime pedal instance to a logical device.
  --place <PEDAL=DEV>        Alias for --place-pedal.
  --chain <ITEM[,ITEM...]>    Runtime source chain. ITEM is SOURCE or INSTANCE=SOURCE.
  --duplicate-after <AFTER=NEW>
                             Duplicate runtime pedal instance AFTER with id NEW.
  --inspect-package          Summarize the compiled source pedal kit and available devices.
  --inspect-pedals           Alias for --inspect-package.
  --inspect-patch            Preview the effective runtime patch without mounting devices.
  --inspect-placement        Mount and summarize every logical device slice in the runtime patch.
  --inspect-device-slice <DEVICE_ID>
                             Mount and summarize only the runtime patch pedals assigned to DEVICE_ID.
  --max-new-tokens <N>       Public output tokens to emit after the prompt. Default: 4
  --capacity <N>             Override resident activation capacity selected from the package.
  --cycle-ticks <N>          Max runtime ticks per always-on cycle. Default: 4
  --max-scheduler-turns <N>  Max engine scheduler turns before stopping. Default: 1024
  --no-special-tokens        Do not add tokenizer special tokens to input text.
  --keep-special-tokens      Keep tokenizer special tokens in decoded output text.
  --generated-only           Print only newly generated text instead of prompt + generated text.
  --json                     Print a machine-readable run report.
  -h, --help                 Show this help.

Example:
  python -m llmoop --compile-model <MODEL_DIR>
  cargo run --manifest-path runtime-rs/Cargo.toml --features 'vulkan tokenizers' --bin llmoop-runtime -- --package packages/model_xxx/vulkan_resident_greedy_package.json --prompt Hello --max-new-tokens 4"
}
