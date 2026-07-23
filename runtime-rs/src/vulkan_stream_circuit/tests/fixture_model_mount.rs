#[test]
fn runtime_sampling_overrides_promote_compiled_greedy_without_recompilation() {
    let mut compiled = fixture_model_greedy_sampler_spec();
    compiled.top_k = 1;
    compiled.top_k_capacity = 256;
    compiled.scratch_byte_capacity = 128 * 256 * 8;

    let sampled = VulkanResidentSamplerRuntimeConfig {
        temperature: Some(0.7),
        top_k: Some(40),
        top_p: Some(0.95),
        min_p: Some(0.02),
        presence_penalty: Some(0.5),
        repetition_penalty: Some(1.1),
    }
    .apply_to(&compiled)
    .unwrap();

    assert_eq!(sampled.method, "temperature_top_k_top_p");
    assert_eq!(
        sampled.sampler_id,
        "runtime_temperature_top_k_top_p_sampler"
    );
    assert_eq!(sampled.temperature, 0.7);
    assert_eq!(sampled.top_k, 40);
    assert_eq!(sampled.top_p, 0.95);
    assert_eq!(sampled.min_p, 0.02);
    assert_eq!(sampled.presence_penalty, 0.5);
    assert_eq!(sampled.repetition_penalty, 1.1);
    assert!(sampled.runtime_parameterized);
    assert!(sampler_kernel_role_matches(
        "runtime_partition_top_k",
        true,
        sampled.method.as_str(),
    ));
    assert!(!sampler_kernel_role_matches(
        "runtime_sample_logits",
        true,
        sampled.method.as_str(),
    ));

    let penalized_greedy = VulkanResidentSamplerRuntimeConfig {
        presence_penalty: Some(0.25),
        ..VulkanResidentSamplerRuntimeConfig::default()
    }
    .apply_to(&compiled)
    .unwrap();
    assert_eq!(penalized_greedy.method, "greedy");
    assert!(sampler_kernel_role_matches(
        "runtime_sample_logits",
        true,
        penalized_greedy.method.as_str(),
    ));

    let error = VulkanResidentSamplerRuntimeConfig {
        top_k: Some(257),
        ..VulkanResidentSamplerRuntimeConfig::default()
    }
    .apply_to(&compiled)
    .unwrap_err();
    assert!(error.to_string().contains("capacity 256"));
}

fn fixture_model_resident_greedy_model(
    device: &VulkanComputeDevice,
    capacity: usize,
) -> Result<VulkanResidentModelPackage, VulkanResidentTokenModelPackageError> {
    VulkanResidentModelPackage::from_manifest_file_with_capacity(
        device,
        fixture_model_package_manifest_path(),
        Some(capacity),
    )
}

#[cfg(feature = "tokenizers")]
fn fixture_model_model_dir_path() -> Option<PathBuf> {
    std::env::var("NERVE_TEST_MODEL_DIR")
        .ok()
        .map(PathBuf::from)
}

#[cfg(feature = "tokenizers")]
fn fixture_model_tokenizer_codec_or_skip(
    test_name: &str,
) -> Option<VulkanResidentHfTokenizerTextCodec> {
    let Some(model_dir) = fixture_model_model_dir_path() else {
        eprintln!("skipping {test_name}: set NERVE_TEST_MODEL_DIR for tokenizer tests");
        return None;
    };
    if !model_dir.join("tokenizer.json").is_file() {
        eprintln!(
            "skipping {test_name}: {:?} does not contain tokenizer.json",
            model_dir
        );
        return None;
    }
    Some(VulkanResidentHfTokenizerTextCodec::from_model_dir(model_dir).unwrap())
}

fn mount_fixture_model_single_device_stream_circuit(
    device: &VulkanComputeDevice,
) -> (
    TensorIndex,
    VulkanMountedPlacedStreamCircuit,
    VulkanReusableKernelArtifactManifest,
    VulkanMountedPlacedBoundDispatchPlan,
) {
    mount_fixture_model_single_device_stream_circuit_with_capacity(device, 4)
}

fn mount_fixture_model_single_device_stream_circuit_with_capacity(
    device: &VulkanComputeDevice,
    dynamic_state_capacity_activations: usize,
) -> (
    TensorIndex,
    VulkanMountedPlacedStreamCircuit,
    VulkanReusableKernelArtifactManifest,
    VulkanMountedPlacedBoundDispatchPlan,
) {
    let graph = fixture_model_execution_graph();
    let tensor_index = TensorIndex::from_json_file(fixture_model_tensor_index_path()).unwrap();
    let execution_plan =
        StreamCircuitExecutionPlan::from_graph_with_tensor_index(&graph, &tensor_index).unwrap();
    let resource_plan =
        StreamCircuitResourcePlan::from_graph_and_plan(&graph, &execution_plan).unwrap();
    let placement_spec = StreamCircuitPlacementSpec::new("gpu0");
    let placement_plan = graph.placement_plan(&placement_spec).unwrap();
    let resident = VulkanPlacedStreamCircuitResidentPlan::from_resource_plan_for_device(
        &resource_plan,
        &placement_plan,
        "gpu0",
        Some(&tensor_index),
        Some(2),
    )
    .unwrap();
    let placed_plan =
        VulkanPlacedStreamCircuitPlan::from_plans(&execution_plan, &resource_plan, resident)
            .unwrap();
    let mounted = VulkanMountedPlacedStreamCircuit::from_placed_plan(
        device,
        placed_plan,
        dynamic_state_capacity_activations,
    )
    .unwrap();
    let manifest = VulkanReusableKernelArtifactManifest::new(
        mounted
            .placed_plan
            .reusable_kernel_plan
            .families
            .iter()
            .map(|family| {
                VulkanReusableKernelArtifact::from_family(
                    family,
                    format!("kernels/{}.spv", family.family_id),
                )
            })
            .collect(),
    );
    let mounted_bound = mounted
        .mounted_placed_bound_dispatch_plan(&manifest)
        .unwrap();
    (tensor_index, mounted, manifest, mounted_bound)
}

fn load_layer_00_parameters(
    mounted: &VulkanMountedPlacedStreamCircuit,
    tensor_index: &TensorIndex,
) {
    load_fixture_model_conv_layer_parameters(mounted, tensor_index, 0);
}

#[test]
fn package_pedal_executions_must_match_mounted_dispatch_order() {
    let device = match VulkanComputeDevice::new() {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping package pedal execution validation: {error}");
            return;
        }
    };
    let mut runtime_model = fixture_model_runtime_model();
    let manifest_path = fixture_model_package_manifest_path();
    let manifest_dir = manifest_path.parent().unwrap();
    let tensor_index_path =
        resolve_resident_model_package_path(manifest_dir, &runtime_model.package.tensor_index_path);
    let default_device_id = runtime_model.placement.default_device_id.clone();
    let (_tensor_index, _resource_plan, placed_plan) =
        plan_resident_package_single_device_stream_circuit(
            &default_device_id,
            &runtime_model.placement,
            &runtime_model.circuit_graph,
            manifest_dir,
            &tensor_index_path,
            runtime_model.package.activation_element_bytes,
        )
        .unwrap();
    let mounted =
        VulkanMountedPlacedStreamCircuit::from_placed_plan(&device, placed_plan, 4).unwrap();
    let kernel_manifest = resident_package_reusable_kernel_manifest(&mounted.placed_plan);
    let mounted_bound = mounted
        .mounted_placed_bound_dispatch_plan(&kernel_manifest)
        .unwrap();

    validate_pedal_executions_against_mounted_dispatches(
        &runtime_model.package.package_id,
        &runtime_model.pedal_executions,
        &mounted_bound,
    )
    .unwrap();

    runtime_model.pedal_executions[0].kernels.swap(0, 1);
    let error = validate_pedal_executions_against_mounted_dispatches(
        &runtime_model.package.package_id,
        &runtime_model.pedal_executions,
        &mounted_bound,
    )
    .unwrap_err();
    assert!(error.to_string().contains(
        "declares pedal layer_00 kernel conv_in_projection with execution_index 1, expected 0"
    ));
}

#[test]
fn single_device_package_rejects_remote_pedal_placement() {
    let manifest = fixture_model_package_manifest();
    let placement = StreamCircuitPlacementSpec::new(RUNTIME_DEFAULT_LOGICAL_DEVICE_ID)
        .with_pedal_device("layer_00", "gpu1");
    let manifest_path = fixture_model_package_manifest_path();
    let manifest_dir = manifest_path.parent().unwrap();
    let tensor_index_path =
        resolve_resident_model_package_path(manifest_dir, &manifest.tensor_index_path);
    let default_device_id = placement.default_device_id.clone();

    let error = plan_resident_package_single_device_stream_circuit(
        &default_device_id,
        &placement,
        &manifest.circuit_graph,
        manifest_dir,
        &tensor_index_path,
        manifest.activation_element_bytes,
    )
    .unwrap_err();
    assert!(error.to_string().contains(
            "single-device resident package for \"runtime_default\" cannot host remote pedals: layer_00@gpu1"
        ));
}

#[test]
fn placed_package_planning_reuses_one_tensor_index_across_device_slices() {
    let graph = fixture_model_execution_graph();
    let package_graph = VulkanResidentPackageCircuitGraph {
        wiring: graph.index.graph.wiring.clone(),
        cables: graph.index.graph.cables.clone(),
        boundary: graph.index.graph.boundary.clone(),
        architecture: graph.index.architecture.clone(),
        dimensions: graph.index.dimensions.clone(),
        input_transducer: graph.index.graph.input_transducer.clone(),
        output_transducer: graph.index.graph.output_transducer.clone(),
        pedals: graph
            .circuits
            .iter()
            .map(|artifact| VulkanResidentPackagePedalCircuit {
                pedal_id: artifact.pedal.id.clone(),
                operator_type: artifact.pedal.operator_type.clone(),
                runtime_role: artifact.pedal.runtime_role,
                implementation: artifact.pedal.implementation.clone(),
                behavioral_role: artifact.pedal.behavioral_role.clone(),
                circuit: artifact.circuit.clone(),
                params: artifact.params.clone(),
                state: artifact.state.clone(),
            })
            .collect(),
    };
    let placement = StreamCircuitPlacementSpec::new("gpu0")
        .with_pedal_device("layer_02", "gpu1")
        .with_pedal_device("layer_05", "gpu1");
    let tensor_index = TensorIndex::from_json_file(fixture_model_tensor_index_path()).unwrap();

    let (_, _, gpu0) = plan_resident_package_placed_stream_circuit_with_tensor_index(
        "gpu0",
        &placement,
        &package_graph,
        &graph.artifact_root,
        &tensor_index,
        Some(2),
    )
    .unwrap();
    let (_, _, gpu1) = plan_resident_package_placed_stream_circuit_with_tensor_index(
        "gpu1",
        &placement,
        &package_graph,
        &graph.artifact_root,
        &tensor_index,
        Some(2),
    )
    .unwrap();

    let gpu0_pedals = gpu0
        .binding_plan
        .circuits
        .iter()
        .map(|circuit| circuit.pedal_id.as_str())
        .collect::<BTreeSet<_>>();
    let gpu1_pedals = gpu1
        .binding_plan
        .circuits
        .iter()
        .map(|circuit| circuit.pedal_id.as_str())
        .collect::<BTreeSet<_>>();
    assert!(gpu0_pedals.is_disjoint(&gpu1_pedals));
    assert_eq!(gpu1_pedals, BTreeSet::from(["layer_02", "layer_05"]));
    assert_eq!(gpu0_pedals.len() + gpu1_pedals.len(), graph.circuits.len());
}

#[test]
fn package_device_slice_mounts_only_pedals_assigned_to_device() {
    let device = match VulkanComputeDevice::new() {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping package device slice mount: {error}");
            return;
        }
    };
    let runtime_model = fixture_model_runtime_model_with_placement(
        StreamCircuitPlacementSpec::new("gpu0")
            .with_pedal_device("layer_01", "cpu0")
            .with_pedal_device("layer_02", "gpu1")
            .with_pedal_device("layer_03", "lan:worker-a"),
    );
    let manifest_path = fixture_model_package_manifest_path();
    let manifest_dir = manifest_path.parent().unwrap();

    let slice = VulkanResidentModelPackageDeviceSlice::from_runtime_model_for_device(
        &device,
        manifest_dir,
        runtime_model,
        "gpu1",
        Some(4),
    )
    .unwrap();

    assert_eq!(slice.device_id, "gpu1");
    assert_eq!(slice.hosted_pedal_count, 1);
    assert_eq!(slice.incoming_cable_count, 1);
    assert_eq!(slice.outgoing_cable_count, 1);
    assert_eq!(slice.permanent_parameter_count, 11);
    assert!(slice.permanent_parameter_bytes > 0);
    assert!(slice.reusable_kernel_word_count > 0);
    assert!(!slice.loaded_manifest().artifacts.is_empty());

    let mounted = slice.create_mounted_stream_circuit(&device).unwrap();
    let reusable_manifest = resident_package_reusable_kernel_manifest(&mounted.placed_plan);
    let mounted_bound = mounted
        .mounted_placed_bound_dispatch_plan(&reusable_manifest)
        .unwrap();

    assert_eq!(mounted.device_id(), "gpu1");
    assert_eq!(mounted.placed_plan.binding_plan.circuits.len(), 1);
    assert_eq!(mounted_bound.dispatches.len(), 16);
    assert!(
        mounted_bound
            .dispatch("layer_02", "kv_memory_append")
            .is_some()
    );
    assert!(
        mounted_bound
            .dispatch("layer_00", "operator_norm")
            .is_none()
    );
    assert_eq!(mounted_bound.model_boundary_descriptor_count, 0);
    assert_eq!(mounted_bound.incoming_cable_descriptor_count, 2);
    assert_eq!(mounted_bound.outgoing_cable_descriptor_count, 1);

    let tick_plan = mounted.stream_tick_plan(&reusable_manifest).unwrap();
    assert_eq!(tick_plan.device_id, "gpu1");
    assert_eq!(tick_plan.stage_count, 18);
    assert_eq!(tick_plan.receive_stage_count, 1);
    assert_eq!(tick_plan.dispatch_stage_count, 16);
    assert_eq!(tick_plan.publish_stage_count, 1);
    let tick_run = mounted.advance_stream_tick(&reusable_manifest, 7).unwrap();
    assert_eq!(
        tick_run.status,
        VulkanMountedPlacedStreamTickRunStatus::Blocked {
            stage_index: 0,
            reason: VulkanMountedPlacedStreamTickBlockReason::CableReceiveTransportUnavailable,
        }
    );
}

fn fixture_model_embedding_row_bytes(tensor_index: &TensorIndex, token_id: u32) -> Vec<u8> {
    let metadata = tensor_index
        .tensors
        .get(FIXTURE_MODEL_EMBED_TOKENS_TENSOR)
        .unwrap();
    let offsets = metadata.data_offsets.as_ref().unwrap();
    let data_start = offsets[0];
    let row_offset = usize::try_from(token_id).unwrap() * FIXTURE_MODEL_FRAME_BYTES;
    let absolute_tensor_offset = data_start + row_offset;
    let source_file = metadata.source_file.as_ref().unwrap();
    let mut file = fs::File::open(source_file).unwrap();
    let mut header_len_bytes = [0u8; 8];
    file.read_exact(&mut header_len_bytes).unwrap();
    let data_base = 8 + u64::from_le_bytes(header_len_bytes);
    file.seek(SeekFrom::Start(
        data_base + u64::try_from(absolute_tensor_offset).unwrap(),
    ))
    .unwrap();
    let mut bytes = vec![0u8; FIXTURE_MODEL_FRAME_BYTES];
    file.read_exact(&mut bytes).unwrap();
    bytes
}

fn load_fixture_model_transducer_parameter_buffers(
    device: &VulkanComputeDevice,
    tensor_index: &TensorIndex,
) -> VulkanPermanentParameterBuffers {
    let graph = fixture_model_execution_graph();
    let execution_plan =
        StreamCircuitExecutionPlan::from_graph_with_tensor_index(&graph, tensor_index).unwrap();
    let resource_plan =
        StreamCircuitResourcePlan::from_graph_and_plan(&graph, &execution_plan).unwrap();
    let transducer_parameter_plan = VulkanPermanentParameterBufferPlan::from_transducer_parameters(
        "gpu0",
        &resource_plan,
        Some(tensor_index),
    )
    .unwrap();
    assert_eq!(transducer_parameter_plan.parameter_count, 2);
    assert_eq!(
        transducer_parameter_plan.total_byte_capacity,
        Some(134_219_776)
    );
    assert!(transducer_parameter_plan.unresolved_tensors.is_empty());
    let transducer_parameter_buffers = transducer_parameter_plan.allocate_buffers(device).unwrap();
    let loaded = transducer_parameter_buffers
        .load_from_tensor_index(tensor_index)
        .unwrap();
    assert_eq!(loaded.parameter_count, 2);
    assert_eq!(loaded.loaded_count, 2);
    assert_eq!(loaded.total_bytes_loaded, 134_219_776);
    transducer_parameter_buffers
}

#[test]
fn transducer_parameter_plans_are_isolated_by_host_boundary() {
    let tensor_index = TensorIndex::from_json_file(fixture_model_tensor_index_path()).unwrap();
    let graph = fixture_model_execution_graph();
    let execution_plan =
        StreamCircuitExecutionPlan::from_graph_with_tensor_index(&graph, &tensor_index).unwrap();
    let resource_plan =
        StreamCircuitResourcePlan::from_graph_and_plan(&graph, &execution_plan).unwrap();

    let input = VulkanPermanentParameterBufferPlan::from_transducer_parameters_for(
        "gpu0",
        &resource_plan,
        Some(&tensor_index),
        "input_transducer",
    )
    .unwrap();
    let output = VulkanPermanentParameterBufferPlan::from_transducer_parameters_for(
        "gpu1",
        &resource_plan,
        Some(&tensor_index),
        "output_transducer",
    )
    .unwrap();

    let expected_input_tensors = resource_plan
        .transducer_parameters
        .iter()
        .filter(|parameter| {
            parameter
                .uses
                .iter()
                .any(|parameter_use| parameter_use.circuit_id == "input_transducer")
        })
        .map(|parameter| parameter.tensor.as_str())
        .collect::<BTreeSet<_>>();
    let expected_output_tensors = resource_plan
        .transducer_parameters
        .iter()
        .filter(|parameter| {
            parameter
                .uses
                .iter()
                .any(|parameter_use| parameter_use.circuit_id == "output_transducer")
        })
        .map(|parameter| parameter.tensor.as_str())
        .collect::<BTreeSet<_>>();

    assert_eq!(
        input
            .parameters
            .iter()
            .map(|parameter| parameter.tensor.as_str())
            .collect::<BTreeSet<_>>(),
        expected_input_tensors
    );
    assert_eq!(
        output
            .parameters
            .iter()
            .map(|parameter| parameter.tensor.as_str())
            .collect::<BTreeSet<_>>(),
        expected_output_tensors
    );
    assert!(input.parameter_count > 0);
    assert!(output.parameter_count > 0);
}

