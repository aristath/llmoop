fn mounted_single_device_stream_circuit_binds_local_cable_buffers() {
    let device = match VulkanComputeDevice::new() {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping single-device placed Vulkan stream-circuit mount: {error}");
            return;
        }
    };
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

    assert_eq!(resident.hosted_pedal_ids.len(), 14);
    assert_eq!(resident.local_cables.len(), 13);
    assert_eq!(resident.incoming_cables.len(), 0);
    assert_eq!(resident.outgoing_cables.len(), 0);

    let placed_plan =
        VulkanPlacedStreamCircuitPlan::from_plans(&execution_plan, &resource_plan, resident)
            .unwrap();
    let mounted =
        VulkanMountedPlacedStreamCircuit::from_placed_plan(&device, placed_plan, 4).unwrap();

    assert_eq!(mounted.device_id(), "gpu0");
    assert!(!mounted.can_execute());
    assert_eq!(mounted.placed_plan.binding_plan.circuits.len(), 14);
    assert_eq!(
        mounted.placed_plan.dispatch_plan.total_dispatch_count(),
        242
    );
    assert_eq!(mounted.parameter_buffers.plan.device_id, "gpu0");
    assert_eq!(mounted.parameter_buffers.plan.parameter_count, 130);
    assert_eq!(
        mounted.parameter_buffers.plan.total_byte_capacity,
        Some(325_166_592)
    );
    assert!(mounted.parameter_buffers.plan.unresolved_tensors.is_empty());
    assert_eq!(mounted.parameter_buffers.total_byte_capacity, 325_166_592);
    let operator_norm_weight = mounted
        .parameter_buffers
        .parameter_buffer("model.layers.0.operator_norm.weight")
        .unwrap();
    assert_eq!(
        operator_norm_weight.parameter.dtype.as_deref(),
        Some("BF16")
    );
    assert_eq!(operator_norm_weight.parameter.shape, Some(vec![1024]));
    assert_eq!(operator_norm_weight.byte_capacity, 2_048);
    operator_norm_weight
        .buffer
        .write_bytes(&[21, 22, 23, 24])
        .unwrap();
    assert_eq!(
        operator_norm_weight.buffer.read_bytes(4).unwrap(),
        vec![21, 22, 23, 24]
    );
    let operator_norm_metadata = tensor_index
        .tensors
        .get("model.layers.0.operator_norm.weight")
        .unwrap();
    let operator_norm_source_available = operator_norm_metadata
        .source_file
        .as_ref()
        .map(|source_file| Path::new(source_file).exists())
        .unwrap_or(false);
    if operator_norm_source_available {
        let loaded_weight = mounted
            .parameter_buffers
            .load_parameter_from_tensor_index(&tensor_index, "model.layers.0.operator_norm.weight")
            .unwrap();
        assert_eq!(loaded_weight.tensor, "model.layers.0.operator_norm.weight");
        assert_eq!(loaded_weight.data_start, 158_345_216);
        assert_eq!(loaded_weight.data_end, 158_347_264);
        assert_eq!(loaded_weight.byte_count, 2_048);
        assert_eq!(
            operator_norm_weight.buffer.read_bytes(16).unwrap(),
            vec![
                0xc6, 0x3e, 0xb9, 0x3e, 0xba, 0x3e, 0xba, 0x3e, 0xc2, 0x3e, 0xba, 0x3e, 0xbe, 0x3e,
                0x12, 0x3f,
            ]
        );
    } else {
        eprintln!(
            "skipping real safetensors parameter load: source file for model.layers.0.operator_norm.weight is unavailable"
        );
    }
    assert_eq!(mounted.boundary_io.plan.device_id, "gpu0");
    assert_eq!(mounted.boundary_io.plan.input_count, 1);
    assert_eq!(mounted.boundary_io.plan.output_count, 1);
    assert_eq!(mounted.boundary_io.plan.total_buffer_count, 2);
    assert_eq!(mounted.boundary_io.plan.total_byte_capacity, Some(4_096));
    assert_eq!(mounted.boundary_io.total_byte_capacity, 4_096);
    let model_input = mounted.boundary_io.input_buffer("input_frame").unwrap();
    assert_eq!(model_input.boundary.pedal_id, "layer_00");
    assert_eq!(model_input.boundary.port_id, "input_frame");
    assert_eq!(model_input.boundary.shape, vec![1024]);
    assert_eq!(model_input.byte_capacity, 2_048);
    model_input.buffer.write_bytes(&[1, 2, 3, 4]).unwrap();
    assert_eq!(model_input.buffer.read_bytes(4).unwrap(), vec![1, 2, 3, 4]);
    let model_output = mounted.boundary_io.output_buffer("output_frame").unwrap();
    assert_eq!(model_output.boundary.pedal_id, "layer_13");
    assert_eq!(model_output.boundary.port_id, "output_frame");
    assert_eq!(model_output.byte_capacity, 2_048);
    assert_eq!(mounted.cable_io.plan.local_cable_count, 13);
    assert_eq!(mounted.cable_io.plan.total_endpoint_count, 0);
    assert_eq!(mounted.cable_io.plan.total_buffer_count, 13);
    assert_eq!(mounted.cable_io.plan.total_byte_capacity, Some(26_624));
    assert_eq!(mounted.cable_io.local_buffers.len(), 13);
    assert_eq!(mounted.cable_io.incoming_buffers.len(), 0);
    assert_eq!(mounted.cable_io.outgoing_buffers.len(), 0);
    assert_eq!(mounted.cable_io.total_byte_capacity, 26_624);
    let first_local_cable = mounted.cable_io.local_cable_buffer(0).unwrap();
    assert_eq!(first_local_cable.cable.cable_id, "cable_0_local");
    assert_eq!(first_local_cable.cable.source_pedal_id, "layer_00");
    assert_eq!(first_local_cable.cable.destination_pedal_id, "layer_01");
    assert_eq!(first_local_cable.cable.byte_capacity, Some(2_048));
    assert_eq!(first_local_cable.byte_capacity, 2_048);
    assert_eq!(first_local_cable.buffer.byte_capacity(), 2_048);
    first_local_cable
        .buffer
        .write_bytes(&[11, 12, 13, 14])
        .unwrap();
    assert_eq!(
        first_local_cable.buffer.read_bytes(4).unwrap(),
        vec![11, 12, 13, 14]
    );

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
    let placed_bound = mounted.placed_bound_dispatch_plan(&manifest).unwrap();
    assert_eq!(placed_bound.device_id, "gpu0");
    assert_eq!(placed_bound.dispatches.len(), 242);
    assert_eq!(placed_bound.model_boundary_descriptor_count, 3);
    assert_eq!(placed_bound.local_cable_descriptor_count, 39);
    assert_eq!(placed_bound.incoming_cable_descriptor_count, 0);
    assert_eq!(placed_bound.outgoing_cable_descriptor_count, 0);

    let mounted_bound = mounted
        .mounted_placed_bound_dispatch_plan(&manifest)
        .unwrap();
    assert_eq!(mounted_bound.device_id, "gpu0");
    assert_eq!(mounted_bound.dispatches.len(), 242);
    assert_eq!(
        mounted_bound.total_descriptor_count,
        placed_bound.total_descriptor_count
    );
    assert_eq!(mounted_bound.model_boundary_descriptor_count, 3);
    assert_eq!(mounted_bound.local_cable_descriptor_count, 39);
    assert_eq!(mounted_bound.cable_endpoint_descriptor_count, 0);
    assert_eq!(mounted_bound.incoming_cable_descriptor_count, 0);
    assert_eq!(mounted_bound.outgoing_cable_descriptor_count, 0);

    let tick_plan = mounted.stream_tick_plan(&manifest).unwrap();
    assert_eq!(tick_plan.device_id, "gpu0");
    assert!(!tick_plan.can_execute);
    assert_eq!(tick_plan.stage_count, 242);
    assert_eq!(tick_plan.receive_stage_count, 0);
    assert_eq!(tick_plan.dispatch_stage_count, 242);
    assert_eq!(tick_plan.publish_stage_count, 0);
    assert_eq!(tick_plan.local_cable_read_count, 26);
    assert_eq!(tick_plan.local_cable_write_count, 13);
    assert_eq!(tick_plan.incoming_cable_read_count, 0);
    assert_eq!(tick_plan.outgoing_cable_write_count, 0);
    assert_eq!(tick_plan.model_input_read_count, 2);
    assert_eq!(tick_plan.model_output_write_count, 1);
    assert_eq!(
        tick_plan.stages[0],
        VulkanMountedPlacedStreamTickStage::Dispatch {
            stage_index: 0,
            dispatch: VulkanMountedPlacedStreamTickDispatch {
                dispatch_index: 0,
                kernel_id: "layer_00.operator_norm".to_string(),
                pedal_id: "layer_00".to_string(),
                node_id: "operator_norm".to_string(),
                op: "rms_norm".to_string(),
                descriptor_count: mounted_bound
                    .dispatch("layer_00", "operator_norm")
                    .unwrap()
                    .descriptors
                    .len(),
                resident_descriptor_count: 2,
                reads: vec![VulkanMountedPlacedStreamTickIo::ModelSignal {
                    signal_id: "input_frame".to_string(),
                }],
                writes: vec![],
            },
        }
    );
    let tick_run = mounted.advance_stream_tick(&manifest, 42).unwrap();
    assert_eq!(tick_run.device_id, "gpu0");
    assert_eq!(tick_run.stream_tick, 42);
    assert!(!tick_run.can_execute);
    assert_eq!(tick_run.planned_stage_count, 242);
    assert_eq!(tick_run.attempted_stage_count, 1);
    assert_eq!(tick_run.completed_stage_count, 0);
    assert_eq!(tick_run.pending_stage_count, 241);
    assert_eq!(
        tick_run.status,
        VulkanMountedPlacedStreamTickRunStatus::Blocked {
            stage_index: 0,
            reason: VulkanMountedPlacedStreamTickBlockReason::KernelDispatchUnavailable,
        }
    );
    assert_eq!(tick_run.stages[0].stage, tick_plan.stages[0]);
    assert_eq!(
        tick_run.stages[0].status,
        VulkanMountedPlacedStreamTickStageStatus::Blocked {
            reason: VulkanMountedPlacedStreamTickBlockReason::KernelDispatchUnavailable,
        }
    );
    assert_eq!(
        tick_run.stages[1].status,
        VulkanMountedPlacedStreamTickStageStatus::Pending
    );
    let operator_norm_dispatch = mounted_bound.dispatch("layer_00", "operator_norm").unwrap();
    let operator_norm_family_id = operator_norm_dispatch.reusable_family_id.as_str();
    let operator_norm_bindings = mounted
        .resident_kernel_buffer_bindings_for_bound_dispatch(operator_norm_dispatch)
        .unwrap();
    assert_eq!(
        operator_norm_bindings.len(),
        operator_norm_dispatch.descriptors.len()
    );
    assert_eq!(operator_norm_bindings[0].binding, 0);
    assert_eq!(operator_norm_bindings[0].byte_len, 2_048);
    assert_eq!(operator_norm_bindings[1].binding, 1);
    assert_eq!(operator_norm_bindings[1].byte_len, 5_120);
    assert_eq!(operator_norm_bindings[2].binding, 2);
    assert_eq!(operator_norm_bindings[2].byte_len, 2_048);

    let empty_loaded_manifest = VulkanLoadedReusableKernelArtifactManifest {
        schema: VULKAN_REUSABLE_KERNEL_ARTIFACT_MANIFEST_SCHEMA.to_string(),
        backend_id: VULKAN_STREAM_CIRCUIT_BACKEND_ID.to_string(),
        artifacts: Vec::new(),
        total_word_count: 0,
    };
    let empty_readiness = mounted
        .resident_kernel_dispatch_readiness_plan(&manifest, &empty_loaded_manifest)
        .unwrap();
    assert_eq!(empty_readiness.device_id, "gpu0");
    assert_eq!(empty_readiness.dispatch_count, 242);
    assert_eq!(empty_readiness.instantiable_count, 0);
    assert_eq!(empty_readiness.blocked_count, 242);
    assert_eq!(empty_readiness.missing_loaded_artifact_count, 242);
    assert_eq!(empty_readiness.descriptor_binding_blocked_count, 0);
    assert_eq!(empty_readiness.push_constant_blocked_count, 0);
    assert_eq!(empty_readiness.instantiable_descriptor_count, 0);
    assert!(matches!(
        empty_readiness
            .dispatch("layer_00", "operator_norm")
            .unwrap()
            .status,
        VulkanMountedPlacedResidentKernelDispatchStatus::Blocked {
            error:
                VulkanMountedPlacedResidentKernelDispatchError::MissingLoadedArtifact {
                    dispatch_index: 0,
                    ref family_id,
                },
        } if family_id == operator_norm_family_id
    ));

    let rms_norm_family = mounted
        .placed_plan
        .reusable_kernel_plan
        .family(operator_norm_family_id)
        .unwrap();
    assert_eq!(rms_norm_family.op, "rms_norm");
    assert!(operator_norm_family_id.starts_with("rms_norm."));
    let rms_norm_loaded_manifest = VulkanLoadedReusableKernelArtifactManifest {
        schema: VULKAN_REUSABLE_KERNEL_ARTIFACT_MANIFEST_SCHEMA.to_string(),
        backend_id: VULKAN_STREAM_CIRCUIT_BACKEND_ID.to_string(),
        total_word_count: 2,
        artifacts: vec![VulkanLoadedReusableKernelArtifact {
            artifact: VulkanReusableKernelArtifact::from_family(
                rms_norm_family,
                format!("kernels/{operator_norm_family_id}.spv"),
            ),
            resolved_path: PathBuf::from(format!("kernels/{operator_norm_family_id}.spv")),
            words: vec![0x0723_0203, 0],
        }],
    };
    let rms_norm_readiness = mounted
        .resident_kernel_dispatch_readiness_plan(&manifest, &rms_norm_loaded_manifest)
        .unwrap();
    assert_eq!(
        rms_norm_readiness.instantiable_count,
        rms_norm_family.command_refs.len()
    );
    assert_eq!(
        rms_norm_readiness.blocked_count,
        rms_norm_readiness.dispatch_count - rms_norm_family.command_refs.len()
    );
    assert_eq!(
        rms_norm_readiness.missing_loaded_artifact_count,
        rms_norm_readiness.blocked_count
    );
    assert_eq!(rms_norm_readiness.descriptor_binding_blocked_count, 0);
    assert_eq!(rms_norm_readiness.push_constant_blocked_count, 0);
    assert!(matches!(
        rms_norm_readiness
            .dispatch("layer_00", "operator_norm")
            .unwrap()
            .status,
        VulkanMountedPlacedResidentKernelDispatchStatus::Instantiable {
            descriptor_count: 3,
            workgroup_count_x: 1,
            local_size_x: DEFAULT_COMPUTE_LOCAL_SIZE_X,
            push_constant_byte_count: 0,
        }
    ));
    if operator_norm_source_available {
        if let Some(spirv_words) = crate::vulkan_compute::compile_test_shader_words_from_source(
            "rms_norm_bf16_serial.comp",
        ) {
            let rms_norm_kernel_manifest = VulkanLoadedReusableKernelArtifactManifest {
                schema: VULKAN_REUSABLE_KERNEL_ARTIFACT_MANIFEST_SCHEMA.to_string(),
                backend_id: VULKAN_STREAM_CIRCUIT_BACKEND_ID.to_string(),
                total_word_count: spirv_words.len(),
                artifacts: vec![VulkanLoadedReusableKernelArtifact {
                    artifact: VulkanReusableKernelArtifact::from_family(
                        rms_norm_family,
                        format!("kernels/{operator_norm_family_id}.spv"),
                    ),
                    resolved_path: PathBuf::from(format!("kernels/{operator_norm_family_id}.spv")),
                    words: spirv_words,
                }],
            };
            let resident_dispatch = mounted
                .create_resident_kernel_dispatch_for_bound_dispatch(
                    &device,
                    operator_norm_dispatch,
                    &rms_norm_kernel_manifest,
                )
                .unwrap();
            let mut input_frame = Vec::with_capacity(2_048);
            for _ in 0..1024 {
                input_frame.extend_from_slice(&[0x80, 0x3f]);
            }
            model_input.buffer.write_bytes(&input_frame).unwrap();

            device
                .run_resident_kernel_dispatch(&resident_dispatch, &[])
                .unwrap();

            assert_eq!(
                operator_norm_bindings[1].buffer.read_bytes(16).unwrap(),
                vec![
                    0xc6, 0x3e, 0xb9, 0x3e, 0xba, 0x3e, 0xba, 0x3e, 0xc2, 0x3e, 0xba, 0x3e, 0xbe,
                    0x3e, 0x12, 0x3f,
                ]
            );

            mounted
                .parameter_buffers
                .load_parameter_from_tensor_index(
                    &tensor_index,
                    "model.layers.0.conv.in_proj.weight",
                )
                .unwrap();
            if let Some(linear_spirv_words) =
                crate::vulkan_compute::compile_test_shader_words_from_source(
                    "linear_bf16_1024x3072.comp",
                )
            {
                let conv_in_dispatch = mounted_bound
                    .dispatch("layer_00", "conv_in_projection")
                    .unwrap();
                let conv_in_bindings = mounted
                    .resident_kernel_buffer_bindings_for_bound_dispatch(conv_in_dispatch)
                    .unwrap();
                assert_eq!(conv_in_bindings[0].byte_len, 5_120);
                assert_eq!(conv_in_bindings[1].byte_len, 6_144);
                assert_eq!(conv_in_bindings[2].byte_len, 6_291_456);
                let linear_family = mounted
                    .placed_plan
                    .reusable_kernel_plan
                    .family(&conv_in_dispatch.reusable_family_id)
                    .unwrap();
                assert_eq!(linear_family.op, "linear");
                assert_eq!(linear_family.command_refs.len(), 8);
                let linear_artifact_path = artifact_path_for_family(linear_family);
                let linear_kernel_manifest = VulkanLoadedReusableKernelArtifactManifest {
                    schema: VULKAN_REUSABLE_KERNEL_ARTIFACT_MANIFEST_SCHEMA.to_string(),
                    backend_id: VULKAN_STREAM_CIRCUIT_BACKEND_ID.to_string(),
                    total_word_count: linear_spirv_words.len(),
                    artifacts: vec![VulkanLoadedReusableKernelArtifact {
                        artifact: VulkanReusableKernelArtifact::from_family(
                            linear_family,
                            linear_artifact_path.clone(),
                        )
                        .with_workgroup_count_x(1_536),
                        resolved_path: PathBuf::from(linear_artifact_path),
                        words: linear_spirv_words,
                    }],
                };
                let linear_dispatch = mounted
                    .create_resident_kernel_dispatch_for_bound_dispatch(
                        &device,
                        conv_in_dispatch,
                        &linear_kernel_manifest,
                    )
                    .unwrap();
                assert_eq!(linear_dispatch.workgroup_count_x(), 1_536);

                device
                    .run_resident_kernel_dispatch(&linear_dispatch, &[])
                    .unwrap();

                assert_eq!(
                    conv_in_bindings[1].buffer.read_bytes(16).unwrap(),
                    vec![
                        0xc7, 0x3e, 0x74, 0xbe, 0x7f, 0x3e, 0x97, 0x3e, 0x5a, 0xbe, 0xd2, 0xbe,
                        0xab, 0xbe, 0xc5, 0xbd,
                    ]
                );

                if let Some(split_spirv_words) =
                    crate::vulkan_compute::compile_test_shader_words_from_source(
                        "split_bf16_3072_to_3x1024.comp",
                    )
                {
                    let split_dispatch = mounted_bound.dispatch("layer_00", "split_b_c_x").unwrap();
                    assert!(split_dispatch.reusable_family_id.starts_with("split."));
                    let split_bindings = mounted
                        .resident_kernel_buffer_bindings_for_bound_dispatch(split_dispatch)
                        .unwrap();
                    assert_eq!(split_bindings[0].byte_len, 6_144);
                    assert_eq!(split_bindings[1].byte_len, 5_120);
                    assert_eq!(split_bindings[2].byte_len, 5_120);
                    assert_eq!(split_bindings[3].byte_len, 5_120);
                    let split_family = mounted
                        .placed_plan
                        .reusable_kernel_plan
                        .family(&split_dispatch.reusable_family_id)
                        .unwrap();
                    let split_kernel_manifest = VulkanLoadedReusableKernelArtifactManifest {
                        schema: VULKAN_REUSABLE_KERNEL_ARTIFACT_MANIFEST_SCHEMA.to_string(),
                        backend_id: VULKAN_STREAM_CIRCUIT_BACKEND_ID.to_string(),
                        total_word_count: split_spirv_words.len(),
                        artifacts: vec![VulkanLoadedReusableKernelArtifact {
                            artifact: VulkanReusableKernelArtifact::from_family(
                                split_family,
                                "kernels/split.spv",
                            ),
                            resolved_path: PathBuf::from("kernels/split.spv"),
                            words: split_spirv_words,
                        }],
                    };
                    let split_resident_dispatch = mounted
                        .create_resident_kernel_dispatch_for_bound_dispatch(
                            &device,
                            split_dispatch,
                            &split_kernel_manifest,
                        )
                        .unwrap();
                    assert_eq!(split_resident_dispatch.workgroup_count_x(), 1);

                    device
                        .run_resident_kernel_dispatch(&split_resident_dispatch, &[])
                        .unwrap();

                    assert_eq!(
                        split_bindings[1].buffer.read_bytes(16).unwrap(),
                        vec![
                            0xc7, 0x3e, 0x74, 0xbe, 0x7f, 0x3e, 0x97, 0x3e, 0x5a, 0xbe, 0xd2, 0xbe,
                            0xab, 0xbe, 0xc5, 0xbd,
                        ]
                    );
                    assert_eq!(
                        split_bindings[2].buffer.read_bytes(16).unwrap(),
                        vec![
                            0x04, 0xbf, 0x91, 0x3e, 0x9c, 0x3e, 0xd8, 0xbe, 0x9d, 0x3d, 0xe1, 0xbc,
                            0x87, 0x3d, 0x15, 0x3f,
                        ]
                    );
                    assert_eq!(
                        split_bindings[3].buffer.read_bytes(16).unwrap(),
                        vec![
                            0x16, 0xbe, 0xeb, 0xbe, 0x8c, 0xbc, 0xc3, 0x3d, 0x4d, 0xbf, 0x63, 0xbb,
                            0x40, 0xbe, 0x48, 0xbf,
                        ]
                    );

                    if let Some(multiply_spirv_words) =
                        crate::vulkan_compute::compile_test_shader_words_from_source(
                            "multiply_bf16_1024.comp",
                        )
                    {
                        let multiply_dispatch =
                            mounted_bound.dispatch("layer_00", "input_gate").unwrap();
                        assert!(
                            multiply_dispatch
                                .reusable_family_id
                                .starts_with("multiply.")
                        );
                        let multiply_bindings = mounted
                            .resident_kernel_buffer_bindings_for_bound_dispatch(multiply_dispatch)
                            .unwrap();
                        assert_eq!(multiply_bindings[0].byte_len, 5_120);
                        assert_eq!(multiply_bindings[1].byte_len, 5_120);
                        assert_eq!(multiply_bindings[2].byte_len, 6_144);
                        let multiply_family = mounted
                            .placed_plan
                            .reusable_kernel_plan
                            .family(&multiply_dispatch.reusable_family_id)
                            .unwrap();
                        let multiply_artifact_path = artifact_path_for_family(multiply_family);
                        let multiply_kernel_manifest = VulkanLoadedReusableKernelArtifactManifest {
                            schema: VULKAN_REUSABLE_KERNEL_ARTIFACT_MANIFEST_SCHEMA.to_string(),
                            backend_id: VULKAN_STREAM_CIRCUIT_BACKEND_ID.to_string(),
                            total_word_count: multiply_spirv_words.len(),
                            artifacts: vec![VulkanLoadedReusableKernelArtifact {
                                artifact: VulkanReusableKernelArtifact::from_family(
                                    multiply_family,
                                    multiply_artifact_path.clone(),
                                ),
                                resolved_path: PathBuf::from(multiply_artifact_path),
                                words: multiply_spirv_words,
                            }],
                        };
                        let multiply_resident_dispatch = mounted
                            .create_resident_kernel_dispatch_for_bound_dispatch(
                                &device,
                                multiply_dispatch,
                                &multiply_kernel_manifest,
                            )
                            .unwrap();
                        assert_eq!(multiply_resident_dispatch.workgroup_count_x(), 1);

                        device
                            .run_resident_kernel_dispatch(&multiply_resident_dispatch, &[])
                            .unwrap();

                        let gated_x_first_16 = vec![
                            0x69, 0xbd, 0xe0, 0x3d, 0x8b, 0xbb, 0xe6, 0x3c, 0x2f, 0x3e, 0xba, 0x3a,
                            0x80, 0x3d, 0x9a, 0x3d,
                        ];
                        assert_eq!(
                            multiply_bindings[2].buffer.read_bytes(16).unwrap(),
                            gated_x_first_16.clone()
                        );

                        if let Some(rolling_spirv_words) =
                            crate::vulkan_compute::compile_test_shader_words_from_source(
                                "rolling_state_update_bf16_3x1024.comp",
                            )
                        {
                            let rolling_dispatch = mounted_bound
                                .dispatch("layer_00", "temporal_memory_update")
                                .unwrap();
                            assert_eq!(rolling_dispatch.reusable_family_id, "rolling_state_update");
                            let rolling_bindings = mounted
                                .resident_kernel_buffer_bindings_for_bound_dispatch(
                                    rolling_dispatch,
                                )
                                .unwrap();
                            assert_eq!(
                                rolling_bindings
                                    .iter()
                                    .map(|binding| binding.byte_len)
                                    .collect::<Vec<_>>(),
                                vec![6_144, 6_144, 6_144, 6_144, 6_144, 6_144]
                            );
                            let zero_temporal_memory = vec![0u8; 6_144];
                            rolling_bindings[3]
                                .buffer
                                .write_bytes(&zero_temporal_memory)
                                .unwrap();
                            rolling_bindings[4]
                                .buffer
                                .write_bytes(&zero_temporal_memory)
                                .unwrap();
                            let rolling_family = mounted
                                .placed_plan
                                .reusable_kernel_plan
                                .family(&rolling_dispatch.reusable_family_id)
                                .unwrap();
                            let rolling_kernel_manifest =
                                VulkanLoadedReusableKernelArtifactManifest {
                                    schema: VULKAN_REUSABLE_KERNEL_ARTIFACT_MANIFEST_SCHEMA
                                        .to_string(),
                                    backend_id: VULKAN_STREAM_CIRCUIT_BACKEND_ID.to_string(),
                                    total_word_count: rolling_spirv_words.len(),
                                    artifacts: vec![VulkanLoadedReusableKernelArtifact {
                                        artifact: VulkanReusableKernelArtifact::from_family(
                                            rolling_family,
                                            "kernels/rolling_state_update.spv",
                                        ),
                                        resolved_path: PathBuf::from(
                                            "kernels/rolling_state_update.spv",
                                        ),
                                        words: rolling_spirv_words,
                                    }],
                                };
                            let rolling_resident_dispatch = mounted
                                .create_resident_kernel_dispatch_for_bound_dispatch(
                                    &device,
                                    rolling_dispatch,
                                    &rolling_kernel_manifest,
                                )
                                .unwrap();
                            assert_eq!(rolling_resident_dispatch.workgroup_count_x(), 1);

                            device
                                .run_resident_kernel_dispatch(&rolling_resident_dispatch, &[])
                                .unwrap();

                            let temporal_window =
                                rolling_bindings[2].buffer.read_bytes(6_144).unwrap();
                            assert!(
                                temporal_window[..4_096].iter().all(|byte| *byte == 0),
                                "first two temporal frames should be empty after a zero-state first tick"
                            );
                            assert_eq!(&temporal_window[4_096..4_112], gated_x_first_16.as_slice());
                            assert_eq!(
                                rolling_bindings[4].buffer.read_bytes(6_144).unwrap(),
                                temporal_window
                            );

                            mounted
                                .parameter_buffers
                                .load_parameter_from_tensor_index(
                                    &tensor_index,
                                    "model.layers.0.conv.conv.weight",
                                )
                                .unwrap();
                            if let Some(depthwise_spirv_words) =
                                crate::vulkan_compute::compile_test_shader_words_from_source(
                                    "depthwise_conv1d_bf16_3x1024.comp",
                                )
                            {
                                let depthwise_dispatch = mounted_bound
                                    .dispatch("layer_00", "depthwise_temporal_conv")
                                    .unwrap();
                                assert_eq!(
                                    depthwise_dispatch.reusable_family_id,
                                    "depthwise_conv1d"
                                );
                                let depthwise_bindings = mounted
                                    .resident_kernel_buffer_bindings_for_bound_dispatch(
                                        depthwise_dispatch,
                                    )
                                    .unwrap();
                                assert_eq!(depthwise_bindings.len(), 4);
                                assert_eq!(depthwise_bindings[0].byte_len, 6_144);
                                assert!(depthwise_bindings[1].byte_len >= 2_048);
                                assert_eq!(depthwise_bindings[2].byte_len, 6_144);
                                assert_eq!(depthwise_bindings[3].byte_len, 6_144);
                                let depthwise_family = mounted
                                    .placed_plan
                                    .reusable_kernel_plan
                                    .family(&depthwise_dispatch.reusable_family_id)
                                    .unwrap();
                                let depthwise_kernel_manifest =
                                    VulkanLoadedReusableKernelArtifactManifest {
                                        schema: VULKAN_REUSABLE_KERNEL_ARTIFACT_MANIFEST_SCHEMA
                                            .to_string(),
                                        backend_id: VULKAN_STREAM_CIRCUIT_BACKEND_ID.to_string(),
                                        total_word_count: depthwise_spirv_words.len(),
                                        artifacts: vec![VulkanLoadedReusableKernelArtifact {
                                            artifact: VulkanReusableKernelArtifact::from_family(
                                                depthwise_family,
                                                "kernels/depthwise_conv1d.spv",
                                            ),
                                            resolved_path: PathBuf::from(
                                                "kernels/depthwise_conv1d.spv",
                                            ),
                                            words: depthwise_spirv_words,
                                        }],
                                    };
                                let depthwise_resident_dispatch = mounted
                                    .create_resident_kernel_dispatch_for_bound_dispatch(
                                        &device,
                                        depthwise_dispatch,
                                        &depthwise_kernel_manifest,
                                    )
                                    .unwrap();
                                assert_eq!(depthwise_resident_dispatch.workgroup_count_x(), 1);

                                device
                                    .run_resident_kernel_dispatch(&depthwise_resident_dispatch, &[])
                                    .unwrap();

                                assert_eq!(
                                    depthwise_bindings[1].buffer.read_bytes(16).unwrap(),
                                    vec![
                                        0x20, 0x3c, 0xb1, 0xba, 0x17, 0x38, 0x6b, 0x38, 0x5b, 0xb9,
                                        0x82, 0x37, 0x6c, 0xb8, 0x8a, 0xba,
                                    ]
                                );

                                if let Some(output_gate_spirv_words) =
                                    crate::vulkan_compute::compile_test_shader_words_from_source(
                                        "multiply_bf16_1024.comp",
                                    )
                                {
                                    let output_gate_dispatch =
                                        mounted_bound.dispatch("layer_00", "output_gate").unwrap();
                                    assert_eq!(output_gate_dispatch.op, "multiply");
                                    let output_gate_bindings = mounted
                                        .resident_kernel_buffer_bindings_for_bound_dispatch(
                                            output_gate_dispatch,
                                        )
                                        .unwrap();
                                    assert_eq!(output_gate_bindings.len(), 3);
                                    assert!(output_gate_bindings[0].byte_len >= 2_048);
                                    assert!(output_gate_bindings[1].byte_len >= 2_048);
                                    assert!(output_gate_bindings[2].byte_len >= 2_048);
                                    let output_gate_family = mounted
                                        .placed_plan
                                        .reusable_kernel_plan
                                        .family(&output_gate_dispatch.reusable_family_id)
                                        .unwrap();
                                    let output_gate_artifact_path = format!(
                                        "kernels/{}.spv",
                                        output_gate_dispatch.reusable_family_id
                                    );
                                    let output_gate_kernel_manifest =
                                        VulkanLoadedReusableKernelArtifactManifest {
                                            schema: VULKAN_REUSABLE_KERNEL_ARTIFACT_MANIFEST_SCHEMA
                                                .to_string(),
                                            backend_id: VULKAN_STREAM_CIRCUIT_BACKEND_ID
                                                .to_string(),
                                            total_word_count: output_gate_spirv_words.len(),
                                            artifacts: vec![VulkanLoadedReusableKernelArtifact {
                                                artifact: VulkanReusableKernelArtifact::from_family(
                                                    output_gate_family,
                                                    output_gate_artifact_path.clone(),
                                                ),
                                                resolved_path: PathBuf::from(
                                                    output_gate_artifact_path,
                                                ),
                                                words: output_gate_spirv_words,
                                            }],
                                        };
                                    let output_gate_resident_dispatch = mounted
                                        .create_resident_kernel_dispatch_for_bound_dispatch(
                                            &device,
                                            output_gate_dispatch,
                                            &output_gate_kernel_manifest,
                                        )
                                        .unwrap();
                                    assert_eq!(
                                        output_gate_resident_dispatch.workgroup_count_x(),
                                        1
                                    );

                                    device
                                        .run_resident_kernel_dispatch(
                                            &output_gate_resident_dispatch,
                                            &[],
                                        )
                                        .unwrap();

                                    assert_eq!(
                                        output_gate_bindings[2].buffer.read_bytes(16).unwrap(),
                                        vec![
                                            0xa5, 0xbb, 0xc9, 0xb9, 0x38, 0x37, 0xc6, 0xb7, 0x86,
                                            0xb7, 0xe5, 0xb4, 0x79, 0xb6, 0x21, 0xba,
                                        ]
                                    );

                                    mounted
                                        .parameter_buffers
                                        .load_parameter_from_tensor_index(
                                            &tensor_index,
                                            "model.layers.0.conv.out_proj.weight",
                                        )
                                        .unwrap();
                                    if let Some(conv_out_projection_spirv_words) =
                                        crate::vulkan_compute::compile_test_shader_words_from_source(
                                            "linear_bf16_1024x1024.comp",
                                        )
                                    {
                                        let conv_out_projection_dispatch = mounted_bound
                                            .dispatch("layer_00", "conv_out_projection")
                                            .unwrap();
                                        assert_eq!(conv_out_projection_dispatch.op, "linear");
                                        let conv_out_projection_bindings = mounted
                                            .resident_kernel_buffer_bindings_for_bound_dispatch(
                                                conv_out_projection_dispatch,
                                            )
                                            .unwrap();
                                        assert_eq!(conv_out_projection_bindings.len(), 3);
                                        assert!(conv_out_projection_bindings[0].byte_len >= 2_048);
                                        assert!(conv_out_projection_bindings[1].byte_len >= 2_048);
                                        assert_eq!(
                                            conv_out_projection_bindings[2].byte_len,
                                            2_097_152
                                        );
                                        let conv_out_projection_family = mounted
                                            .placed_plan
                                            .reusable_kernel_plan
                                            .family(
                                                &conv_out_projection_dispatch.reusable_family_id,
                                            )
                                            .unwrap();
                                        let conv_out_projection_artifact_path = format!(
                                            "kernels/{}.spv",
                                            conv_out_projection_dispatch.reusable_family_id
                                        );
                                        let conv_out_projection_kernel_manifest =
                                                VulkanLoadedReusableKernelArtifactManifest {
                                                    schema:
                                                        VULKAN_REUSABLE_KERNEL_ARTIFACT_MANIFEST_SCHEMA
                                                            .to_string(),
                                                    backend_id: VULKAN_STREAM_CIRCUIT_BACKEND_ID
                                                        .to_string(),
                                                    total_word_count:
                                                        conv_out_projection_spirv_words.len(),
                                                    artifacts: vec![
                                                        VulkanLoadedReusableKernelArtifact {
                                                            artifact:
                                                                VulkanReusableKernelArtifact::from_family(
                                                                    conv_out_projection_family,
                                                                    conv_out_projection_artifact_path.clone(),
                                                                )
                                                                .with_workgroup_count_x(512),
                                                            resolved_path: PathBuf::from(
                                                                conv_out_projection_artifact_path,
                                                            ),
                                                            words: conv_out_projection_spirv_words,
                                                        },
                                                    ],
                                                };
                                        let conv_out_projection_resident_dispatch = mounted
                                            .create_resident_kernel_dispatch_for_bound_dispatch(
                                                &device,
                                                conv_out_projection_dispatch,
                                                &conv_out_projection_kernel_manifest,
                                            )
                                            .unwrap();
                                        assert!(
                                            conv_out_projection_resident_dispatch
                                                .workgroup_count_x()
                                                >= 8
                                        );

                                        device
                                            .run_resident_kernel_dispatch(
                                                &conv_out_projection_resident_dispatch,
                                                &[],
                                            )
                                            .unwrap();

                                        assert_eq!(
                                            conv_out_projection_bindings[1]
                                                .buffer
                                                .read_bytes(16)
                                                .unwrap(),
                                            vec![
                                                0x2f, 0xb9, 0xe4, 0xb9, 0xa3, 0xb9, 0x0c, 0xb9,
                                                0x4d, 0xba, 0x82, 0xb9, 0xfd, 0x39, 0x26, 0x3a,
                                            ]
                                        );

                                        if let Some(residual_spirv_words) =
                                                crate::vulkan_compute::compile_test_shader_words_from_source(
                                                    "add_bf16_1024.comp",
                                                )
                                            {
                                                let residual_dispatch = mounted_bound
                                                    .dispatch("layer_00", "operator_residual")
                                                    .unwrap();
                                                assert_eq!(
                                                    residual_dispatch.op,
                                                    "residual_add"
                                                );
                                                let residual_bindings = mounted
                                                    .resident_kernel_buffer_bindings_for_bound_dispatch(
                                                        residual_dispatch,
                                                    )
                                                    .unwrap();
                                                assert_eq!(residual_bindings.len(), 3);
                                                assert!(residual_bindings[0].byte_len >= 2_048);
                                                assert!(residual_bindings[1].byte_len >= 2_048);
                                                assert!(residual_bindings[2].byte_len >= 2_048);
                                                let residual_family = mounted
                                                    .placed_plan
                                                    .reusable_kernel_plan
                                                    .family(
                                                        &residual_dispatch.reusable_family_id,
                                                    )
                                                    .unwrap();
                                                let residual_artifact_path = format!(
                                                    "kernels/{}.spv",
                                                    residual_dispatch.reusable_family_id
                                                );
                                                let residual_kernel_manifest =
                                                    VulkanLoadedReusableKernelArtifactManifest {
                                                        schema:
                                                            VULKAN_REUSABLE_KERNEL_ARTIFACT_MANIFEST_SCHEMA
                                                                .to_string(),
                                                        backend_id: VULKAN_STREAM_CIRCUIT_BACKEND_ID
                                                            .to_string(),
                                                        total_word_count: residual_spirv_words.len(),
                                                        artifacts: vec![
                                                            VulkanLoadedReusableKernelArtifact {
                                                                artifact:
                                                                    VulkanReusableKernelArtifact::from_family(
                                                                        residual_family,
                                                                        residual_artifact_path.clone(),
                                                                    ),
                                                                resolved_path: PathBuf::from(
                                                                    residual_artifact_path,
                                                                ),
                                                                words: residual_spirv_words,
                                                            },
                                                        ],
                                                    };
                                                let residual_resident_dispatch = mounted
                                                    .create_resident_kernel_dispatch_for_bound_dispatch(
                                                        &device,
                                                        residual_dispatch,
                                                        &residual_kernel_manifest,
                                                    )
                                                    .unwrap();
                                                assert_eq!(
                                                    residual_resident_dispatch.workgroup_count_x(),
                                                    1
                                                );

                                                device
                                                    .run_resident_kernel_dispatch(
                                                        &residual_resident_dispatch,
                                                        &[],
                                                    )
                                                    .unwrap();

                                                let residual_output = residual_bindings[2]
                                                    .buffer
                                                    .read_bytes(2_048)
                                                    .unwrap();
                                                assert_eq!(
                                                    &residual_output[..16],
                                                    &[
                                                        0x80, 0x3f, 0x80, 0x3f, 0x80, 0x3f,
                                                        0x80, 0x3f, 0x80, 0x3f, 0x80, 0x3f,
                                                        0x80, 0x3f, 0x80, 0x3f,
                                                    ]
                                                );
                                                assert_eq!(
                                                    &residual_output[588..604],
                                                    &[
                                                        0x7e, 0x3f, 0x80, 0x3f, 0x80, 0x3f,
                                                        0x80, 0x3f, 0x80, 0x3f, 0x80, 0x3f,
                                                        0x80, 0x3f, 0x80, 0x3f,
                                                    ]
                                                );

                                                mounted
                                                    .parameter_buffers
                                                    .load_parameter_from_tensor_index(
                                                        &tensor_index,
                                                        "model.layers.0.ffn_norm.weight",
                                                    )
                                                    .unwrap();
                                                if let Some(ffn_norm_spirv_words) =
                                                    crate::vulkan_compute::compile_test_shader_words_from_source(
                                                        "rms_norm_bf16_serial.comp",
                                                    )
                                                {
                                                    let ffn_norm_dispatch = mounted_bound
                                                        .dispatch("layer_00", "ffn_norm")
                                                        .unwrap();
                                                    assert_eq!(ffn_norm_dispatch.op, "rms_norm");
                                                    let ffn_norm_bindings = mounted
                                                        .resident_kernel_buffer_bindings_for_bound_dispatch(
                                                            ffn_norm_dispatch,
                                                        )
                                                        .unwrap();
                                                    assert_eq!(ffn_norm_bindings.len(), 3);
                                                    assert!(ffn_norm_bindings[0].byte_len >= 2_048);
                                                    assert!(ffn_norm_bindings[1].byte_len >= 2_048);
                                                    assert_eq!(ffn_norm_bindings[2].byte_len, 2_048);
                                                    let ffn_norm_family = mounted
                                                        .placed_plan
                                                        .reusable_kernel_plan
                                                        .family(
                                                            &ffn_norm_dispatch.reusable_family_id,
                                                        )
                                                        .unwrap();
                                                    let ffn_norm_artifact_path = format!(
                                                        "kernels/{}.spv",
                                                        ffn_norm_dispatch.reusable_family_id
                                                    );
                                                    let ffn_norm_kernel_manifest =
                                                        VulkanLoadedReusableKernelArtifactManifest {
                                                            schema:
                                                                VULKAN_REUSABLE_KERNEL_ARTIFACT_MANIFEST_SCHEMA
                                                                    .to_string(),
                                                            backend_id:
                                                                VULKAN_STREAM_CIRCUIT_BACKEND_ID
                                                                    .to_string(),
                                                            total_word_count:
                                                                ffn_norm_spirv_words.len(),
                                                            artifacts: vec![
                                                                VulkanLoadedReusableKernelArtifact {
                                                                    artifact:
                                                                        VulkanReusableKernelArtifact::from_family(
                                                                            ffn_norm_family,
                                                                            ffn_norm_artifact_path.clone(),
                                                                        ),
                                                                    resolved_path: PathBuf::from(
                                                                        ffn_norm_artifact_path,
                                                                    ),
                                                                    words: ffn_norm_spirv_words,
                                                                },
                                                            ],
                                                        };
                                                    let ffn_norm_resident_dispatch = mounted
                                                        .create_resident_kernel_dispatch_for_bound_dispatch(
                                                            &device,
                                                            ffn_norm_dispatch,
                                                            &ffn_norm_kernel_manifest,
                                                        )
                                                        .unwrap();
                                                    assert_eq!(
                                                        ffn_norm_resident_dispatch
                                                            .workgroup_count_x(),
                                                        1
                                                    );

                                                    device
                                                        .run_resident_kernel_dispatch(
                                                            &ffn_norm_resident_dispatch,
                                                            &[],
                                                        )
                                                        .unwrap();

                                                    assert_eq!(
                                                        ffn_norm_bindings[1]
                                                            .buffer
                                                            .read_bytes(16)
                                                            .unwrap(),
                                                        vec![
                                                            0x6b, 0x3e, 0x6e, 0x3e, 0x69, 0x3e,
                                                            0x6e, 0x3e, 0x78, 0x3e, 0x6e, 0x3e,
                                                            0x79, 0x3e, 0x99, 0x3e,
                                                        ]
                                                    );

                                                    if let Some(ffn_projection_spirv_words) =
                                                        crate::vulkan_compute::compile_test_shader_words_from_source(
                                                            "linear_bf16_1024x2560.comp",
                                                        )
                                                    {
                                                        mounted
                                                            .parameter_buffers
                                                            .load_parameter_from_tensor_index(
                                                                &tensor_index,
                                                                "model.layers.0.feed_forward.w1.weight",
                                                            )
                                                            .unwrap();
                                                        let ffn_gate_dispatch = mounted_bound
                                                            .dispatch(
                                                                "layer_00",
                                                                "ffn_gate_projection",
                                                            )
                                                            .unwrap();
                                                        assert_eq!(ffn_gate_dispatch.op, "linear");
                                                        let ffn_gate_bindings = mounted
                                                            .resident_kernel_buffer_bindings_for_bound_dispatch(
                                                                ffn_gate_dispatch,
                                                            )
                                                            .unwrap();
                                                        assert_eq!(ffn_gate_bindings.len(), 3);
                                                        assert!(
                                                            ffn_gate_bindings[0].byte_len >= 2_048
                                                        );
                                                        assert_eq!(
                                                            ffn_gate_bindings[1].byte_len,
                                                            5_120
                                                        );
                                                        assert_eq!(
                                                            ffn_gate_bindings[2].byte_len,
                                                            5_242_880
                                                        );
                                                        let ffn_gate_family = mounted
                                                            .placed_plan
                                                            .reusable_kernel_plan
                                                            .family(
                                                                &ffn_gate_dispatch
                                                                    .reusable_family_id,
                                                            )
                                                            .unwrap();
                                                        let ffn_gate_artifact_path = format!(
                                                            "kernels/{}.spv",
                                                            ffn_gate_dispatch.reusable_family_id
                                                        );
                                                        let ffn_gate_kernel_manifest =
                                                            VulkanLoadedReusableKernelArtifactManifest {
                                                                schema:
                                                                    VULKAN_REUSABLE_KERNEL_ARTIFACT_MANIFEST_SCHEMA
                                                                        .to_string(),
                                                                backend_id:
                                                                    VULKAN_STREAM_CIRCUIT_BACKEND_ID
                                                                        .to_string(),
                                                                total_word_count:
                                                                    ffn_projection_spirv_words
                                                                        .len(),
                                                                artifacts: vec![
                                                                    VulkanLoadedReusableKernelArtifact {
                                                                        artifact:
                                                                            VulkanReusableKernelArtifact::from_family(
                                                                                ffn_gate_family,
                                                                                ffn_gate_artifact_path.clone(),
                                                                            )
                                                                            .with_workgroup_count_x(1_280),
                                                                        resolved_path:
                                                                            PathBuf::from(
                                                                                ffn_gate_artifact_path,
                                                                            ),
                                                                        words:
                                                                            ffn_projection_spirv_words
                                                                                .clone(),
                                                                    },
                                                                ],
                                                            };
                                                        let ffn_gate_resident_dispatch = mounted
                                                            .create_resident_kernel_dispatch_for_bound_dispatch(
                                                                &device,
                                                                ffn_gate_dispatch,
                                                                &ffn_gate_kernel_manifest,
                                                            )
                                                            .unwrap();
                                                        assert!(
                                                            ffn_gate_resident_dispatch
                                                                .workgroup_count_x()
                                                                >= 20
                                                        );

                                                        device
                                                            .run_resident_kernel_dispatch(
                                                                &ffn_gate_resident_dispatch,
                                                                &[],
                                                            )
                                                            .unwrap();

                                                        assert_eq!(
                                                            ffn_gate_bindings[1]
                                                                .buffer
                                                                .read_bytes(16)
                                                                .unwrap(),
                                                            vec![
                                                                0x0a, 0x3d, 0x16, 0x3e, 0xea,
                                                                0x3d, 0x7c, 0x3e, 0x88, 0x3e,
                                                                0x07, 0x3e, 0x4a, 0x3e, 0x38,
                                                                0x3d,
                                                            ]
                                                        );

                                                        mounted
                                                            .parameter_buffers
                                                            .load_parameter_from_tensor_index(
                                                                &tensor_index,
                                                                "model.layers.0.feed_forward.w3.weight",
                                                            )
                                                            .unwrap();
                                                        let ffn_up_dispatch = mounted_bound
                                                            .dispatch(
                                                                "layer_00",
                                                                "ffn_up_projection",
                                                            )
                                                            .unwrap();
                                                        assert_eq!(ffn_up_dispatch.op, "linear");
                                                        let ffn_up_bindings = mounted
                                                            .resident_kernel_buffer_bindings_for_bound_dispatch(
                                                                ffn_up_dispatch,
                                                            )
                                                            .unwrap();
                                                        assert_eq!(ffn_up_bindings.len(), 3);
                                                        assert!(
                                                            ffn_up_bindings[0].byte_len >= 2_048
                                                        );
                                                        assert_eq!(
                                                            ffn_up_bindings[1].byte_len,
                                                            5_120
                                                        );
                                                        assert_eq!(
                                                            ffn_up_bindings[2].byte_len,
                                                            5_242_880
                                                        );
                                                        let ffn_up_family = mounted
                                                            .placed_plan
                                                            .reusable_kernel_plan
                                                            .family(
                                                                &ffn_up_dispatch
                                                                    .reusable_family_id,
                                                            )
                                                            .unwrap();
                                                        let ffn_up_artifact_path = format!(
                                                            "kernels/{}.spv",
                                                            ffn_up_dispatch.reusable_family_id
                                                        );
                                                        let ffn_up_kernel_manifest =
                                                            VulkanLoadedReusableKernelArtifactManifest {
                                                                schema:
                                                                    VULKAN_REUSABLE_KERNEL_ARTIFACT_MANIFEST_SCHEMA
                                                                        .to_string(),
                                                                backend_id:
                                                                    VULKAN_STREAM_CIRCUIT_BACKEND_ID
                                                                        .to_string(),
                                                                total_word_count:
                                                                    ffn_projection_spirv_words
                                                                        .len(),
                                                                artifacts: vec![
                                                                    VulkanLoadedReusableKernelArtifact {
                                                                        artifact:
                                                                            VulkanReusableKernelArtifact::from_family(
                                                                                ffn_up_family,
                                                                                ffn_up_artifact_path.clone(),
                                                                            )
                                                                            .with_workgroup_count_x(1_280),
                                                                        resolved_path:
                                                                            PathBuf::from(
                                                                                ffn_up_artifact_path,
                                                                            ),
                                                                        words:
                                                                            ffn_projection_spirv_words,
                                                                    },
                                                                ],
                                                            };
                                                        let ffn_up_resident_dispatch = mounted
                                                            .create_resident_kernel_dispatch_for_bound_dispatch(
                                                                &device,
                                                                ffn_up_dispatch,
                                                                &ffn_up_kernel_manifest,
                                                            )
                                                            .unwrap();
                                                        assert!(
                                                            ffn_up_resident_dispatch
                                                                .workgroup_count_x()
                                                                >= 20
                                                        );

                                                        device
                                                            .run_resident_kernel_dispatch(
                                                                &ffn_up_resident_dispatch,
                                                                &[],
                                                            )
                                                            .unwrap();

                                                        assert_eq!(
                                                            ffn_up_bindings[1]
                                                                .buffer
                                                                .read_bytes(16)
                                                                .unwrap(),
                                                            vec![
                                                                0x35, 0xbe, 0xe6, 0xbe, 0x5d,
                                                                0xbe, 0x1d, 0x3e, 0x2a, 0xbe,
                                                                0x8b, 0x3c, 0x5e, 0x3e, 0xb1,
                                                                0xbe,
                                                            ]
                                                        );

                                                        if let Some(silu_spirv_words) =
                                                            crate::vulkan_compute::compile_test_shader_words_from_source(
                                                                "silu_bf16_2560.comp",
                                                            )
                                                        {
                                                            let silu_dispatch = mounted_bound
                                                                .dispatch(
                                                                    "layer_00",
                                                                    "ffn_gate_activation",
                                                                )
                                                                .unwrap();
                                                            assert_eq!(silu_dispatch.op, "silu");
                                                            let silu_bindings = mounted
                                                                .resident_kernel_buffer_bindings_for_bound_dispatch(
                                                                    silu_dispatch,
                                                                )
                                                                .unwrap();
                                                            assert_eq!(silu_bindings.len(), 2);
                                                            assert_eq!(
                                                                silu_bindings[0].byte_len,
                                                                5_120
                                                            );
                                                            assert_eq!(
                                                                silu_bindings[1].byte_len,
                                                                5_120
                                                            );
                                                            let silu_family = mounted
                                                                .placed_plan
                                                                .reusable_kernel_plan
                                                                .family(
                                                                    &silu_dispatch
                                                                        .reusable_family_id,
                                                                )
                                                                .unwrap();
                                                            let silu_artifact_path = format!(
                                                                "kernels/{}.spv",
                                                                silu_dispatch.reusable_family_id
                                                            );
                                                            let silu_kernel_manifest =
                                                                VulkanLoadedReusableKernelArtifactManifest {
                                                                    schema:
                                                                        VULKAN_REUSABLE_KERNEL_ARTIFACT_MANIFEST_SCHEMA
                                                                            .to_string(),
                                                                    backend_id:
                                                                        VULKAN_STREAM_CIRCUIT_BACKEND_ID
                                                                            .to_string(),
                                                                    total_word_count:
                                                                        silu_spirv_words.len(),
                                                                    artifacts: vec![
                                                                        VulkanLoadedReusableKernelArtifact {
                                                                            artifact:
                                                                                VulkanReusableKernelArtifact::from_family(
                                                                                    silu_family,
                                                                                    silu_artifact_path.clone(),
                                                                                ),
                                                                            resolved_path:
                                                                                PathBuf::from(
                                                                                    silu_artifact_path,
                                                                                ),
                                                                            words:
                                                                                silu_spirv_words,
                                                                        },
                                                                    ],
                                                                };
                                                            let silu_resident_dispatch = mounted
                                                                .create_resident_kernel_dispatch_for_bound_dispatch(
                                                                    &device,
                                                                    silu_dispatch,
                                                                    &silu_kernel_manifest,
                                                                )
                                                                .unwrap();
                                                            assert_eq!(
                                                                silu_resident_dispatch
                                                                    .workgroup_count_x(),
                                                                1
                                                            );

                                                            device
                                                                .run_resident_kernel_dispatch(
                                                                    &silu_resident_dispatch,
                                                                    &[],
                                                                )
                                                                .unwrap();

                                                            assert_eq!(
                                                                silu_bindings[1]
                                                                    .buffer
                                                                    .read_bytes(16)
                                                                    .unwrap(),
                                                                vec![
                                                                    0x8c, 0x3c, 0xa1, 0x3d, 0x77,
                                                                    0x3d, 0x0d, 0x3e, 0x1a, 0x3e,
                                                                    0x90, 0x3d, 0xde, 0x3d, 0xbc,
                                                                    0x3c,
                                                                ]
                                                            );

                                                            if let Some(ffn_multiply_spirv_words) =
                                                                crate::vulkan_compute::compile_test_shader_words_from_source(
                                                                    "multiply_bf16_2560.comp",
                                                                )
                                                            {
                                                                let ffn_multiply_dispatch =
                                                                    mounted_bound
                                                                        .dispatch(
                                                                            "layer_00",
                                                                            "ffn_gate_multiply",
                                                                        )
                                                                        .unwrap();
                                                                assert_eq!(
                                                                    ffn_multiply_dispatch.op,
                                                                    "multiply"
                                                                );
                                                                let ffn_multiply_bindings = mounted
                                                                    .resident_kernel_buffer_bindings_for_bound_dispatch(
                                                                        ffn_multiply_dispatch,
                                                                    )
                                                                    .unwrap();
                                                                assert_eq!(
                                                                    ffn_multiply_bindings.len(),
                                                                    3
                                                                );
                                                                assert_eq!(
                                                                    ffn_multiply_bindings[0]
                                                                        .byte_len,
                                                                    5_120
                                                                );
                                                                assert_eq!(
                                                                    ffn_multiply_bindings[1]
                                                                        .byte_len,
                                                                    5_120
                                                                );
                                                                assert_eq!(
                                                                    ffn_multiply_bindings[2]
                                                                        .byte_len,
                                                                    5_120
                                                                );
                                                                let ffn_multiply_family = mounted
                                                                    .placed_plan
                                                                    .reusable_kernel_plan
                                                                    .family(
                                                                        &ffn_multiply_dispatch
                                                                            .reusable_family_id,
                                                                    )
                                                                    .unwrap();
                                                                let ffn_multiply_artifact_path =
                                                                    format!(
                                                                        "kernels/{}.spv",
                                                                        ffn_multiply_dispatch
                                                                            .reusable_family_id
                                                                    );
                                                                let ffn_multiply_kernel_manifest =
                                                                    VulkanLoadedReusableKernelArtifactManifest {
                                                                        schema:
                                                                            VULKAN_REUSABLE_KERNEL_ARTIFACT_MANIFEST_SCHEMA
                                                                                .to_string(),
                                                                        backend_id:
                                                                            VULKAN_STREAM_CIRCUIT_BACKEND_ID
                                                                                .to_string(),
                                                                        total_word_count:
                                                                            ffn_multiply_spirv_words
                                                                                .len(),
                                                                        artifacts: vec![
                                                                            VulkanLoadedReusableKernelArtifact {
                                                                                artifact:
                                                                                    VulkanReusableKernelArtifact::from_family(
                                                                                        ffn_multiply_family,
                                                                                        ffn_multiply_artifact_path.clone(),
                                                                                    ),
                                                                                resolved_path:
                                                                                    PathBuf::from(
                                                                                        ffn_multiply_artifact_path,
                                                                                    ),
                                                                                words:
                                                                                    ffn_multiply_spirv_words,
                                                                            },
                                                                        ],
                                                                    };
                                                                let ffn_multiply_resident_dispatch =
                                                                    mounted
                                                                        .create_resident_kernel_dispatch_for_bound_dispatch(
                                                                            &device,
                                                                            ffn_multiply_dispatch,
                                                                            &ffn_multiply_kernel_manifest,
                                                                        )
                                                                        .unwrap();
                                                                assert_eq!(
                                                                    ffn_multiply_resident_dispatch
                                                                        .workgroup_count_x(),
                                                                    1
                                                                );

                                                                device
                                                                    .run_resident_kernel_dispatch(
                                                                        &ffn_multiply_resident_dispatch,
                                                                        &[],
                                                                    )
                                                                    .unwrap();

                                                                assert_eq!(
                                                                    ffn_multiply_bindings[2]
                                                                        .buffer
                                                                        .read_bytes(16)
                                                                        .unwrap(),
                                                                    vec![
                                                                        0x46, 0xbb, 0x11, 0xbd,
                                                                        0x55, 0xbc, 0xad, 0x3c,
                                                                        0xcd, 0xbc, 0x9c, 0x3a,
                                                                        0xc1, 0x3c, 0x02, 0xbc,
                                                                    ]
                                                                );

                                                                mounted
                                                                    .parameter_buffers
                                                                    .load_parameter_from_tensor_index(
                                                                        &tensor_index,
                                                                        "model.layers.0.feed_forward.w2.weight",
                                                                    )
                                                                    .unwrap();
                                                                if let Some(ffn_down_spirv_words) =
                                                                    crate::vulkan_compute::compile_test_shader_words_from_source(
                                                                        "linear_bf16_2560x1024.comp",
                                                                    )
                                                                {
                                                                    let ffn_down_dispatch =
                                                                        mounted_bound
                                                                            .dispatch(
                                                                                "layer_00",
                                                                                "ffn_down_projection",
                                                                            )
                                                                            .unwrap();
                                                                    assert_eq!(
                                                                        ffn_down_dispatch.op,
                                                                        "linear"
                                                                    );
                                                                    let ffn_down_bindings = mounted
                                                                        .resident_kernel_buffer_bindings_for_bound_dispatch(
                                                                            ffn_down_dispatch,
                                                                        )
                                                                        .unwrap();
                                                                    assert_eq!(
                                                                        ffn_down_bindings.len(),
                                                                        3
                                                                    );
                                                                    assert_eq!(
                                                                        ffn_down_bindings[0]
                                                                            .byte_len,
                                                                        5_120
                                                                    );
                                                                    assert!(
                                                                        ffn_down_bindings[1]
                                                                            .byte_len
                                                                            >= 2_048
                                                                    );
                                                                    assert_eq!(
                                                                        ffn_down_bindings[2]
                                                                            .byte_len,
                                                                        5_242_880
                                                                    );
                                                                    let ffn_down_family = mounted
                                                                        .placed_plan
                                                                        .reusable_kernel_plan
                                                                        .family(
                                                                            &ffn_down_dispatch
                                                                                .reusable_family_id,
                                                                        )
                                                                        .unwrap();
                                                                    let ffn_down_artifact_path =
                                                                        format!(
                                                                            "kernels/{}.spv",
                                                                            ffn_down_dispatch
                                                                                .reusable_family_id
                                                                        );
                                                                    let ffn_down_kernel_manifest =
                                                                        VulkanLoadedReusableKernelArtifactManifest {
                                                                            schema:
                                                                                VULKAN_REUSABLE_KERNEL_ARTIFACT_MANIFEST_SCHEMA
                                                                                    .to_string(),
                                                                            backend_id:
                                                                                VULKAN_STREAM_CIRCUIT_BACKEND_ID
                                                                                    .to_string(),
                                                                            total_word_count:
                                                                                ffn_down_spirv_words
                                                                                    .len(),
                                                                            artifacts: vec![
                                                                                VulkanLoadedReusableKernelArtifact {
                                                                                    artifact:
                                                                                        VulkanReusableKernelArtifact::from_family(
                                                                                            ffn_down_family,
                                                                                            ffn_down_artifact_path.clone(),
                                                                                        )
                                                                                        .with_workgroup_count_x(512),
                                                                                    resolved_path:
                                                                                        PathBuf::from(
                                                                                            ffn_down_artifact_path,
                                                                                        ),
                                                                                    words:
                                                                                        ffn_down_spirv_words,
                                                                                },
                                                                            ],
                                                                        };
                                                                    let ffn_down_resident_dispatch =
                                                                        mounted
                                                                            .create_resident_kernel_dispatch_for_bound_dispatch(
                                                                                &device,
                                                                                ffn_down_dispatch,
                                                                                &ffn_down_kernel_manifest,
                                                                            )
                                                                            .unwrap();
                                                                    assert!(
                                                                        ffn_down_resident_dispatch
                                                                            .workgroup_count_x()
                                                                            >= 8
                                                                    );

                                                                    device
                                                                        .run_resident_kernel_dispatch(
                                                                            &ffn_down_resident_dispatch,
                                                                            &[],
                                                                        )
                                                                        .unwrap();

                                                                    assert_eq!(
                                                                        ffn_down_bindings[1]
                                                                            .buffer
                                                                            .read_bytes(16)
                                                                            .unwrap(),
                                                                        vec![
                                                                            0x37, 0x3d, 0x80,
                                                                            0x3c, 0x06, 0x3c,
                                                                            0x1d, 0xbc, 0xc2,
                                                                            0x3c, 0xac, 0x3c,
                                                                            0xc2, 0x3c, 0xa2,
                                                                            0x3c,
                                                                        ]
                                                                    );

                                                                    if let Some(final_residual_spirv_words) =
                                                                        crate::vulkan_compute::compile_test_shader_words_from_source(
                                                                            "add_bf16_1024.comp",
                                                                        )
                                                                    {
                                                                        let final_residual_dispatch =
                                                                            mounted_bound
                                                                                .dispatch(
                                                                                    "layer_00",
                                                                                    "ffn_residual",
                                                                                )
                                                                                .unwrap();
                                                                        assert_eq!(
                                                                            final_residual_dispatch.op,
                                                                            "residual_add"
                                                                        );
                                                                        let final_residual_bindings = mounted
                                                                            .resident_kernel_buffer_bindings_for_bound_dispatch(
                                                                                final_residual_dispatch,
                                                                            )
                                                                            .unwrap();
                                                                        assert_eq!(
                                                                            final_residual_bindings
                                                                                .len(),
                                                                            3
                                                                        );
                                                                        assert!(
                                                                            final_residual_bindings[0]
                                                                                .byte_len
                                                                                >= 2_048
                                                                        );
                                                                        assert!(
                                                                            final_residual_bindings[1]
                                                                                .byte_len
                                                                                >= 2_048
                                                                        );
                                                                        assert!(
                                                                            final_residual_bindings[2]
                                                                                .byte_len
                                                                                >= 2_048
                                                                        );
                                                                        let final_residual_family =
                                                                            mounted
                                                                                .placed_plan
                                                                                .reusable_kernel_plan
                                                                                .family(
                                                                                    &final_residual_dispatch
                                                                                        .reusable_family_id,
                                                                                )
                                                                                .unwrap();
                                                                        let final_residual_artifact_path =
                                                                            format!(
                                                                                "kernels/{}.spv",
                                                                                final_residual_dispatch
                                                                                    .reusable_family_id
                                                                            );
                                                                        let final_residual_kernel_manifest =
                                                                            VulkanLoadedReusableKernelArtifactManifest {
                                                                                schema:
                                                                                    VULKAN_REUSABLE_KERNEL_ARTIFACT_MANIFEST_SCHEMA
                                                                                        .to_string(),
                                                                                backend_id:
                                                                                    VULKAN_STREAM_CIRCUIT_BACKEND_ID
                                                                                        .to_string(),
                                                                                total_word_count:
                                                                                    final_residual_spirv_words
                                                                                        .len(),
                                                                                artifacts: vec![
                                                                                    VulkanLoadedReusableKernelArtifact {
                                                                                        artifact:
                                                                                            VulkanReusableKernelArtifact::from_family(
                                                                                                final_residual_family,
                                                                                                final_residual_artifact_path.clone(),
                                                                                            ),
                                                                                        resolved_path:
                                                                                            PathBuf::from(
                                                                                                final_residual_artifact_path,
                                                                                            ),
                                                                                        words:
                                                                                            final_residual_spirv_words,
                                                                                    },
                                                                                ],
                                                                            };
                                                                        let final_residual_resident_dispatch =
                                                                            mounted
                                                                                .create_resident_kernel_dispatch_for_bound_dispatch(
                                                                                    &device,
                                                                                    final_residual_dispatch,
                                                                                    &final_residual_kernel_manifest,
                                                                                )
                                                                                .unwrap();
                                                                        assert_eq!(
                                                                            final_residual_resident_dispatch
                                                                                .workgroup_count_x(),
                                                                            1
                                                                        );

                                                                        device
                                                                            .run_resident_kernel_dispatch(
                                                                                &final_residual_resident_dispatch,
                                                                                &[],
                                                                            )
                                                                            .unwrap();

                                                                        assert_eq!(
                                                                            final_residual_bindings[2]
                                                                                .buffer
                                                                                .read_bytes(16)
                                                                                .unwrap(),
                                                                            vec![
                                                                                0x86, 0x3f,
                                                                                0x82, 0x3f,
                                                                                0x81, 0x3f,
                                                                                0x7e, 0x3f,
                                                                                0x83, 0x3f,
                                                                                0x83, 0x3f,
                                                                                0x83, 0x3f,
                                                                                0x83, 0x3f,
                                                                            ]
                                                                        );
                                                                    } else {
                                                                        eprintln!(
                                                                            "skipping BF16 final residual Vulkan dispatch: no GLSL to SPIR-V compiler found"
                                                                        );
                                                                    }
                                                                } else {
                                                                    eprintln!(
                                                                        "skipping BF16 FFN down projection Vulkan dispatch: no GLSL to SPIR-V compiler found"
                                                                    );
                                                                }
                                                            } else {
                                                                eprintln!(
                                                                    "skipping BF16 FFN multiply Vulkan dispatch: no GLSL to SPIR-V compiler found"
                                                                );
                                                            }
                                                        } else {
                                                            eprintln!(
                                                                "skipping BF16 SiLU Vulkan dispatch: no GLSL to SPIR-V compiler found"
                                                            );
                                                        }
                                                    } else {
                                                        eprintln!(
                                                            "skipping BF16 FFN projection Vulkan dispatches: no GLSL to SPIR-V compiler found"
                                                        );
                                                    }
                                                } else {
                                                    eprintln!(
                                                        "skipping BF16 FFN RMSNorm Vulkan dispatch: no GLSL to SPIR-V compiler found"
                                                    );
                                                }
                                            } else {
                                                eprintln!(
                                                    "skipping BF16 operator residual Vulkan dispatch: no GLSL to SPIR-V compiler found"
                                                );
                                            }
                                    } else {
                                        eprintln!(
                                            "skipping BF16 conv out projection Vulkan dispatch: no GLSL to SPIR-V compiler found"
                                        );
                                    }
                                } else {
                                    eprintln!(
                                        "skipping BF16 output gate Vulkan dispatch: no GLSL to SPIR-V compiler found"
                                    );
                                }
                            } else {
                                eprintln!(
                                    "skipping BF16 depthwise conv Vulkan dispatch: no GLSL to SPIR-V compiler found"
                                );
                            }
                        } else {
                            eprintln!(
                                "skipping BF16 rolling state Vulkan dispatch: no GLSL to SPIR-V compiler found"
                            );
                        }
                    } else {
                        eprintln!(
                            "skipping BF16 multiply Vulkan dispatch: no GLSL to SPIR-V compiler found"
                        );
                    }
                } else {
                    eprintln!(
                        "skipping BF16 split Vulkan dispatch: no GLSL to SPIR-V compiler found"
                    );
                }
            } else {
                eprintln!("skipping linear BF16 Vulkan dispatch: no GLSL to SPIR-V compiler found");
            }
        } else {
            eprintln!("skipping serial RMSNorm Vulkan dispatch: no GLSL to SPIR-V compiler found");
        }
    }

    assert_eq!(
        mounted_bound
            .dispatch("layer_00", "operator_norm")
            .unwrap()
            .descriptors[0]
            .target,
        VulkanMountedPlacedBoundDescriptorTarget::ModelInput {
            signal_id: "input_frame".to_string(),
        }
    );
    assert_eq!(
        mounted_bound
            .dispatch("layer_00", "ffn_residual")
            .unwrap()
            .descriptors
            .last()
            .unwrap()
            .target,
        VulkanMountedPlacedBoundDescriptorTarget::LocalCableOutputBuffer {
            cable: VulkanPlacedLocalCableBufferBinding {
                buffer_index: 0,
                cable: mounted
                    .cable_io
                    .local_cable_buffer(0)
                    .unwrap()
                    .cable
                    .clone(),
                byte_capacity: 2_048,
            },
        }
    );
    assert_eq!(
        mounted_bound
            .dispatch("layer_01", "operator_norm")
            .unwrap()
            .descriptors[0]
            .target,
        VulkanMountedPlacedBoundDescriptorTarget::LocalCableInputBuffer {
            cable: VulkanPlacedLocalCableBufferBinding {
                buffer_index: 0,
                cable: mounted
                    .cable_io
                    .local_cable_buffer(0)
                    .unwrap()
                    .cable
                    .clone(),
                byte_capacity: 2_048,
            },
        }
    );
    assert_eq!(
        mounted_bound
            .dispatch("layer_01", "operator_residual")
            .unwrap()
            .descriptors[0]
            .target,
        VulkanMountedPlacedBoundDescriptorTarget::LocalCableInputBuffer {
            cable: VulkanPlacedLocalCableBufferBinding {
                buffer_index: 0,
                cable: mounted
                    .cable_io
                    .local_cable_buffer(0)
                    .unwrap()
                    .cable
                    .clone(),
                byte_capacity: 2_048,
            },
        }
    );
    assert_eq!(
        mounted_bound
            .dispatch("layer_13", "ffn_residual")
            .unwrap()
            .descriptors
            .last()
            .unwrap()
            .target,
        VulkanMountedPlacedBoundDescriptorTarget::ModelOutput {
            signal_id: "output_frame".to_string(),
        }
    );
    let layer_01_norm_tick = match &tick_plan.stages[16] {
        VulkanMountedPlacedStreamTickStage::Dispatch { dispatch, .. } => dispatch,
        stage => panic!("expected layer_01 operator_norm dispatch, got {stage:?}"),
    };
    assert_eq!(layer_01_norm_tick.pedal_id, "layer_01");
    assert_eq!(layer_01_norm_tick.node_id, "operator_norm");
    assert_eq!(
        layer_01_norm_tick.reads,
        vec![VulkanMountedPlacedStreamTickIo::LocalCableBuffer {
            cable_index: 0,
            buffer_index: 0,
            byte_capacity: 2_048,
        }]
    );
}

