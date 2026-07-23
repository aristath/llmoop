fn selected_test_vulkan_device() -> Result<VulkanComputeDevice, VulkanError> {
    match std::env::var("NERVE_TEST_VULKAN_DEVICE_INDEX") {
        Ok(raw_index) => {
            let index = raw_index.parse::<usize>().map_err(|error| {
                VulkanError(format!(
                    "invalid NERVE_TEST_VULKAN_DEVICE_INDEX {raw_index:?}: {error}"
                ))
            })?;
            VulkanComputeDevice::new_for_physical_device_index(index)
        }
        Err(std::env::VarError::NotPresent) => VulkanComputeDevice::new(),
        Err(error) => Err(VulkanError(format!(
            "could not read NERVE_TEST_VULKAN_DEVICE_INDEX: {error}"
        ))),
    }
}

#[test]
fn backend_loop_window_is_device_owned_and_snapshot_memory_bounded() {
    assert_eq!(backend_loop_window_for_static_state_bytes(0, 4_096), 64);
    assert_eq!(
        backend_loop_window_for_static_state_bytes(2 * 1024 * 1024, 4_096),
        32
    );
    assert_eq!(
        backend_loop_window_for_static_state_bytes(128 * 1024 * 1024, 4_096),
        1
    );
    assert_eq!(backend_loop_window_for_static_state_bytes(0, 8), 8);
}

#[test]
fn placed_feedback_window_accepts_bridged_multi_device_execution_graphs() {
    let eligible = VulkanResidentInProcessPlacedFeedbackLoopEligibility {
        device_slice_count: 3,
        every_slice_has_terminal_segment: true,
        distributed_dispatches_are_bridged: true,
        has_push_constants: false,
        static_state_bytes: 0,
        sampler_history_capacity: 4_096,
    };
    assert_eq!(eligible.window_width(), Some(64));
    assert_eq!(
        VulkanResidentInProcessPlacedFeedbackLoopEligibility {
            static_state_bytes: 2 * 1024 * 1024,
            ..eligible
        }
        .window_width(),
        Some(32)
    );
    assert_eq!(
        VulkanResidentInProcessPlacedFeedbackLoopEligibility {
            sampler_history_capacity: 8,
            ..eligible
        }
        .window_width(),
        Some(8)
    );
    assert_eq!(
        VulkanResidentInProcessPlacedFeedbackLoopEligibility {
            device_slice_count: 0,
            ..eligible
        }
        .window_width(),
        None
    );
    assert_eq!(
        VulkanResidentInProcessPlacedFeedbackLoopEligibility {
            every_slice_has_terminal_segment: false,
            ..eligible
        }
        .window_width(),
        None
    );
    assert_eq!(
        VulkanResidentInProcessPlacedFeedbackLoopEligibility {
            distributed_dispatches_are_bridged: false,
            ..eligible
        }
        .window_width(),
        None
    );
    assert_eq!(
        VulkanResidentInProcessPlacedFeedbackLoopEligibility {
            has_push_constants: true,
            ..eligible
        }
        .window_width(),
        None
    );
    assert_eq!(
        VulkanResidentInProcessPlacedFeedbackLoopEligibility {
            sampler_history_capacity: 1,
            ..eligible
        }
        .window_width(),
        None
    );
}

fn fixture_tick_dispatch_stage(stage_index: usize) -> VulkanMountedPlacedStreamTickStage {
    VulkanMountedPlacedStreamTickStage::Dispatch {
        stage_index,
        dispatch: VulkanMountedPlacedStreamTickDispatch {
            dispatch_index: stage_index,
            kernel_id: format!("kernel_{stage_index}"),
            component_id: format!("component_{stage_index}"),
            node_id: format!("node_{stage_index}"),
            op: "fixture".to_string(),
            descriptor_count: 0,
            resident_descriptor_count: 0,
            reads: Vec::new(),
            writes: Vec::new(),
        },
    }
}

#[test]
fn resident_dispatch_segments_stop_at_transport_boundaries() {
    let stages = vec![
        fixture_tick_dispatch_stage(0),
        fixture_tick_dispatch_stage(1),
        VulkanMountedPlacedStreamTickStage::PublishEdge {
            stage_index: 2,
            edge_index: 0,
            endpoint_id: "out".to_string(),
            buffer_index: 0,
            byte_capacity: 16,
            remote_device_id: "gpu1".to_string(),
            remote_component_id: "remote".to_string(),
        },
        VulkanMountedPlacedStreamTickStage::ReceiveEdge {
            stage_index: 3,
            edge_index: 1,
            endpoint_id: "in".to_string(),
            buffer_index: 0,
            byte_capacity: 16,
            remote_device_id: "gpu1".to_string(),
            remote_component_id: "remote".to_string(),
        },
        fixture_tick_dispatch_stage(4),
        fixture_tick_dispatch_stage(5),
    ];

    assert_eq!(
        resident_dispatch_segment_stage_ranges(&stages),
        vec![(0, 2), (4, 6)]
    );
}

#[test]
fn distributed_dispatches_split_resident_command_segments() {
    let stages = (0..6).map(fixture_tick_dispatch_stage).collect::<Vec<_>>();

    let ranges = resident_dispatch_segment_stage_ranges_excluding_dispatches(
        &stages,
        &BTreeSet::from([2, 4]),
    );

    assert_eq!(ranges, vec![(0, 2), (3, 4), (5, 6)]);
}

#[test]
fn distributed_dependency_topology_covers_edges_and_adjacent_dispatches() {
    let stages = (0..6).map(fixture_tick_dispatch_stage).collect::<Vec<_>>();
    let distributed_indices = BTreeSet::from([0, 2, 3, 5]);
    let distributed_stages = distributed_dispatch_stages(
        &VulkanMountedPlacedStreamTickPlan {
            backend_id: VULKAN_STREAM_CIRCUIT_BACKEND_ID.to_string(),
            device_id: "gpu0".to_string(),
            stages: stages.clone(),
            stage_count: stages.len(),
            receive_stage_count: 0,
            dispatch_stage_count: stages.len(),
            publish_stage_count: 0,
            local_edge_read_count: 0,
            local_edge_write_count: 0,
            incoming_edge_read_count: 0,
            outgoing_edge_write_count: 0,
            model_input_read_count: 0,
            model_output_write_count: 0,
            can_execute: false,
        },
        &distributed_indices,
    )
    .unwrap();
    let ranges =
        resident_dispatch_segment_stage_ranges_excluding_dispatches(&stages, &distributed_indices);
    let distributed_groups = distributed_dispatch_stage_groups(
        &distributed_stages,
        &[vec![0], vec![2], vec![3], vec![5]],
    )
    .unwrap();

    assert_eq!(ranges, vec![(1, 2), (4, 5)]);
    assert_eq!(
        distributed_dispatch_dependency_topologies(&distributed_groups, &ranges),
        BTreeMap::from([
            (
                0,
                VulkanMountedPlacedDistributedDispatchDependencies {
                    dispatch_index: 0,
                    has_owner_producer: false,
                    has_owner_continuation: true,
                },
            ),
            (
                2,
                VulkanMountedPlacedDistributedDispatchDependencies {
                    dispatch_index: 2,
                    has_owner_producer: true,
                    has_owner_continuation: false,
                },
            ),
            (
                3,
                VulkanMountedPlacedDistributedDispatchDependencies {
                    dispatch_index: 3,
                    has_owner_producer: false,
                    has_owner_continuation: true,
                },
            ),
            (
                5,
                VulkanMountedPlacedDistributedDispatchDependencies {
                    dispatch_index: 5,
                    has_owner_producer: true,
                    has_owner_continuation: false,
                },
            ),
        ])
    );
}

#[test]
fn distributed_dependency_topology_uses_composed_group_boundaries() {
    let stages = (0..6).map(fixture_tick_dispatch_stage).collect::<Vec<_>>();
    let distributed_indices = BTreeSet::from([2, 3]);
    let tick_plan = VulkanMountedPlacedStreamTickPlan {
        backend_id: VULKAN_STREAM_CIRCUIT_BACKEND_ID.to_string(),
        device_id: "gpu0".to_string(),
        stages: stages.clone(),
        stage_count: stages.len(),
        receive_stage_count: 0,
        dispatch_stage_count: stages.len(),
        publish_stage_count: 0,
        local_edge_read_count: 0,
        local_edge_write_count: 0,
        incoming_edge_read_count: 0,
        outgoing_edge_write_count: 0,
        model_input_read_count: 0,
        model_output_write_count: 0,
        can_execute: false,
    };
    let distributed_stages = distributed_dispatch_stages(&tick_plan, &distributed_indices).unwrap();
    let distributed_groups =
        distributed_dispatch_stage_groups(&distributed_stages, &[vec![2, 3]]).unwrap();
    let ranges =
        resident_dispatch_segment_stage_ranges_excluding_dispatches(&stages, &distributed_indices);

    assert_eq!(ranges, vec![(0, 2), (4, 6)]);
    assert_eq!(
        distributed_dispatch_dependency_topologies(&distributed_groups, &ranges),
        BTreeMap::from([(
            2,
            VulkanMountedPlacedDistributedDispatchDependencies {
                dispatch_index: 2,
                has_owner_producer: true,
                has_owner_continuation: true,
            },
        )])
    );
}

#[test]
fn cursor_completes_an_entire_matching_distributed_group() {
    let stages = vec![
        fixture_tick_dispatch_stage(0),
        fixture_tick_dispatch_stage(1),
    ];
    let VulkanMountedPlacedStreamTickStage::Dispatch { dispatch, .. } = &stages[0] else {
        unreachable!();
    };
    let distributed_dispatch = dispatch.clone();
    let grouped_dispatch = dispatch.clone();
    let VulkanMountedPlacedStreamTickStage::Dispatch {
        dispatch: second_dispatch,
        ..
    } = &stages[1]
    else {
        unreachable!();
    };
    let second_distributed_dispatch = second_dispatch.clone();
    let second_grouped_dispatch = second_dispatch.clone();
    let tick_plan = VulkanMountedPlacedStreamTickPlan {
        backend_id: VULKAN_STREAM_CIRCUIT_BACKEND_ID.to_string(),
        device_id: "gpu0".to_string(),
        stages,
        stage_count: 2,
        receive_stage_count: 0,
        dispatch_stage_count: 2,
        publish_stage_count: 0,
        local_edge_read_count: 0,
        local_edge_write_count: 0,
        incoming_edge_read_count: 0,
        outgoing_edge_write_count: 0,
        model_input_read_count: 0,
        model_output_write_count: 0,
        can_execute: false,
    };
    let execution_plan = VulkanMountedPlacedResidentStreamTickExecutionPlan {
        tick_plan: Arc::new(tick_plan),
        dispatch_segment_count: 0,
        dispatch_count: 0,
        distributed_dispatch_count: 2,
        dispatch_segments: Vec::new(),
        distributed_dispatch_stages: BTreeMap::from([
            (0, distributed_dispatch),
            (1, second_distributed_dispatch),
        ]),
        distributed_dispatch_groups: BTreeMap::from([(
            0,
            VulkanMountedPlacedDistributedDispatchStageGroup {
                dispatches: vec![grouped_dispatch, second_grouped_dispatch],
                end_stage_index: 2,
            },
        )]),
        distributed_dispatch_dependencies: BTreeMap::from([(
            0,
            VulkanMountedPlacedDistributedDispatchDependencies {
                dispatch_index: 0,
                has_owner_producer: false,
                has_owner_continuation: false,
            },
        )]),
    };
    let mut cursor = execution_plan.resident_stream_tick_cursor(7);

    assert_eq!(
        cursor
            .pending_distributed_dispatch(&execution_plan)
            .map(|dispatch| dispatch.dispatch_index),
        Some(0)
    );
    let error = cursor
        .complete_pending_distributed_dispatch(&execution_plan, 1)
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("expected distributed dispatch 0")
    );
    assert!(!cursor.is_completed());

    cursor
        .complete_pending_distributed_dispatch(&execution_plan, 0)
        .unwrap();
    assert!(cursor.is_completed());
    assert_eq!(cursor.completed_stage_count, 2);
}

fn assert_bf16_bytes_close(actual: &[u8], expected: &[u8], max_absolute_error: f32) {
    assert_eq!(actual.len(), expected.len());
    assert_eq!(actual.len() % 2, 0);
    for (index, (actual, expected)) in actual
        .chunks_exact(2)
        .zip(expected.chunks_exact(2))
        .enumerate()
    {
        let actual = f32::from_bits(u32::from(u16::from_le_bytes([actual[0], actual[1]])) << 16);
        let expected =
            f32::from_bits(u32::from(u16::from_le_bytes([expected[0], expected[1]])) << 16);
        assert!(
            (actual - expected).abs() <= max_absolute_error,
            "BF16 value {index} differs: actual={actual}, expected={expected}, tolerance={max_absolute_error}"
        );
    }
}

fn assert_f32_bytes_close(actual: &[u8], expected: &[u8], max_absolute_error: f32) {
    assert_eq!(actual.len(), expected.len());
    assert_eq!(actual.len() % 4, 0);
    for (index, (actual, expected)) in actual
        .chunks_exact(4)
        .zip(expected.chunks_exact(4))
        .enumerate()
    {
        let actual = f32::from_le_bytes([actual[0], actual[1], actual[2], actual[3]]);
        let expected = f32::from_le_bytes([expected[0], expected[1], expected[2], expected[3]]);
        assert!(
            (actual - expected).abs() <= max_absolute_error,
            "F32 value {index} differs: actual={actual}, expected={expected}, tolerance={max_absolute_error}"
        );
    }
}

fn numeric_state_error(actual: &[u8], expected: &[u8], dtype: &str) -> (f64, f64) {
    assert_eq!(actual.len(), expected.len());
    let (squared_error, squared_reference, max_absolute_error, count) = match dtype {
        "BF16" => actual.chunks_exact(2).zip(expected.chunks_exact(2)).fold(
            (0.0, 0.0, 0.0_f64, 0usize),
            |totals, (actual, expected)| {
                let actual = f64::from(f32::from_bits(
                    u32::from(u16::from_le_bytes([actual[0], actual[1]])) << 16,
                ));
                let expected = f64::from(f32::from_bits(
                    u32::from(u16::from_le_bytes([expected[0], expected[1]])) << 16,
                ));
                let error = (actual - expected).abs();
                (
                    totals.0 + error * error,
                    totals.1 + expected * expected,
                    totals.2.max(error),
                    totals.3 + 1,
                )
            },
        ),
        "F32" => actual.chunks_exact(4).zip(expected.chunks_exact(4)).fold(
            (0.0, 0.0, 0.0_f64, 0usize),
            |totals, (actual, expected)| {
                let actual = f64::from(f32::from_le_bytes(actual.try_into().unwrap()));
                let expected = f64::from(f32::from_le_bytes(expected.try_into().unwrap()));
                let error = (actual - expected).abs();
                (
                    totals.0 + error * error,
                    totals.1 + expected * expected,
                    totals.2.max(error),
                    totals.3 + 1,
                )
            },
        ),
        other => panic!("unsupported state dtype {other:?}"),
    };
    assert!(count > 0);
    (
        (squared_error / squared_reference.max(f64::EPSILON)).sqrt(),
        max_absolute_error,
    )
}

fn assert_f32_bits_close(
    actual: u32,
    expected: u32,
    max_absolute_error: f32,
    max_relative_error: f32,
) {
    let actual = f32::from_bits(actual);
    let expected = f32::from_bits(expected);
    let error = (actual - expected).abs();
    let tolerance = max_absolute_error.max(expected.abs() * max_relative_error);
    assert!(
        error <= tolerance,
        "F32 values differ: actual={actual}, expected={expected}, tolerance={tolerance}"
    );
}

fn compile_temperature_top_k_top_p_sampler_test_kernels(
    vocab_size: usize,
    temperature: f32,
    top_k: u32,
    top_p: f32,
    partition_count: u32,
    local_size_x: u32,
) -> Option<Vec<VulkanResidentSamplerKernelArtifact>> {
    let shader_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("shaders");
    let candidates =
        std::fs::read_to_string(shader_dir.join("temperature_top_k_candidates_f32.comp.template"))
            .ok()?
            .replace("{{VOCAB_SIZE}}", &vocab_size.to_string())
            .replace("{{TOP_K}}", &top_k.to_string())
            .replace("{{PARTITION_COUNT}}", &partition_count.to_string())
            .replace("{{LOCAL_SIZE_X}}", &local_size_x.to_string());
    let sampler = std::fs::read_to_string(
        shader_dir.join("temperature_top_k_top_p_sampler_f32.comp.template"),
    )
    .ok()?
    .replace("{{TEMPERATURE}}", &temperature.to_string())
    .replace("{{TOP_K}}", &top_k.to_string())
    .replace("{{TOP_P}}", &top_p.to_string())
    .replace("{{MIN_P}}", "0.0")
    .replace("{{PARTITION_COUNT}}", &partition_count.to_string())
    .replace("{{LOCAL_SIZE_X}}", &local_size_x.to_string());
    let compile = |suffix: &str, source: String| {
        let path = std::env::temp_dir().join(format!(
            "nerve-sampling-test-{}-{suffix}.comp",
            std::process::id()
        ));
        std::fs::write(&path, source).ok()?;
        let words = crate::vulkan_compute::compile_shader_words_from_source_path(&path);
        let _ = std::fs::remove_file(path);
        words
    };
    Some(vec![
        VulkanResidentSamplerKernelArtifact {
            role: "partition_top_k".to_string(),
            spirv_words: compile("candidates", candidates)?,
            local_size_x,
            workgroup_count_x: partition_count,
        },
        VulkanResidentSamplerKernelArtifact {
            role: "sample_candidates".to_string(),
            spirv_words: compile("sample", sampler)?,
            local_size_x,
            workgroup_count_x: 1,
        },
    ])
}

fn compile_repetition_temperature_sampler_test_kernels(
    vocab_size: usize,
    repetition_penalty: f32,
    top_k: u32,
    partition_count: u32,
    local_size_x: u32,
) -> Option<Vec<VulkanResidentSamplerKernelArtifact>> {
    let shader_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("shaders");
    let render = |template: &str| std::fs::read_to_string(shader_dir.join(template)).ok();
    let tracker = render("record_seen_token.comp.template")?
        .replace("{{VOCAB_SIZE}}", &vocab_size.to_string());
    let batch_tracker = render("record_seen_tokens_batch64.comp.template")?
        .replace("{{VOCAB_SIZE}}", &vocab_size.to_string());
    let candidates = render("temperature_top_k_candidates_repetition_f32.comp.template")?
        .replace("{{VOCAB_SIZE}}", &vocab_size.to_string())
        .replace("{{REPETITION_PENALTY}}", &repetition_penalty.to_string())
        .replace("{{PRESENCE_PENALTY}}", "0.0")
        .replace("{{TOP_K}}", &top_k.to_string())
        .replace("{{PARTITION_COUNT}}", &partition_count.to_string())
        .replace("{{LOCAL_SIZE_X}}", &local_size_x.to_string());
    let sampler = render("temperature_top_k_top_p_sampler_f32.comp.template")?
        .replace("{{TEMPERATURE}}", "1.0")
        .replace("{{TOP_K}}", &top_k.to_string())
        .replace("{{TOP_P}}", "1.0")
        .replace("{{MIN_P}}", "0.0")
        .replace("{{PARTITION_COUNT}}", &partition_count.to_string())
        .replace("{{LOCAL_SIZE_X}}", &local_size_x.to_string());
    let compile = |suffix: &str, source: String| {
        let path = std::env::temp_dir().join(format!(
            "nerve-repetition-sampling-test-{}-{suffix}.comp",
            std::process::id()
        ));
        std::fs::write(&path, source).ok()?;
        let words = crate::vulkan_compute::compile_shader_words_from_source_path(&path);
        let _ = std::fs::remove_file(path);
        words
    };
    Some(vec![
        VulkanResidentSamplerKernelArtifact {
            role: "record_current_token".to_string(),
            spirv_words: compile("tracker", tracker)?,
            local_size_x: 1,
            workgroup_count_x: 1,
        },
        VulkanResidentSamplerKernelArtifact {
            role: "record_token_batch".to_string(),
            spirv_words: compile("batch-tracker", batch_tracker)?,
            local_size_x: VULKAN_BACKEND_LOOP_MAX_WINDOW as u32,
            workgroup_count_x: 1,
        },
        VulkanResidentSamplerKernelArtifact {
            role: "partition_top_k".to_string(),
            spirv_words: compile("candidates", candidates)?,
            local_size_x,
            workgroup_count_x: partition_count,
        },
        VulkanResidentSamplerKernelArtifact {
            role: "sample_candidates".to_string(),
            spirv_words: compile("sample", sampler)?,
            local_size_x,
            workgroup_count_x: 1,
        },
    ])
}

fn compile_runtime_temperature_sampler_test_kernels(
    vocab_size: usize,
    top_k_capacity: u32,
    partition_count: u32,
    local_size_x: u32,
) -> Option<Vec<VulkanResidentSamplerKernelArtifact>> {
    let shader_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("shaders");
    let render = |template: &str| std::fs::read_to_string(shader_dir.join(template)).ok();
    let tracker = render("record_seen_token.comp.template")?
        .replace("{{VOCAB_SIZE}}", &vocab_size.to_string());
    let batch_tracker = render("record_seen_tokens_batch64.comp.template")?
        .replace("{{VOCAB_SIZE}}", &vocab_size.to_string());
    let candidates = render("temperature_top_k_candidates_runtime_f32.comp.template")?
        .replace("{{VOCAB_SIZE}}", &vocab_size.to_string())
        .replace("{{TOP_K_CAPACITY}}", &top_k_capacity.to_string())
        .replace("{{PARTITION_COUNT}}", &partition_count.to_string())
        .replace("{{LOCAL_SIZE_X}}", &local_size_x.to_string());
    let sampler = render("temperature_top_k_top_p_sampler_runtime_f32.comp.template")?
        .replace("{{TOP_K_CAPACITY}}", &top_k_capacity.to_string())
        .replace("{{PARTITION_COUNT}}", &partition_count.to_string())
        .replace("{{LOCAL_SIZE_X}}", &local_size_x.to_string());
    let compile = |suffix: &str, source: String| {
        let path = std::env::temp_dir().join(format!(
            "nerve-runtime-sampling-test-{}-{suffix}.comp",
            std::process::id()
        ));
        std::fs::write(&path, source).ok()?;
        let words = crate::vulkan_compute::compile_shader_words_from_source_path(&path);
        let _ = std::fs::remove_file(path);
        words
    };
    Some(vec![
        VulkanResidentSamplerKernelArtifact {
            role: "runtime_record_current_token".to_string(),
            spirv_words: compile("tracker", tracker)?,
            local_size_x: 1,
            workgroup_count_x: 1,
        },
        VulkanResidentSamplerKernelArtifact {
            role: "runtime_record_token_batch".to_string(),
            spirv_words: compile("batch-tracker", batch_tracker)?,
            local_size_x: VULKAN_BACKEND_LOOP_MAX_WINDOW as u32,
            workgroup_count_x: 1,
        },
        VulkanResidentSamplerKernelArtifact {
            role: "runtime_partition_top_k".to_string(),
            spirv_words: compile("candidates", candidates)?,
            local_size_x,
            workgroup_count_x: partition_count,
        },
        VulkanResidentSamplerKernelArtifact {
            role: "runtime_sample_candidates".to_string(),
            spirv_words: compile("sample", sampler)?,
            local_size_x,
            workgroup_count_x: 1,
        },
    ])
}

fn greedy_sampler_test_kernels(spirv_words: Vec<u32>) -> Vec<VulkanResidentSamplerKernelArtifact> {
    vec![VulkanResidentSamplerKernelArtifact {
        role: "sample_logits".to_string(),
        spirv_words,
        local_size_x: 1_024,
        workgroup_count_x: 1,
    }]
}

fn sampler_test_hash_u32(mut value: u32) -> u32 {
    value ^= value >> 16;
    value = value.wrapping_mul(0x7feb_352d);
    value ^= value >> 15;
    value = value.wrapping_mul(0x846c_a68b);
    value ^= value >> 16;
    value
}

fn fixture_model_index_path() -> PathBuf {
    compiled_artifact_dir(
        "NERVE_TEST_LOWERED_DIR",
        "lowered",
        "execution_graph.circuits.json",
    )
    .join("execution_graph.circuits.json")
}

fn fixture_model_tensor_index_path() -> PathBuf {
    compiled_artifact_dir("NERVE_TEST_TRANSPILED_DIR", "transpiled", "tensors.json")
        .join("tensors.json")
}

fn fixture_model_package_manifest_path() -> PathBuf {
    compiled_artifact_dir(
        "NERVE_TEST_PACKAGE_DIR",
        "packages",
        "vulkan_resident_package.json",
    )
    .join("vulkan_resident_package.json")
}

fn fixture_model_package_manifest() -> VulkanResidentModelPackageManifest {
    VulkanResidentModelPackageManifest::from_json_file(fixture_model_package_manifest_path())
        .unwrap()
}

fn fixture_model_runtime_model() -> VulkanResidentRuntimeModel {
    fixture_model_package_manifest()
        .mount_runtime_graph_controls(None, &BTreeMap::new(), &[], None)
        .unwrap()
}

fn fixture_model_runtime_model_with_placement(
    placement: StreamCircuitPlacementSpec,
) -> VulkanResidentRuntimeModel {
    let manifest = fixture_model_package_manifest();
    let source_graph = manifest
        .circuit_graph
        .to_resolved_lowered_execution_graph(PathBuf::from("."))
        .unwrap();
    let runtime_graph = source_graph
        .runtime_graph_from_placement(&placement)
        .unwrap();
    manifest.mount_runtime_graph(&runtime_graph).unwrap()
}

fn fixture_model_execution_graph() -> ResolvedLoweredExecutionGraph {
    let full = ResolvedLoweredExecutionGraph::from_index_file(fixture_model_index_path()).unwrap();
    let processor_ids = full
        .circuits
        .iter()
        .filter(|artifact| artifact.circuit.runtime_role.is_signal_processor())
        .map(|artifact| artifact.component.id.as_str())
        .collect::<BTreeSet<_>>();
    let circuits = full
        .circuits
        .iter()
        .filter(|artifact| processor_ids.contains(artifact.component.id.as_str()))
        .cloned()
        .collect::<Vec<_>>();
    let mut index = full.index.clone();
    index.graph.circuits = circuits
        .iter()
        .map(|artifact| artifact.component.clone())
        .collect();
    index.graph.edges = full
        .index
        .graph
        .edges
        .iter()
        .filter(|edge| {
            edge.connection.is_forward()
                && processor_ids.contains(edge.source.component_id.as_str())
                && processor_ids.contains(edge.destination.component_id.as_str())
        })
        .cloned()
        .collect();
    index.graph.boundary = StreamCircuitGraphBoundary {
        external_inputs: execution_boundary_inputs(&full, &processor_ids),
        public_outputs: execution_boundary_outputs(&full, &processor_ids),
    };
    let mut operator_counts = BTreeMap::new();
    for artifact in &circuits {
        *operator_counts
            .entry(artifact.component.operator_type.clone())
            .or_insert(0) += 1;
    }
    index.summary = LoweredExecutionGraphSummary {
        circuit_count: circuits.len(),
        operator_counts,
    };
    ResolvedLoweredExecutionGraph {
        artifact_root: full.artifact_root,
        index,
        circuits,
    }
}

fn copy_package_integrity_artifacts(
    source_root: &Path,
    destination_root: &Path,
    manifest: &VulkanResidentModelPackageManifest,
) {
    for relative_path in manifest.artifact_integrity.files.keys() {
        let destination = destination_root.join(relative_path);
        std::fs::create_dir_all(destination.parent().unwrap()).unwrap();
        std::fs::copy(source_root.join(relative_path), destination).unwrap();
    }
}
