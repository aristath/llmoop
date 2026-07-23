use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use super::*;
use crate::stream_circuit::{
    ResolvedLoweredExecutionGraph, StreamCircuitPlacementSpec, StreamCircuitRuntimeGraph,
};
use crate::stream_plan::{StreamCircuitExecutionPlan, StreamCircuitResourcePlan};
use crate::test_support::compiled_artifact_dir;

const FIXTURE_MODEL_TOKEN_EMBEDDING_TRANSDUCER_ID: &str = "input_transducer.token_embedding";
const FIXTURE_MODEL_OUTPUT_EMBEDDING_NORM_TRANSDUCER_ID: &str = "output_transducer.embedding_norm";
const FIXTURE_MODEL_TIED_OUTPUT_PROJECTION_TRANSDUCER_ID: &str =
    "output_transducer.tied_output_projection";
const FIXTURE_MODEL_GREEDY_SAMPLER_COMPONENT_ID: &str = "greedy_sampler";
const FIXTURE_MODEL_EMBED_TOKENS_TENSOR: &str = "model.embed_tokens.weight";
const FIXTURE_MODEL_INPUT_FRAME_SIGNAL: &str = "input_frame";
const FIXTURE_MODEL_OUTPUT_FRAME_SIGNAL: &str = "output_frame";
const FIXTURE_MODEL_HIDDEN_SIZE: usize = 1_024;

#[test]
fn package_loader_rejects_stale_compiler_contracts_before_package_setup() {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let manifest_path = std::env::temp_dir().join(format!(
        "nerve-stale-compiler-contract-{}-{unique}.json",
        std::process::id()
    ));

    std::fs::write(
        &manifest_path,
        serde_json::to_vec(&serde_json::json!({
            "schema": "nerve.vulkan_resident_model_package.v2"
        }))
        .unwrap(),
    )
    .unwrap();
    let schema_error = VulkanResidentModelPackageManifest::from_json_file(&manifest_path)
        .unwrap_err()
        .to_string();
    assert!(schema_error.contains("recompile the model"));

    std::fs::write(
        &manifest_path,
        serde_json::to_vec(&serde_json::json!({
            "schema": VULKAN_RESIDENT_MODEL_PACKAGE_MANIFEST_SCHEMA,
            "compiler_fingerprint": "stale"
        }))
        .unwrap(),
    )
    .unwrap();
    let fingerprint_error = VulkanResidentModelPackageManifest::from_json_file(&manifest_path)
        .unwrap_err()
        .to_string();
    assert!(fingerprint_error.contains("does not match runtime fingerprint"));
    assert!(fingerprint_error.contains("recompile the model"));

    std::fs::remove_file(manifest_path).unwrap();
}

#[test]
fn loaded_artifact_manifest_preserves_compiled_launch_geometry() {
    let loaded = VulkanLoadedReusableKernelArtifactManifest {
        schema: VULKAN_REUSABLE_KERNEL_ARTIFACT_MANIFEST_SCHEMA.to_string(),
        backend_id: VULKAN_STREAM_CIRCUIT_BACKEND_ID.to_string(),
        artifacts: vec![VulkanLoadedReusableKernelArtifact {
            artifact: VulkanReusableKernelArtifact {
                family_id: "sparse-moe-gate-up".to_string(),
                op: "sparse_moe_gate_up".to_string(),
                path: "kernels/sparse-moe-gate-up.spv".to_string(),
                entry_point: DEFAULT_SPIRV_ENTRY_POINT.to_string(),
                local_size_x: 64,
                workgroup_count_x: 2_048,
                descriptor_signature: Vec::new(),
                push_constants: Vec::new(),
                uses_stream_tick: false,
            },
            resolved_path: PathBuf::from("kernels/sparse-moe-gate-up.spv"),
            words: vec![0x0723_0203],
        }],
        total_word_count: 1,
    };

    let physical = loaded.artifact_manifest();

    assert_eq!(physical.artifacts.len(), 1);
    assert_eq!(physical.artifacts[0].workgroup_count_x, 2_048);
    assert_eq!(physical.artifacts[0].local_size_x, 64);
}

#[test]
fn component_batch_signal_liveness_reuses_only_compatible_dead_buffers() {
    let key = |signal_id: &str| VulkanComponentBatchSignalKey::Activation {
        component_id: "component".to_string(),
        signal_id: signal_id.to_string(),
    };
    let lifetimes = vec![
        VulkanComponentBatchSignalLifetime {
            key: key("first"),
            frame_byte_capacity: 4_096,
            host_visible: false,
            first_dispatch: 0,
            last_dispatch: 2,
        },
        VulkanComponentBatchSignalLifetime {
            key: key("overlapping"),
            frame_byte_capacity: 4_096,
            host_visible: false,
            first_dispatch: 2,
            last_dispatch: 3,
        },
        VulkanComponentBatchSignalLifetime {
            key: key("reusable"),
            frame_byte_capacity: 4_096,
            host_visible: false,
            first_dispatch: 3,
            last_dispatch: 4,
        },
        VulkanComponentBatchSignalLifetime {
            key: key("different_size"),
            frame_byte_capacity: 8_192,
            host_visible: false,
            first_dispatch: 5,
            last_dispatch: 6,
        },
        VulkanComponentBatchSignalLifetime {
            key: VulkanComponentBatchSignalKey::IncomingEdge(7),
            frame_byte_capacity: 4_096,
            host_visible: true,
            first_dispatch: 5,
            last_dispatch: 6,
        },
    ];

    let (indices, buffers) = allocate_component_batch_signal_lifetimes(lifetimes);

    assert_eq!(buffers.len(), 4);
    assert_ne!(indices[&key("first")], indices[&key("overlapping")]);
    assert_eq!(indices[&key("first")], indices[&key("reusable")]);
    assert_ne!(indices[&key("first")], indices[&key("different_size")]);
    assert_ne!(
        indices[&key("first")],
        indices[&VulkanComponentBatchSignalKey::IncomingEdge(7)]
    );
}

#[test]
fn component_batch_execution_uses_standalone_component_submissions() {
    let span = |component_id: &str,
                dispatch_index: usize,
                step_start: usize,
                step_end: usize,
                distributed: bool| VulkanComponentBatchDispatchSpan {
        component_id: component_id.to_string(),
        dispatch_index,
        step_start,
        step_end,
        distributed,
    };
    let spans = vec![
        span("layer_00", 0, 0, 2, false),
        span("layer_00", 1, 2, 5, false),
        span("layer_01", 2, 5, 7, false),
        span("layer_01", 3, 7, 7, true),
        span("layer_01", 4, 7, 10, false),
        span("layer_02", 5, 10, 12, false),
    ];

    assert_eq!(
        component_batch_execution_units(&spans).unwrap(),
        vec![
            VulkanComponentBatchExecutionUnit::LocalComponent {
                component_id: "layer_00".to_string(),
                step_start: 0,
                step_end: 5,
            },
            VulkanComponentBatchExecutionUnit::LocalComponent {
                component_id: "layer_01".to_string(),
                step_start: 5,
                step_end: 7,
            },
            VulkanComponentBatchExecutionUnit::DistributedDispatch { dispatch_index: 3 },
            VulkanComponentBatchExecutionUnit::LocalComponent {
                component_id: "layer_01".to_string(),
                step_start: 7,
                step_end: 10,
            },
            VulkanComponentBatchExecutionUnit::LocalComponent {
                component_id: "layer_02".to_string(),
                step_start: 10,
                step_end: 12,
            },
        ]
    );
}

#[test]
fn component_batch_execution_does_not_create_empty_local_submissions() {
    let spans = vec![
        VulkanComponentBatchDispatchSpan {
            component_id: "layer_00".to_string(),
            dispatch_index: 0,
            step_start: 0,
            step_end: 0,
            distributed: true,
        },
        VulkanComponentBatchDispatchSpan {
            component_id: "layer_01".to_string(),
            dispatch_index: 1,
            step_start: 0,
            step_end: 0,
            distributed: true,
        },
    ];

    assert_eq!(
        component_batch_execution_units(&spans).unwrap(),
        vec![
            VulkanComponentBatchExecutionUnit::DistributedDispatch { dispatch_index: 0 },
            VulkanComponentBatchExecutionUnit::DistributedDispatch { dispatch_index: 1 },
        ]
    );
}

#[test]
fn component_batch_execution_submits_only_distributed_group_leaders() {
    let spans = (0..3)
        .map(|dispatch_index| VulkanComponentBatchDispatchSpan {
            component_id: "layer_00".to_string(),
            dispatch_index,
            step_start: 0,
            step_end: 0,
            distributed: true,
        })
        .collect::<Vec<_>>();

    assert_eq!(
        component_batch_execution_units_for_distributed_groups(&spans, &BTreeSet::from([0, 2]),)
            .unwrap(),
        vec![
            VulkanComponentBatchExecutionUnit::DistributedDispatch { dispatch_index: 0 },
            VulkanComponentBatchExecutionUnit::DistributedDispatch { dispatch_index: 2 },
        ]
    );
}

#[test]
fn component_batch_execution_rejects_noncontiguous_dispatch_steps() {
    let spans = vec![
        VulkanComponentBatchDispatchSpan {
            component_id: "layer_00".to_string(),
            dispatch_index: 0,
            step_start: 0,
            step_end: 2,
            distributed: false,
        },
        VulkanComponentBatchDispatchSpan {
            component_id: "layer_00".to_string(),
            dispatch_index: 1,
            step_start: 3,
            step_end: 4,
            distributed: false,
        },
    ];

    let error = component_batch_execution_units(&spans).unwrap_err();
    assert!(error.to_string().contains("starts at step 3, expected 2"));
}

const FIXTURE_MODEL_FRAME_BYTES: usize = FIXTURE_MODEL_HIDDEN_SIZE * 2;
const FIXTURE_MODEL_LOGITS_BYTES: usize = 65_536 * 4;
const FIXTURE_MODEL_SAMPLER_OUTPUT_BYTES: usize = 16;
const FIXTURE_MODEL_EMBED_TOKENS_BYTES: usize = 65_536 * FIXTURE_MODEL_FRAME_BYTES;

#[test]
fn speculative_verification_commits_through_the_first_mismatch() {
    let result = verify_speculative_token_prefix(&[11, 12, 13], &[11, 99, 88, 77]).unwrap();

    assert_eq!(result.accepted_draft_count, 1);
    assert_eq!(result.committed_target_tick_count, 2);
    assert_eq!(result.emitted_token_ids, [11, 99]);
}

#[test]
fn speculative_verification_emits_the_bonus_token_when_all_drafts_match() {
    let result = verify_speculative_token_prefix(&[11, 12], &[11, 12, 13]).unwrap();

    assert_eq!(result.accepted_draft_count, 2);
    assert_eq!(result.committed_target_tick_count, 3);
    assert_eq!(result.emitted_token_ids, [11, 12, 13]);
}

#[test]
fn speculative_verification_rejects_incomplete_target_results() {
    let error = verify_speculative_token_prefix(&[11, 12], &[11, 12]).unwrap_err();

    assert!(
        error
            .to_string()
            .contains("2 draft tokens but 2 target predictions; expected 3")
    );
}

#[test]
fn speculative_verification_stops_at_the_first_emitted_stop_token() {
    let mut result = verify_speculative_token_prefix(&[11, 12], &[11, 12, 99]).unwrap();

    truncate_speculative_verification_at_stop(&mut result, &BTreeSet::from([11]));

    assert_eq!(result.accepted_draft_count, 1);
    assert_eq!(result.committed_target_tick_count, 1);
    assert_eq!(result.emitted_token_ids, [11]);
}

#[test]
fn component_batches_never_select_a_numerically_unproven_kernel() {
    let artifact =
        |lane_tile_width, exact_primary_equivalence| VulkanResidentComponentBatchKernelArtifact {
            component_id: "processor".to_string(),
            node_id: "project".to_string(),
            execution_domain: VulkanResidentComponentKernelExecutionDomain::DecodeAndPrefill,
            batch_mode: VulkanResidentComponentKernelBatchMode::WeightShared,
            lane_tile_width,
            exact_primary_equivalence,
            exact_causal_sequence_equivalence: exact_primary_equivalence,
            device_requirements: VulkanResidentVulkanDeviceRequirements::default(),
            stages: Vec::new(),
        };
    let artifacts = vec![
        artifact(64, false),
        artifact(2, true),
        artifact(4, true),
        artifact(8, true),
        artifact(16, true),
    ];

    let verification = select_component_batch_kernel_artifact(
        &artifacts,
        "processor",
        "project",
        VulkanComponentBatchExecutionMode::IndependentCandidates,
        6,
    )
    .unwrap();
    assert_eq!(verification.lane_tile_width, 8);
    assert!(verification.exact_primary_equivalence);

    let causal = select_component_batch_kernel_artifact(
        &artifacts,
        "processor",
        "project",
        VulkanComponentBatchExecutionMode::CausalSequence,
        6,
    )
    .unwrap();
    assert_eq!(causal.lane_tile_width, 16);
    assert!(causal.exact_primary_equivalence);

    let heterogeneous = select_component_batch_kernel_artifact_where(
        &artifacts,
        "processor",
        "project",
        VulkanComponentBatchExecutionMode::CausalSequence,
        6,
        |artifact| artifact.lane_tile_width != 64,
    )
    .unwrap();
    assert_eq!(heterogeneous.lane_tile_width, 16);
}

#[test]
fn component_batches_select_only_artifacts_for_the_requested_execution_domain() {
    let artifact = |execution_domain, lane_tile_width| VulkanResidentComponentBatchKernelArtifact {
        component_id: "processor".to_string(),
        node_id: "project".to_string(),
        execution_domain,
        batch_mode: VulkanResidentComponentKernelBatchMode::WeightShared,
        lane_tile_width,
        exact_primary_equivalence: true,
        exact_causal_sequence_equivalence: true,
        device_requirements: VulkanResidentVulkanDeviceRequirements::default(),
        stages: Vec::new(),
    };
    let artifacts = vec![
        artifact(VulkanResidentComponentKernelExecutionDomain::Prefill, 4),
        artifact(VulkanResidentComponentKernelExecutionDomain::Decode, 8),
        artifact(
            VulkanResidentComponentKernelExecutionDomain::DecodeAndPrefill,
            16,
        ),
    ];

    let decode = select_component_batch_kernel_artifact(
        &artifacts,
        "processor",
        "project",
        VulkanComponentBatchExecutionMode::IndependentCandidates,
        4,
    )
    .unwrap();
    assert_eq!(
        decode.execution_domain,
        VulkanResidentComponentKernelExecutionDomain::Decode
    );
    assert_eq!(decode.lane_tile_width, 8);

    let prefill = select_component_batch_kernel_artifact(
        &artifacts,
        "processor",
        "project",
        VulkanComponentBatchExecutionMode::CausalSequence,
        4,
    )
    .unwrap();
    assert_eq!(
        prefill.execution_domain,
        VulkanResidentComponentKernelExecutionDomain::DecodeAndPrefill
    );
    assert_eq!(prefill.lane_tile_width, 16);
}

#[test]
fn component_batches_use_causal_exactness_for_temporal_prefill_kernels() {
    let artifacts = vec![VulkanResidentComponentBatchKernelArtifact {
        component_id: "processor".to_string(),
        node_id: "attention".to_string(),
        execution_domain: VulkanResidentComponentKernelExecutionDomain::Prefill,
        batch_mode: VulkanResidentComponentKernelBatchMode::CausalScan,
        lane_tile_width: 64,
        exact_primary_equivalence: false,
        exact_causal_sequence_equivalence: true,
        device_requirements: VulkanResidentVulkanDeviceRequirements::default(),
        stages: Vec::new(),
    }];

    assert!(
        select_component_batch_kernel_artifact(
            &artifacts,
            "processor",
            "attention",
            VulkanComponentBatchExecutionMode::IndependentCandidates,
            4,
        )
        .is_none()
    );
    let causal = select_component_batch_kernel_artifact(
        &artifacts,
        "processor",
        "attention",
        VulkanComponentBatchExecutionMode::CausalSequence,
        4,
    )
    .unwrap();
    assert!(causal.exact_causal_sequence_equivalence);
    assert!(!causal.exact_primary_equivalence);
}

#[test]
fn component_batch_execution_mode_follows_runtime_activation_shape() {
    let prefill = RuntimeStreamActivationBatchKind::PrefillChunk {
        execution_class_id: "package".to_string(),
        token_count: 8,
    };
    let decode = RuntimeStreamActivationBatchKind::DecodeFeedback {
        execution_class_id: "package".to_string(),
        max_tokens: 4,
    };

    assert_eq!(
        VulkanComponentBatchExecutionMode::from_runtime_activation_batch_kind(&prefill),
        VulkanComponentBatchExecutionMode::CausalSequence
    );
    assert_eq!(
        VulkanComponentBatchExecutionMode::from_runtime_activation_batch_kind(&decode),
        VulkanComponentBatchExecutionMode::IndependentCandidates
    );
}

#[test]
fn component_batch_execution_contract_requires_matching_shader_mode() {
    let execution = |batch_mode, batch_shader_path: Option<String>| {
        let batch_implementations = batch_shader_path
            .into_iter()
            .map(|shader_path| VulkanResidentComponentBatchImplementationSpec {
                execution_domain: VulkanResidentComponentKernelExecutionDomain::DecodeAndPrefill,
                lane_tile_width: 16,
                exact_primary_equivalence: true,
                exact_causal_sequence_equivalence: true,
                device_requirements: VulkanResidentVulkanDeviceRequirements::default(),
                stages: vec![VulkanResidentComponentBatchStageSpec {
                    shader_path,
                    local_size_x: 64,
                    workgroup_count_x: 1,
                }],
            })
            .collect();
        vec![VulkanResidentComponentExecutionSpec {
            component_id: "processor".to_string(),
            operator_type: "fixture".to_string(),
            implementation: "exact_reference".to_string(),
            kernels: vec![VulkanResidentComponentKernelSpec {
                execution_index: 0,
                node_id: "project".to_string(),
                op: "linear".to_string(),
                execution_domain: VulkanResidentComponentKernelExecutionDomain::Decode,
                shader_path: "shaders/project.spv".to_string(),
                local_size_x: 64,
                workgroup_count_x: 1,
                batch_mode,
                batch_implementations,
            }],
        }]
    };

    validate_component_executions(
        "fixture",
        &execution(VulkanResidentComponentKernelBatchMode::SerialLanes, None),
    )
    .unwrap();
    validate_component_executions(
        "fixture",
        &execution(
            VulkanResidentComponentKernelBatchMode::WeightShared,
            Some("shaders/project_batch.spv".to_string()),
        ),
    )
    .unwrap();
    validate_component_executions(
        "fixture",
        &execution(
            VulkanResidentComponentKernelBatchMode::CausalScan,
            Some("shaders/project_scan.spv".to_string()),
        ),
    )
    .unwrap();

    let serial_error = validate_component_executions(
        "fixture",
        &execution(
            VulkanResidentComponentKernelBatchMode::SerialLanes,
            Some("shaders/project_batch.spv".to_string()),
        ),
    )
    .unwrap_err();
    assert!(serial_error.to_string().contains("invalid SerialLanes"));

    let batch_error = validate_component_executions(
        "fixture",
        &execution(VulkanResidentComponentKernelBatchMode::WeightShared, None),
    )
    .unwrap_err();
    assert!(batch_error.to_string().contains("invalid WeightShared"));
}

#[test]
fn component_batch_control_preserves_temporal_position_and_capacity() {
    let bytes = component_batch_control_bytes(64, 0x1122_3344_5566_7788, 65_536);

    assert_eq!(&bytes[0..4], &64u32.to_le_bytes());
    assert_eq!(&bytes[4..12], &0x1122_3344_5566_7788u64.to_le_bytes());
    assert_eq!(&bytes[12..16], &65_536u32.to_le_bytes());
}

#[test]
fn distributed_batch_output_binding_repeats_the_full_lane_stride() {
    let (offset, byte_capacity) =
        distributed_batch_shard_output_binding_range(8_192, 4, 2_048, 2_048).unwrap();

    assert_eq!(offset, 2_048);
    assert_eq!(byte_capacity, 26_624);
    assert_eq!(offset + byte_capacity, 28_672);
    assert!(offset + byte_capacity <= 4 * 8_192);
}

#[test]
fn distributed_batch_output_binding_rejects_a_shard_past_the_frame() {
    let error = distributed_batch_shard_output_binding_range(8_192, 4, 7_168, 2_048).unwrap_err();

    assert!(error.to_string().contains("exceeds frame capacity 8192"));
}

#[test]
fn distributed_batch_workgroups_preserve_the_compiled_row_granularity() {
    assert_eq!(
        distributed_batch_rows_per_workgroup(32_768, 512, "layer", "ffn").unwrap(),
        64
    );

    let error = distributed_batch_rows_per_workgroup(32_769, 512, "layer", "ffn").unwrap_err();
    assert!(
        error
            .to_string()
            .contains("cannot partition 32769 rows across 512 workgroups")
    );
}
