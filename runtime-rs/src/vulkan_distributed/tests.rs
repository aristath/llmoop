#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use sha2::{Digest, Sha256};

    use super::*;
    use crate::stream_plan::TensorMetadata;
    use crate::vulkan_stream_circuit::{
        VulkanKernelDescriptorUsage, VulkanKernelScalarBinding, VulkanKernelScalarSource,
        VulkanResolvedDescriptorBinding, VulkanReusableKernelArtifact,
    };

    #[test]
    fn placed_components_do_not_implicitly_shard_their_internal_dispatches() {
        let device_ids = vec!["gpu0".to_string(), "gpu1".to_string()];
        let plan = VulkanDistributedExecutionPlan::for_placed_components(&device_ids, 256).unwrap();

        assert_eq!(plan.device_ids, device_ids);
        assert!(plan.dispatches.is_empty());
        assert!(plan.dispatch_groups.is_empty());
        assert_eq!(plan.shared_input_byte_capacity, 0);
        assert_eq!(plan.shared_output_byte_capacity, 0);
        assert_eq!(plan.distributed_parameter_byte_count, 0);
    }

    #[test]
    fn timeline_dependency_clock_is_monotonic_and_refuses_wraparound() {
        let clock = VulkanDistributedDependencyClock::new();

        assert_eq!(clock.reserve("owner", 7).unwrap(), 1);
        assert_eq!(clock.reserve("owner", 7).unwrap(), 2);
        clock.validate_advance(64, "owner", 7).unwrap();
        clock.advance(64);
        assert_eq!(clock.reserve("owner", 7).unwrap(), 67);

        clock.next_value.set(u64::MAX);
        let error = clock.reserve("owner", 7).unwrap_err();
        assert!(error.to_string().contains("exhausted its timeline"));
        assert_eq!(clock.next_value.get(), u64::MAX);
    }

    #[test]
    fn plans_balanced_parameter_and_output_shards_from_compiled_contracts() {
        let plan = fixture_plan("row_major");

        assert_eq!(plan.dispatches.len(), 1);
        assert_eq!(plan.shared_input_byte_capacity, 8);
        assert_eq!(plan.shared_output_byte_capacity, 24);
        assert_eq!(plan.storage_buffer_offset_alignment, 4);
        assert_eq!(plan.distributed_parameter_byte_count, 192);
        let dispatch = &plan.dispatches[0];
        assert_eq!(dispatch.owner_device_id, "owner");
        assert_eq!(dispatch.row_alignment, 2);
        assert_eq!(dispatch.input_activation.component_id, "component");
        assert_eq!(dispatch.input_activation.signal_id, "normalized");
        assert_eq!(dispatch.input_activation.slot, 0);
        assert_eq!(dispatch.output_activation.component_id, "component");
        assert_eq!(dispatch.output_activation.signal_id, "hidden");
        assert_eq!(dispatch.output_activation.slot, 1);
        assert_eq!(
            dispatch
                .shards
                .iter()
                .map(|shard| (
                    shard.device_id.as_str(),
                    shard.row_start,
                    shard.row_count,
                    shard.workgroup_count_x,
                    shard.output_byte_offset,
                    shard.output_byte_count,
                ))
                .collect::<Vec<_>>(),
            vec![
                ("owner", 0, 4, 2, 0, 8),
                ("helper-a", 4, 4, 2, 8, 8),
                ("helper-b", 8, 2, 1, 16, 4),
                ("helper-c", 10, 2, 1, 20, 4),
            ]
        );
        assert_eq!(
            dispatch.shards[1]
                .parameters
                .iter()
                .map(|fragment| (
                    fragment.binding,
                    fragment.tensor.as_str(),
                    fragment.byte_offset,
                    fragment.byte_count,
                ))
                .collect::<Vec<_>>(),
            vec![(2, "gate", 32, 32), (3, "up", 32, 32)]
        );
    }

    #[test]
    fn plans_sparse_expert_ranges_with_shared_routes_and_full_outputs() {
        let activation = |binding, signal: &str, slot, bytes| VulkanResolvedDescriptorBinding {
            binding,
            usage: if binding == 2 {
                VulkanKernelDescriptorUsage::OutputSignal
            } else {
                VulkanKernelDescriptorUsage::InputSignal
            },
            name: signal.to_string(),
            resource: VulkanDescriptorResourceAddress::ActivationSlot {
                component_id: "moe".to_string(),
                signal_id: signal.to_string(),
                slot,
                byte_capacity: bytes,
                signal_byte_capacity: bytes,
            },
        };
        let parameter = |binding, tensor: &str, bytes| VulkanResolvedDescriptorBinding {
            binding,
            usage: VulkanKernelDescriptorUsage::Parameter,
            name: tensor.to_string(),
            resource: VulkanDescriptorResourceAddress::PermanentParameter {
                param_id: tensor.to_string(),
                tensor: tensor.to_string(),
                byte_count: Some(bytes),
            },
        };
        let mut prepared = VulkanPreparedDispatchPlan {
            backend_id: "vulkan_stream_circuit".to_string(),
            reusable_family_count: 1,
            dispatches: vec![VulkanPreparedDispatch {
                dispatch_index: 9,
                kernel_id: "moe.sparse-down".to_string(),
                component_id: "moe".to_string(),
                circuit_id: "moe-circuit".to_string(),
                node_index: 4,
                node_id: "sparse-down".to_string(),
                op: "sparse_moe_down".to_string(),
                reusable_family_id: "sparse-family".to_string(),
                artifact_path: "sparse.spv".to_string(),
                entry_point: "main".to_string(),
                local_size_x: 64,
                descriptors: vec![
                    activation(0, "intermediates", 0, 8192),
                    activation(1, "routes", 1, 32),
                    activation(2, "outputs", 2, 32768),
                    parameter(3, "expert-weight", 256 * 2048 * 512),
                    parameter(4, "expert-scale", 256 * 16 * 4 * 2),
                ],
                push_constants: vec![VulkanKernelScalarBinding {
                    name: "expert_start".to_string(),
                    scalar_type: "u32".to_string(),
                    source: VulkanKernelScalarSource::PushConstant,
                }],
                uses_stream_tick: false,
            }],
            total_descriptor_count: 5,
        };
        let tensor_index = TensorIndex {
            schema: "nerve.tensor_index.v1".to_string(),
            tensors: BTreeMap::from([
                (
                    "expert-weight".to_string(),
                    TensorMetadata {
                        dtype: "F8_E4M3".to_string(),
                        shape: vec![256, 2048, 512],
                        logical_shape: None,
                        parameter_count: Some(256 * 2048 * 512),
                        byte_count: Some(256 * 2048 * 512),
                        data_offsets: Some(vec![0, 256 * 2048 * 512]),
                        source_file: Some("weights.safetensors".to_string()),
                        data_sha256: None,
                        layout: Some("row_major".to_string()),
                    },
                ),
                (
                    "expert-scale".to_string(),
                    TensorMetadata {
                        dtype: "BF16".to_string(),
                        shape: vec![256, 16, 4],
                        logical_shape: None,
                        parameter_count: Some(256 * 16 * 4),
                        byte_count: Some(256 * 16 * 4 * 2),
                        data_offsets: Some(vec![0, 256 * 16 * 4 * 2]),
                        source_file: Some("weights.safetensors".to_string()),
                        data_sha256: None,
                        layout: Some("row_major".to_string()),
                    },
                ),
            ]),
        };
        let artifacts =
            VulkanReusableKernelArtifactManifest::new(vec![VulkanReusableKernelArtifact {
                family_id: "sparse-family".to_string(),
                op: "sparse_moe_down".to_string(),
                path: "sparse.spv".to_string(),
                entry_point: "main".to_string(),
                local_size_x: 64,
                workgroup_count_x: 8192,
                descriptor_signature: Vec::new(),
                push_constants: vec![VulkanKernelScalarBinding {
                    name: "expert_start".to_string(),
                    scalar_type: "u32".to_string(),
                    source: VulkanKernelScalarSource::PushConstant,
                }],
                uses_stream_tick: false,
            }]);

        let plan = VulkanDistributedExecutionPlan::from_prepared_plans(
            &[("owner", &prepared)],
            &tensor_index,
            &artifacts,
            &["owner".to_string(), "helper".to_string()],
            256,
        )
        .unwrap();

        assert_eq!(plan.dispatches.len(), 1);
        let dispatch = &plan.dispatches[0];
        assert_eq!(
            dispatch.distribution,
            VulkanDistributedDispatchDistribution::ExpertRange
        );
        assert_eq!(dispatch.input_activation.binding, 0);
        assert_eq!(dispatch.auxiliary_input_activations[0].binding, 1);
        assert_eq!(dispatch.output_activation.binding, 2);
        assert_eq!(dispatch.shards.len(), 2);
        assert_eq!(dispatch.shards[0].device_id, "owner");
        assert_eq!(dispatch.shards[0].row_start, 0);
        assert_eq!(dispatch.shards[0].row_count, 128);
        assert_eq!(dispatch.shards[0].base_workgroup_z, 0);
        assert_eq!(dispatch.shards[1].device_id, "helper");
        assert_eq!(dispatch.shards[1].row_start, 128);
        assert_eq!(dispatch.shards[1].row_count, 128);
        assert_eq!(dispatch.shards[1].base_workgroup_z, 128);
        assert!(
            dispatch
                .shards
                .iter()
                .all(|shard| shard.workgroup_count_x == 8192
                    && shard.output_byte_offset == 0
                    && shard.output_byte_count == 32768)
        );
        assert_eq!(
            dispatch.shards[1].parameters[0].byte_offset,
            128 * 2048 * 512
        );
        assert_eq!(
            dispatch.shards[1].parameters[0].byte_count,
            128 * 2048 * 512
        );
        assert_eq!(
            dispatch.shards[1].parameters[1].byte_offset,
            128 * 16 * 4 * 2
        );
        assert_eq!(
            dispatch.shards[1].parameters[1].byte_count,
            128 * 16 * 4 * 2
        );

        prepared.dispatches[0].push_constants.clear();
        let legacy_plan = VulkanDistributedExecutionPlan::from_prepared_plans(
            &[("owner", &prepared)],
            &tensor_index,
            &artifacts,
            &["owner".to_string(), "helper".to_string()],
            256,
        )
        .unwrap();
        assert!(
            legacy_plan.dispatches.is_empty(),
            "sparse expert sharding requires the explicit expert_start contract"
        );
    }

    #[test]
    fn groups_only_adjacent_dataflow_compatible_expert_dispatches() {
        let mut producer = fixture_plan("row_major").dispatches.remove(0);
        producer.dispatch_index = 7;
        producer.distribution = VulkanDistributedDispatchDistribution::ExpertRange;
        producer.output_activation = producer.input_activation.clone();
        producer.output_activation.binding = 1;
        let mut consumer = producer.clone();
        consumer.dispatch_index = 8;
        consumer.node_id = "consumer".to_string();
        consumer.input_activation = producer.output_activation.clone();
        consumer.input_activation.binding = 0;

        let groups = distributed_dispatch_groups(&[producer.clone(), consumer.clone()]);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].dispatch_indices(), vec![7, 8]);

        let mut non_adjacent = consumer.clone();
        non_adjacent.dispatch_index = 9;
        assert_eq!(
            distributed_dispatch_groups(&[producer.clone(), non_adjacent]).len(),
            2
        );

        let mut different_dataflow = consumer.clone();
        different_dataflow.input_activation.signal_id = "another-signal".to_string();
        assert_eq!(
            distributed_dispatch_groups(&[producer.clone(), different_dataflow]).len(),
            2
        );

        let mut different_shards = consumer.clone();
        different_shards.shards[1].row_start += 1;
        assert_eq!(
            distributed_dispatch_groups(&[producer.clone(), different_shards]).len(),
            2
        );

        let mut row_distributed = consumer;
        row_distributed.distribution = VulkanDistributedDispatchDistribution::OutputRows;
        assert_eq!(
            distributed_dispatch_groups(&[producer, row_distributed]).len(),
            2
        );
    }

    #[test]
    fn distributed_shards_always_start_with_the_dispatch_owner() {
        let tensor_index = fixture_tensor_index("row_major");
        let prepared_plan = fixture_prepared_plan();
        let artifact_manifest = fixture_artifact_manifest();
        let plan = VulkanDistributedExecutionPlan::from_prepared_plans(
            &[("owner", &prepared_plan)],
            &tensor_index,
            &artifact_manifest,
            &[
                "helper-a".to_string(),
                "helper-b".to_string(),
                "owner".to_string(),
            ],
            4,
        )
        .unwrap();

        assert_eq!(plan.dispatches.len(), 1);
        assert_eq!(plan.dispatches[0].shards[0].device_id, "owner");
        assert!(
            plan.dispatches[0]
                .shards
                .iter()
                .any(|shard| shard.device_id != "owner")
        );
    }

    #[test]
    fn distributed_planner_keeps_unsplittable_dispatch_on_its_owner() {
        let tensor_index = fixture_tensor_index("row_major");
        let prepared_plan = fixture_prepared_plan();
        let artifact_manifest = fixture_artifact_manifest();
        let plan = VulkanDistributedExecutionPlan::from_prepared_plans(
            &[("owner", &prepared_plan)],
            &tensor_index,
            &artifact_manifest,
            &["helper".to_string(), "owner".to_string()],
            1024,
        )
        .unwrap();

        assert!(plan.dispatches.is_empty());
        assert_eq!(plan.distributed_parameter_byte_count, 0);
    }

    #[test]
    fn preserves_packed_row_pairs_at_shard_boundaries() {
        let plan = fixture_plan("vulkan_bf16_row_pair_u32");

        assert_eq!(plan.dispatches[0].row_alignment, 2);
        assert!(
            plan.dispatches[0]
                .shards
                .iter()
                .all(|shard| shard.row_start % 2 == 0 && shard.row_count % 2 == 0)
        );
    }

    #[test]
    fn aligns_shared_output_offsets_and_keeps_a_workgroup_aligned_tail() {
        let plan = fixture_plan_result_with_alignment("row_major", 16).unwrap();
        let dispatch = &plan.dispatches[0];

        assert_eq!(dispatch.row_alignment, 8);
        assert_eq!(
            dispatch
                .shards
                .iter()
                .map(|shard| (
                    shard.device_id.as_str(),
                    shard.row_start,
                    shard.row_count,
                    shard.workgroup_count_x,
                    shard.output_byte_offset,
                ))
                .collect::<Vec<_>>(),
            vec![("owner", 0, 8, 4, 0), ("helper-a", 8, 4, 2, 16)]
        );
        assert!(dispatch.shards.iter().all(|shard| {
            shard
                .output_byte_offset
                .is_multiple_of(plan.storage_buffer_offset_alignment)
        }));
    }

    #[test]
    fn plans_one_shared_allocation_per_owner_activation_slot() {
        let execution_plan = fixture_plan("row_major");

        let activation_plan =
            VulkanDistributedActivationBufferPlan::from_execution_plan(&execution_plan).unwrap();

        assert_eq!(activation_plan.allocation_count, 2);
        assert_eq!(activation_plan.import_count, 8);
        assert_eq!(activation_plan.reference_count, 2);
        assert_eq!(activation_plan.total_shared_byte_capacity, 32);
        assert_eq!(
            activation_plan.allocation("owner", "component", 0).unwrap(),
            &VulkanDistributedActivationBufferAllocation {
                owner_device_id: "owner".to_string(),
                component_id: "component".to_string(),
                slot: 0,
                byte_capacity: 8,
                signal_ids: vec!["normalized".to_string()],
                device_ids: vec![
                    "helper-a".to_string(),
                    "helper-b".to_string(),
                    "helper-c".to_string(),
                    "owner".to_string(),
                ],
                input_use_count: 1,
                output_use_count: 0,
            }
        );
        assert_eq!(
            activation_plan
                .allocation("owner", "component", 1)
                .unwrap()
                .output_use_count,
            1
        );
    }

    #[test]
    fn rejects_zero_lane_shared_activation_allocations_before_device_access() {
        let execution_plan = fixture_plan("row_major");
        let activation_plan =
            VulkanDistributedActivationBufferPlan::from_execution_plan(&execution_plan).unwrap();

        let error =
            VulkanDistributedActivationBuffers::allocate_for_lanes(&activation_plan, 0, |_| {
                Err::<&VulkanComputeDevice, _>("device resolver must not run")
            })
            .err()
            .unwrap();

        assert_eq!(
            error.to_string(),
            "distributed activation lane capacity must not be zero"
        );
    }

    #[test]
    fn reuses_shared_activation_allocations_across_repeated_dispatches() {
        let mut execution_plan = fixture_plan("row_major");
        let mut repeated = execution_plan.dispatches[0].clone();
        repeated.dispatch_index = 8;
        repeated.input_activation.signal_id = "normalized-again".to_string();
        execution_plan.dispatches.push(repeated);

        let activation_plan =
            VulkanDistributedActivationBufferPlan::from_execution_plan(&execution_plan).unwrap();

        assert_eq!(activation_plan.allocation_count, 2);
        assert_eq!(activation_plan.import_count, 8);
        assert_eq!(activation_plan.reference_count, 4);
        assert_eq!(activation_plan.total_shared_byte_capacity, 32);
        let input = activation_plan.allocation("owner", "component", 0).unwrap();
        assert_eq!(input.input_use_count, 2);
        assert_eq!(
            input.signal_ids,
            vec!["normalized".to_string(), "normalized-again".to_string()]
        );
    }

    #[test]
    fn rejects_conflicting_capacities_for_the_same_activation_slot() {
        let mut execution_plan = fixture_plan("row_major");
        let mut repeated = execution_plan.dispatches[0].clone();
        repeated.dispatch_index = 8;
        repeated.input_activation.byte_capacity = 16;
        execution_plan.dispatches.push(repeated);

        let error = VulkanDistributedActivationBufferPlan::from_execution_plan(&execution_plan)
            .unwrap_err();

        assert!(
            error
                .to_string()
                .contains("activation component.slot_0 has conflicting capacities 8 and 16")
        );
    }

    #[test]
    fn rejects_non_contiguous_projection_layouts() {
        let error = fixture_plan_result("column_major").unwrap_err();

        assert!(
            error
                .to_string()
                .contains("tensor \"gate\" has non-shardable layout Some(\"column_major\")")
        );
    }

    #[test]
    fn keeps_push_constant_dispatches_on_their_owner_device() {
        let mut prepared_plan = fixture_prepared_plan();
        prepared_plan.dispatches[0].push_constants = vec![VulkanKernelScalarBinding {
            name: "stream_tick".to_string(),
            scalar_type: "u64".to_string(),
            source: VulkanKernelScalarSource::PushConstant,
        }];

        let plan = VulkanDistributedExecutionPlan::from_prepared_plans(
            &[("owner", &prepared_plan)],
            &fixture_tensor_index("row_major"),
            &fixture_artifact_manifest(),
            &["owner".to_string(), "helper".to_string()],
            4,
        )
        .unwrap();

        assert!(plan.dispatches.is_empty());
        assert_eq!(plan.distributed_parameter_byte_count, 0);
    }

    #[test]
    fn keeps_quantized_parallel_projections_on_their_owner_device() {
        let mut prepared_plan = fixture_prepared_plan();
        let gate = prepared_plan.dispatches[0].descriptors[2].clone();
        let up = prepared_plan.dispatches[0].descriptors[3].clone();
        let parameter =
            |binding, param_id: &str, tensor: &str, byte_count| VulkanResolvedDescriptorBinding {
                binding,
                usage: VulkanKernelDescriptorUsage::Parameter,
                name: param_id.to_string(),
                resource: VulkanDescriptorResourceAddress::PermanentParameter {
                    param_id: param_id.to_string(),
                    tensor: tensor.to_string(),
                    byte_count: Some(byte_count),
                },
            };
        prepared_plan.dispatches[0].descriptors = vec![
            prepared_plan.dispatches[0].descriptors[0].clone(),
            prepared_plan.dispatches[0].descriptors[1].clone(),
            VulkanResolvedDescriptorBinding { binding: 2, ..gate },
            parameter(3, "gate_scale", "gate_scale", 2),
            VulkanResolvedDescriptorBinding { binding: 4, ..up },
            parameter(5, "up_scale", "up_scale", 2),
        ];
        let mut tensor_index = fixture_tensor_index("row_major");
        for tensor in ["gate", "up"] {
            let metadata = tensor_index.tensors.get_mut(tensor).unwrap();
            metadata.dtype = "F8_E4M3".to_string();
            metadata.byte_count = Some(48);
        }
        let scale = TensorMetadata {
            dtype: "BF16".to_string(),
            shape: vec![1, 1],
            logical_shape: None,
            parameter_count: Some(1),
            byte_count: Some(2),
            data_offsets: Some(vec![0, 2]),
            source_file: Some("weights.safetensors".to_string()),
            data_sha256: None,
            layout: Some("row_major".to_string()),
        };
        tensor_index
            .tensors
            .insert("gate_scale".to_string(), scale.clone());
        tensor_index.tensors.insert("up_scale".to_string(), scale);

        let plan = VulkanDistributedExecutionPlan::from_prepared_plans(
            &[("owner", &prepared_plan)],
            &tensor_index,
            &fixture_artifact_manifest(),
            &["owner".to_string(), "helper".to_string()],
            4,
        )
        .unwrap();

        assert!(plan.dispatches.is_empty());
        assert_eq!(plan.distributed_parameter_byte_count, 0);
    }

    #[test]
    fn immutable_parameter_shards_are_reused_by_duplicated_components() {
        let mut execution_plan = fixture_plan("row_major");
        let mut duplicate = execution_plan.dispatches[0].clone();
        duplicate.dispatch_index = 8;
        duplicate.component_id = "duplicated-component".to_string();
        duplicate.node_id = "duplicated-ffn".to_string();
        execution_plan.dispatches.push(duplicate);

        let allocation_plan = VulkanDistributedParameterAllocationPlan::from_execution_plan(
            &execution_plan,
            &fixture_tensor_index("row_major"),
        )
        .unwrap();

        assert_eq!(allocation_plan.allocation_count, 8);
        assert_eq!(allocation_plan.tensor_count, 2);
        assert_eq!(allocation_plan.total_byte_capacity, 192);
        assert!(
            allocation_plan
                .allocations
                .iter()
                .all(|allocation| allocation.use_count == 2)
        );
    }

    #[test]
    fn loads_each_tensor_once_and_streams_verified_shards_to_devices() {
        let execution_plan = fixture_plan("row_major");
        let fixture = DistributedStorageFixture::new();
        let allocation_plan = VulkanDistributedParameterAllocationPlan::from_execution_plan(
            &execution_plan,
            &fixture.tensor_index,
        )
        .unwrap();
        let mut writes = Vec::new();

        let report = allocation_plan
            .load_from_tensor_index(&fixture.tensor_index, |allocation, bytes| {
                writes.push((allocation.clone(), bytes.to_vec()));
                Ok(())
            })
            .unwrap();

        assert_eq!(report.tensor_count, 2);
        assert_eq!(report.source_file_count, 1);
        assert_eq!(report.allocation_count, 8);
        assert_eq!(report.write_count, 8);
        assert_eq!(report.total_bytes_read, 192);
        assert_eq!(report.total_bytes_written, 192);
        let (allocation, bytes) = writes
            .iter()
            .find(|(allocation, _)| {
                allocation.device_id == "helper-a" && allocation.tensor == "gate"
            })
            .unwrap();
        assert_eq!(allocation.byte_offset, 32);
        assert_eq!(allocation.byte_count, 32);
        assert_eq!(bytes, &fixture.gate_bytes[32..64]);
    }

    #[test]
    fn excludes_full_parameters_only_when_all_prepared_uses_are_distributed() {
        let execution_plan = fixture_plan("row_major");
        let prepared_plan = fixture_prepared_plan();

        let exclusions =
            VulkanDistributedParameterExclusionPlan::from_execution_and_prepared_plans(
                &execution_plan,
                &[("owner", &prepared_plan)],
                &fixture_tensor_index("row_major"),
            )
            .unwrap();

        assert_eq!(exclusions.device_count, 1);
        assert_eq!(exclusions.unique_tensor_count, 2);
        assert_eq!(exclusions.excluded_full_allocation_count, 2);
        assert_eq!(exclusions.excluded_full_byte_capacity, 192);
        assert_eq!(
            exclusions.tensors_for_device("owner"),
            BTreeSet::from(["gate".to_string(), "up".to_string()])
        );
        assert!(exclusions.tensors_for_device("helper-a").is_empty());
    }

    #[test]
    fn refuses_to_exclude_a_tensor_still_used_by_a_canonical_dispatch() {
        let execution_plan = fixture_plan("row_major");
        let mut prepared_plan = fixture_prepared_plan();
        let mut canonical = prepared_plan.dispatches[0].clone();
        canonical.dispatch_index = 8;
        canonical.node_index = 4;
        canonical.node_id = "canonical-use".to_string();
        canonical.op = "linear".to_string();
        canonical.descriptors.retain(|descriptor| {
            matches!(
                descriptor.resource,
                VulkanDescriptorResourceAddress::PermanentParameter { .. }
            )
        });
        prepared_plan.dispatches.push(canonical);

        let error = VulkanDistributedParameterExclusionPlan::from_execution_and_prepared_plans(
            &execution_plan,
            &[("owner", &prepared_plan)],
            &fixture_tensor_index("row_major"),
        )
        .unwrap_err();

        assert!(
            error
                .to_string()
                .contains("canonical dispatch component.canonical-use still uses it")
        );
    }

    fn fixture_plan(layout: &str) -> VulkanDistributedExecutionPlan {
        fixture_plan_result(layout).unwrap()
    }

    fn fixture_plan_result(
        layout: &str,
    ) -> Result<VulkanDistributedExecutionPlan, VulkanDistributedPlanError> {
        fixture_plan_result_with_alignment(layout, 4)
    }

    fn fixture_plan_result_with_alignment(
        layout: &str,
        storage_buffer_offset_alignment: usize,
    ) -> Result<VulkanDistributedExecutionPlan, VulkanDistributedPlanError> {
        let tensor_index = fixture_tensor_index(layout);
        let prepared_plan = fixture_prepared_plan();
        let artifact_manifest = fixture_artifact_manifest();
        VulkanDistributedExecutionPlan::from_prepared_plans(
            &[("owner", &prepared_plan)],
            &tensor_index,
            &artifact_manifest,
            &[
                "owner".to_string(),
                "helper-a".to_string(),
                "helper-b".to_string(),
                "helper-c".to_string(),
            ],
            storage_buffer_offset_alignment,
        )
    }

    fn fixture_prepared_plan() -> VulkanPreparedDispatchPlan {
        let activation =
            |binding, name: &str, signal: &str, bytes| VulkanResolvedDescriptorBinding {
                binding,
                usage: if binding == 0 {
                    VulkanKernelDescriptorUsage::InputSignal
                } else {
                    VulkanKernelDescriptorUsage::OutputSignal
                },
                name: name.to_string(),
                resource: VulkanDescriptorResourceAddress::ActivationSlot {
                    component_id: "component".to_string(),
                    signal_id: signal.to_string(),
                    slot: binding,
                    byte_capacity: bytes,
                    signal_byte_capacity: bytes,
                },
            };
        let parameter = |binding, tensor: &str| VulkanResolvedDescriptorBinding {
            binding,
            usage: VulkanKernelDescriptorUsage::Parameter,
            name: tensor.to_string(),
            resource: VulkanDescriptorResourceAddress::PermanentParameter {
                param_id: tensor.to_string(),
                tensor: tensor.to_string(),
                byte_count: Some(96),
            },
        };
        VulkanPreparedDispatchPlan {
            backend_id: "vulkan_stream_circuit".to_string(),
            reusable_family_count: 1,
            dispatches: vec![VulkanPreparedDispatch {
                dispatch_index: 7,
                kernel_id: "component.ffn".to_string(),
                component_id: "component".to_string(),
                circuit_id: "circuit".to_string(),
                node_index: 3,
                node_id: "ffn".to_string(),
                op: DISTRIBUTABLE_PARALLEL_PROJECTION_OP.to_string(),
                reusable_family_id: "family".to_string(),
                artifact_path: "ffn.spv".to_string(),
                entry_point: "main".to_string(),
                local_size_x: 64,
                descriptors: vec![
                    activation(0, "input", "normalized", 8),
                    activation(1, "output", "hidden", 24),
                    parameter(2, "gate"),
                    parameter(3, "up"),
                ],
                push_constants: Vec::new(),
                uses_stream_tick: false,
            }],
            total_descriptor_count: 4,
        }
    }

    fn fixture_artifact_manifest() -> VulkanReusableKernelArtifactManifest {
        VulkanReusableKernelArtifactManifest::new(vec![VulkanReusableKernelArtifact {
            family_id: "family".to_string(),
            op: DISTRIBUTABLE_PARALLEL_PROJECTION_OP.to_string(),
            path: "ffn.spv".to_string(),
            entry_point: "main".to_string(),
            local_size_x: 64,
            workgroup_count_x: 6,
            descriptor_signature: Vec::new(),
            push_constants: Vec::new(),
            uses_stream_tick: false,
        }])
    }

    fn fixture_tensor_index(layout: &str) -> TensorIndex {
        let metadata = |layout: &str| TensorMetadata {
            dtype: "BF16".to_string(),
            shape: vec![12, 4],
            logical_shape: None,
            parameter_count: Some(48),
            byte_count: Some(96),
            data_offsets: Some(vec![0, 96]),
            source_file: Some("weights.safetensors".to_string()),
            data_sha256: None,
            layout: Some(layout.to_string()),
        };
        TensorIndex {
            schema: "nerve.tensor_index.v1".to_string(),
            tensors: BTreeMap::from([
                ("gate".to_string(), metadata(layout)),
                ("up".to_string(), metadata(layout)),
            ]),
        }
    }

    struct DistributedStorageFixture {
        root: PathBuf,
        tensor_index: TensorIndex,
        gate_bytes: Vec<u8>,
    }

    impl DistributedStorageFixture {
        fn new() -> Self {
            let unique = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let root = std::env::temp_dir().join(format!(
                "nerve-distributed-storage-{}-{unique}",
                std::process::id()
            ));
            fs::create_dir_all(&root).unwrap();
            let source = root.join("weights.safetensors");
            let gate_bytes = (0..96).map(|value| value as u8).collect::<Vec<_>>();
            let up_bytes = (0..96)
                .map(|value| 255u8.wrapping_sub(value as u8))
                .collect::<Vec<_>>();
            let header = b"{}";
            let mut file_bytes = Vec::new();
            file_bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
            file_bytes.extend_from_slice(header);
            file_bytes.extend_from_slice(&gate_bytes);
            file_bytes.extend_from_slice(&up_bytes);
            fs::write(&source, file_bytes).unwrap();
            let metadata = |data_offsets: Vec<usize>, bytes: &[u8]| TensorMetadata {
                dtype: "BF16".to_string(),
                shape: vec![12, 4],
                logical_shape: None,
                parameter_count: Some(48),
                byte_count: Some(96),
                data_offsets: Some(data_offsets),
                source_file: Some(source.to_string_lossy().into_owned()),
                data_sha256: Some(
                    Sha256::digest(bytes)
                        .iter()
                        .map(|byte| format!("{byte:02x}"))
                        .collect(),
                ),
                layout: Some("row_major".to_string()),
            };
            let tensor_index = TensorIndex {
                schema: "nerve.tensor_index.v1".to_string(),
                tensors: BTreeMap::from([
                    ("gate".to_string(), metadata(vec![0, 96], &gate_bytes)),
                    ("up".to_string(), metadata(vec![96, 192], &up_bytes)),
                ]),
            };
            Self {
                root,
                tensor_index,
                gate_bytes,
            }
        }
    }

    impl Drop for DistributedStorageFixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.root);
        }
    }
}
