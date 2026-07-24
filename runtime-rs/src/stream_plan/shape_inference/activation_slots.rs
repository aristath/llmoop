fn planned_activation_slots(
    activation_plan: &CircuitActivationPlan,
    frame: &ActivationFramePlan,
) -> Vec<PlannedActivationSlot> {
    let mut signals_by_slot: BTreeMap<usize, Vec<String>> = BTreeMap::new();
    for assignment in &frame.assignments {
        signals_by_slot
            .entry(assignment.slot)
            .or_default()
            .push(assignment.signal_id.clone());
    }

    signals_by_slot
        .into_iter()
        .map(|(slot, signal_ids)| {
            let mut max_elements = Some(0usize);
            let mut max_bytes = Some(0usize);
            for signal_id in &signal_ids {
                let signal = activation_plan.signal(signal_id);
                let elements = signal
                    .and_then(|signal| signal.shape.as_ref())
                    .and_then(|shape| product(shape));
                match (max_elements, elements) {
                    (Some(max), Some(elements)) => max_elements = Some(max.max(elements)),
                    _ => max_elements = None,
                }
                let bytes = elements.and_then(|elements| {
                    signal
                        .and_then(|signal| signal.element_bytes)
                        .and_then(|element_bytes| elements.checked_mul(element_bytes))
                });
                match (max_bytes, bytes) {
                    (Some(max), Some(bytes)) => max_bytes = Some(max.max(bytes)),
                    _ => max_bytes = None,
                }
            }
            PlannedActivationSlot {
                slot,
                signal_ids,
                max_elements,
                max_bytes,
            }
        })
        .collect()
}

fn validate_node_dependencies(
    component_id: &str,
    node: &CircuitNode,
    available: &BTreeSet<String>,
    state_ids: &BTreeSet<&String>,
    param_ids: &BTreeSet<&String>,
) -> Result<(), CircuitPlanError> {
    for input in &node.inputs {
        if !available.contains(input) {
            return Err(CircuitPlanError(format!(
                "{} node {} input {:?} is not available at its schedule position",
                component_id, node.id, input
            )));
        }
    }
    for param in &node.params {
        if !param_ids.contains(param) {
            return Err(CircuitPlanError(format!(
                "{} node {} parameter {:?} is not declared",
                component_id, node.id, param
            )));
        }
    }
    for state in node.state_reads.iter().chain(node.state_writes.iter()) {
        if !state_ids.contains(state) {
            return Err(CircuitPlanError(format!(
                "{} node {} state {:?} is not declared",
                component_id, node.id, state
            )));
        }
    }
    if node.outputs.is_empty() {
        return Err(CircuitPlanError(format!(
            "{} node {} has no outputs",
            component_id, node.id
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use crate::stream_circuit::ResolvedLoweredExecutionGraph;
    use crate::test_support::compiled_artifact_dir;

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

    #[test]
    fn package_tensor_index_rejects_sources_outside_the_package() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "nerve-package-tensor-index-{}-{unique}",
            std::process::id()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let index_path = root.join("tensors.json");
        std::fs::write(
            &index_path,
            serde_json::to_vec(&serde_json::json!({
                "schema": TENSOR_INDEX_SCHEMA,
                "tensors": {
                    "weight": {
                        "dtype": "BF16",
                        "shape": [4, 4],
                        "source_file": "../outside.safetensors"
                    }
                }
            }))
            .unwrap(),
        )
        .unwrap();

        let error = TensorIndex::from_package_json_file(&index_path).unwrap_err();

        assert!(error.0.contains("must stay inside the package"));

        std::fs::write(
            &index_path,
            serde_json::to_vec(&serde_json::json!({
                "schema": TENSOR_INDEX_SCHEMA,
                "tensors": {
                    "weight": {
                        "dtype": "BF16",
                        "shape": [4, 4],
                        "source_file": "weights/weight.safetensors"
                    }
                }
            }))
            .unwrap(),
        )
        .unwrap();
        let error = TensorIndex::from_package_json_file(&index_path).unwrap_err();
        assert!(error.0.contains("no valid data SHA-256"));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn infers_unequal_fused_projection_split_shapes() {
        let node = crate::stream_circuit::CircuitNode {
            id: "qkv_split".to_string(),
            op: "split".to_string(),
            inputs: vec!["qkv".to_string()],
            outputs: vec!["q".to_string(), "k".to_string(), "v".to_string()],
            params: Vec::new(),
            state_reads: Vec::new(),
            state_writes: Vec::new(),
            attrs: serde_json::json!({"part_widths": [16, 8, 8]}),
        };
        let signals = BTreeMap::from([(
            "qkv".to_string(),
            PlannedSignal {
                id: "qkv".to_string(),
                producer: SignalProducer::BoundaryInput,
                consumers: vec!["qkv_split".to_string()],
                shape: Some(vec![32]),
                element_bytes: None,
                storage: SignalStorage::Boundary,
                is_boundary_output: false,
            },
        )]);

        assert_eq!(
            infer_split_output_shapes("layer_00", &node, &signals).unwrap(),
            vec![Some(vec![16]), Some(vec![8]), Some(vec![8])]
        );
    }

    #[test]
    fn infers_fp8_quantization_representation_shapes() {
        let node = crate::stream_circuit::CircuitNode {
            id: "quantize".to_string(),
            op: "quantize_fp8_e4m3".to_string(),
            inputs: vec!["hidden".to_string()],
            outputs: vec!["hidden_fp8".to_string(), "hidden_scale".to_string()],
            params: Vec::new(),
            state_reads: Vec::new(),
            state_writes: Vec::new(),
            attrs: serde_json::json!({
                "element_count": 5120,
                "block_columns": 128,
                "output_element_bytes": [1, 4]
            }),
        };
        let signals = BTreeMap::from([(
            "hidden".to_string(),
            PlannedSignal {
                id: "hidden".to_string(),
                producer: SignalProducer::BoundaryInput,
                consumers: vec!["quantize".to_string()],
                shape: Some(vec![5120]),
                element_bytes: Some(2),
                storage: SignalStorage::Boundary,
                is_boundary_output: false,
            },
        )]);

        assert_eq!(
            infer_node_output_shapes(
                "layer_00",
                &node,
                &signals,
                &BTreeMap::new(),
                None,
            )
            .unwrap(),
            vec![Some(vec![5120]), Some(vec![40])]
        );
    }

    #[test]
    fn sparse_moe_signal_shapes_scale_with_selected_routes_not_all_experts() {
        let signals = BTreeMap::new();
        let params = BTreeMap::new();
        let attrs = serde_json::json!({
            "hidden_size": 2048,
            "intermediate_size": 512,
            "num_experts": 256,
            "experts_per_token": 8
        });
        let node = |op: &str| crate::stream_circuit::CircuitNode {
            id: op.to_string(),
            op: op.to_string(),
            inputs: Vec::new(),
            outputs: vec!["output".to_string()],
            params: Vec::new(),
            state_reads: Vec::new(),
            state_writes: Vec::new(),
            attrs: attrs.clone(),
        };

        assert_eq!(
            infer_node_output_shapes("layer_00", &node("moe_topk"), &signals, &params, None)
                .unwrap(),
            vec![Some(vec![8, 2])]
        );
        assert_eq!(
            infer_node_output_shapes(
                "layer_00",
                &node("sparse_moe_gate_up"),
                &signals,
                &params,
                None,
            )
            .unwrap(),
            vec![Some(vec![8, 512])]
        );
        assert_eq!(
            infer_node_output_shapes(
                "layer_00",
                &node("sparse_moe_down"),
                &signals,
                &params,
                None,
            )
            .unwrap(),
            vec![Some(vec![8, 2048])]
        );
    }

    #[test]
    fn infers_fused_append_attention_output_shape_from_query_frame() {
        let node = crate::stream_circuit::CircuitNode {
            id: "append_attention".to_string(),
            op: "append_scaled_dot_product_attention".to_string(),
            inputs: vec![
                "q".to_string(),
                "k".to_string(),
                "v".to_string(),
                "kv_memory".to_string(),
            ],
            outputs: vec!["attention_out".to_string()],
            params: Vec::new(),
            state_reads: vec!["kv_memory".to_string()],
            state_writes: vec!["kv_memory".to_string()],
            attrs: serde_json::json!({
                "attention": {
                    "query_heads": 16,
                    "key_value_heads": 8,
                    "head_width": 64
                }
            }),
        };
        let signals = BTreeMap::from([(
            "q".to_string(),
            PlannedSignal {
                id: "q".to_string(),
                producer: SignalProducer::BoundaryInput,
                consumers: vec!["append_attention".to_string()],
                shape: Some(vec![1024]),
                element_bytes: None,
                storage: SignalStorage::Boundary,
                is_boundary_output: false,
            },
        )]);

        assert_eq!(
            infer_node_output_shapes("attention", &node, &signals, &BTreeMap::new(), None,)
                .unwrap(),
            vec![Some(vec![1024])]
        );
    }

    #[test]
    fn infers_per_head_softplus_gate_output_shape_from_attention_frame() {
        let node = crate::stream_circuit::CircuitNode {
            id: "attention_output_gate".to_string(),
            op: "softplus_multiply".to_string(),
            inputs: vec!["attention_out".to_string(), "attention_gate".to_string()],
            outputs: vec!["attention_gated".to_string()],
            params: Vec::new(),
            state_reads: Vec::new(),
            state_writes: Vec::new(),
            attrs: serde_json::json!({
                "head_count": 48,
                "head_width": 128
            }),
        };
        let signals = BTreeMap::from([
            (
                "attention_out".to_string(),
                PlannedSignal {
                    id: "attention_out".to_string(),
                    producer: SignalProducer::BoundaryInput,
                    consumers: vec!["attention_output_gate".to_string()],
                    shape: Some(vec![6144]),
                    element_bytes: None,
                    storage: SignalStorage::Boundary,
                    is_boundary_output: false,
                },
            ),
            (
                "attention_gate".to_string(),
                PlannedSignal {
                    id: "attention_gate".to_string(),
                    producer: SignalProducer::BoundaryInput,
                    consumers: vec!["attention_output_gate".to_string()],
                    shape: Some(vec![48]),
                    element_bytes: None,
                    storage: SignalStorage::Boundary,
                    is_boundary_output: false,
                },
            ),
        ]);

        assert_eq!(
            infer_node_output_shapes("layer_00", &node, &signals, &BTreeMap::new(), None,).unwrap(),
            vec![Some(vec![6144])]
        );
    }

    #[test]
    fn rejects_parallel_linear_branch_metadata_mismatch_without_tensor_index() {
        let node = crate::stream_circuit::CircuitNode {
            id: "qkv".to_string(),
            op: "parallel_linear_3way".to_string(),
            inputs: vec!["hidden".to_string()],
            outputs: vec!["q".to_string(), "k".to_string(), "v".to_string()],
            params: vec![
                "q_weight".to_string(),
                "k_weight".to_string(),
                "v_weight".to_string(),
            ],
            state_reads: Vec::new(),
            state_writes: Vec::new(),
            attrs: serde_json::json!({"branch_count": 2}),
        };

        let error = infer_parallel_linear_output_shapes(
            "attention",
            &node,
            &BTreeMap::new(),
            &BTreeMap::new(),
            None,
        )
        .unwrap_err();

        assert!(error.0.contains("expected 3"), "{}", error.0);
    }

    #[test]
    fn infers_fp8_parallel_linear_output_shapes_from_branch_weight_parameters() {
        let node = crate::stream_circuit::CircuitNode {
            id: "qk".to_string(),
            op: "parallel_linear_2way".to_string(),
            inputs: vec!["hidden".to_string()],
            outputs: vec!["q".to_string(), "k".to_string()],
            params: vec![
                "q_weight".to_string(),
                "q_weight_scale_inv".to_string(),
                "k_weight".to_string(),
                "k_weight_scale_inv".to_string(),
            ],
            state_reads: Vec::new(),
            state_writes: Vec::new(),
            attrs: serde_json::json!({
                "branch_count": 2,
                "branch_parameter_counts": [2, 2]
            }),
        };
        let signals = BTreeMap::from([(
            "hidden".to_string(),
            PlannedSignal {
                id: "hidden".to_string(),
                producer: SignalProducer::BoundaryInput,
                consumers: vec!["qk".to_string()],
                shape: Some(vec![5120]),
                element_bytes: None,
                storage: SignalStorage::Boundary,
                is_boundary_output: false,
            },
        )]);
        let params = BTreeMap::from([
            (
                "q_weight".to_string(),
                ParameterRef {
                    tensor: Some("q.weight".to_string()),
                    role: None,
                    extra: serde_json::Map::new(),
                },
            ),
            (
                "q_weight_scale_inv".to_string(),
                ParameterRef {
                    tensor: Some("q.weight_scale_inv".to_string()),
                    role: None,
                    extra: serde_json::Map::new(),
                },
            ),
            (
                "k_weight".to_string(),
                ParameterRef {
                    tensor: Some("k.weight".to_string()),
                    role: None,
                    extra: serde_json::Map::new(),
                },
            ),
            (
                "k_weight_scale_inv".to_string(),
                ParameterRef {
                    tensor: Some("k.weight_scale_inv".to_string()),
                    role: None,
                    extra: serde_json::Map::new(),
                },
            ),
        ]);
        let weight = |rows| TensorMetadata {
            dtype: "F8_E4M3".to_string(),
            shape: vec![rows, 5120],
            logical_shape: None,
            parameter_count: None,
            byte_count: None,
            data_offsets: None,
            source_file: None,
            data_sha256: None,
            layout: None,
        };
        let scale = TensorMetadata {
            dtype: "BF16".to_string(),
            shape: vec![40, 40],
            logical_shape: None,
            parameter_count: None,
            byte_count: None,
            data_offsets: None,
            source_file: None,
            data_sha256: None,
            layout: None,
        };
        let tensor_index = TensorIndex {
            schema: TENSOR_INDEX_SCHEMA.to_string(),
            tensors: BTreeMap::from([
                ("q.weight".to_string(), weight(5120)),
                ("q.weight_scale_inv".to_string(), scale.clone()),
                ("k.weight".to_string(), weight(5120)),
                ("k.weight_scale_inv".to_string(), scale),
            ]),
        };

        assert_eq!(
            infer_parallel_linear_output_shapes(
                "attention",
                &node,
                &signals,
                &params,
                Some(&tensor_index),
            )
            .unwrap(),
            vec![Some(vec![5120]), Some(vec![5120])]
        );
    }

    #[test]
    fn infers_fused_parallel_ffn_projection_output_shape() {
        let node = crate::stream_circuit::CircuitNode {
            id: "fused_ffn".to_string(),
            op: "parallel_linear_silu_multiply".to_string(),
            inputs: vec!["hidden".to_string()],
            outputs: vec!["ffn_hidden".to_string()],
            params: vec!["gate_weight".to_string(), "up_weight".to_string()],
            state_reads: Vec::new(),
            state_writes: Vec::new(),
            attrs: serde_json::json!({
                "branch_count": 2,
                "element_count": 2560,
                "intermediate_rounding": "BF16"
            }),
        };
        let signals = BTreeMap::from([(
            "hidden".to_string(),
            PlannedSignal {
                id: "hidden".to_string(),
                producer: SignalProducer::BoundaryInput,
                consumers: vec!["fused_ffn".to_string()],
                shape: Some(vec![1024]),
                element_bytes: None,
                storage: SignalStorage::Boundary,
                is_boundary_output: false,
            },
        )]);
        let params = BTreeMap::from([
            (
                "gate_weight".to_string(),
                ParameterRef {
                    tensor: Some("gate.weight".to_string()),
                    role: None,
                    extra: serde_json::Map::new(),
                },
            ),
            (
                "up_weight".to_string(),
                ParameterRef {
                    tensor: Some("up.weight".to_string()),
                    role: None,
                    extra: serde_json::Map::new(),
                },
            ),
        ]);
        let tensor_index = TensorIndex {
            schema: TENSOR_INDEX_SCHEMA.to_string(),
            tensors: BTreeMap::from([
                (
                    "gate.weight".to_string(),
                    TensorMetadata {
                        dtype: "BF16".to_string(),
                        shape: vec![2560, 1024],
                        logical_shape: None,
                        parameter_count: None,
                        byte_count: None,
                        data_offsets: None,
                        source_file: None,
                        data_sha256: None,
                        layout: None,
                    },
                ),
                (
                    "up.weight".to_string(),
                    TensorMetadata {
                        dtype: "BF16".to_string(),
                        shape: vec![2560, 1024],
                        logical_shape: None,
                        parameter_count: None,
                        byte_count: None,
                        data_offsets: None,
                        source_file: None,
                        data_sha256: None,
                        layout: None,
                    },
                ),
            ]),
        };

        assert_eq!(
            infer_node_output_shapes("layer_00", &node, &signals, &params, Some(&tensor_index),)
                .unwrap(),
            vec![Some(vec![2560])]
        );
    }

    #[test]
    fn plans_fixture_model_lowered_execution_graph_activation_schedule() {
        let graph = ResolvedLoweredExecutionGraph::from_index_file(fixture_model_index_path()).unwrap();

        let plan = StreamCircuitExecutionPlan::from_graph(&graph).unwrap();

        assert_eq!(plan.topology, "explicit_graph");
        assert_eq!(plan.circuits.len(), 14);
        assert_eq!(plan.total_node_count(), 242);
        assert_eq!(plan.produced_signal_count(), 264);
        assert_eq!(plan.temporary_signal_count(), 230);
        assert_eq!(plan.state_view_signal_count(), 20);
        assert_eq!(plan.layer_local_activation_slot_count(), 56);
        assert_eq!(plan.operator_counts().get("linear"), Some(&82));
        assert_eq!(
            plan.state_type_counts().get("append_only_attention_memory"),
            Some(&6)
        );

        let layer_00 = &plan.circuits[0];
        let layer_00_frame = layer_00.activation_frame_plan();
        assert_eq!(layer_00.component_id, "layer_00");
        assert_eq!(layer_00.nodes.len(), 16);
        assert_eq!(layer_00.temporary_signals.len(), 16);
        assert_eq!(
            layer_00.state_view_signals,
            vec!["temporal_window".to_string()]
        );
        assert_eq!(layer_00_frame.slot_count, 4);
        assert_eq!(layer_00.input_ports[0].id, "input_frame");
        assert_eq!(layer_00.output_ports[0].id, "output_frame");
        assert_eq!(
            layer_00
                .nodes
                .iter()
                .find(|node| node.id == "temporal_memory_update")
                .unwrap()
                .state_writes,
            vec!["temporal_memory".to_string()]
        );

        let layer_02 = &plan.circuits[2];
        assert_eq!(layer_02.temporary_signals.len(), 17);
        assert_eq!(
            layer_02.state_view_signals,
            vec!["k_memory".to_string(), "v_memory".to_string()]
        );
        assert_eq!(layer_02.activation_frame_plan().slot_count, 4);
        assert!(
            layer_02
                .nodes
                .iter()
                .any(|node| node.op == "append_state_update")
        );
        assert!(
            layer_02
                .nodes
                .iter()
                .any(|node| node.op == "scaled_dot_product_attention")
        );
    }

    #[test]
    fn tensor_index_enables_fixture_model_signal_shape_planning() {
        let graph = ResolvedLoweredExecutionGraph::from_index_file(fixture_model_index_path()).unwrap();
        let tensor_index = TensorIndex::from_json_file(fixture_model_tensor_index_path()).unwrap();

        let plan = StreamCircuitExecutionPlan::from_graph_with_tensor_index(&graph, &tensor_index)
            .unwrap();
        let resource_plan = StreamCircuitResourcePlan::from_graph_and_plan(&graph, &plan).unwrap();

        assert_eq!(tensor_index.schema, TENSOR_INDEX_SCHEMA);
        assert_eq!(resource_plan.temporary_signal_count, 230);
        assert_eq!(resource_plan.state_view_signal_count, 20);
        assert_eq!(resource_plan.unknown_temporary_shape_count, 0);
        assert_eq!(resource_plan.unknown_state_view_shape_count, 12);
        assert!(resource_plan.intermediate_activation_shapes_known());

        let layer_00 = &plan.circuits[0];
        assert_eq!(
            layer_00.signal("conv_projected").unwrap().shape,
            Some(vec![3072])
        );
        assert_eq!(layer_00.signal("gate_b").unwrap().shape, Some(vec![1024]));
        assert_eq!(
            layer_00.signal("temporal_window").unwrap().shape,
            Some(vec![3, 1024])
        );
        assert_eq!(
            layer_00.signal("ffn_hidden").unwrap().shape,
            Some(vec![2560])
        );

        let layer_02 = &plan.circuits[2];
        assert_eq!(
            layer_02.signal("q_projected").unwrap().shape,
            Some(vec![1024])
        );
        assert_eq!(
            layer_02.signal("k_projected").unwrap().shape,
            Some(vec![512])
        );
        assert_eq!(layer_02.signal("k_memory").unwrap().shape, None);
        assert_eq!(
            layer_02.signal("k_memory").unwrap().storage,
            SignalStorage::StateView
        );
        assert_eq!(layer_02.signal("v_memory").unwrap().shape, None);
        assert_eq!(
            layer_02.signal("v_memory").unwrap().storage,
            SignalStorage::StateView
        );
        assert_eq!(
            layer_02.signal("attention_out").unwrap().shape,
            Some(vec![1024])
        );

        let layer_00_bank = resource_plan
            .activation_banks
            .iter()
            .find(|bank| bank.component_id == "layer_00")
            .unwrap();
        assert_eq!(layer_00_bank.slot_count, 4);
        assert_eq!(
            layer_00_bank
                .slots
                .iter()
                .map(|slot| slot.max_elements)
                .collect::<Vec<_>>(),
            vec![Some(2560), Some(3072), Some(2560), Some(2560)]
        );

        let layer_02_bank = resource_plan
            .activation_banks
            .iter()
            .find(|bank| bank.component_id == "layer_02")
            .unwrap();
        assert_eq!(layer_02_bank.slot_count, 4);
        assert_eq!(
            layer_02_bank
                .slots
                .iter()
                .map(|slot| slot.max_elements)
                .collect::<Vec<_>>(),
            vec![Some(1024), Some(2560), Some(2560), Some(2560)]
        );
    }

    #[test]
    fn tensor_index_uses_logical_shape_without_changing_storage_shape() {
        let index: TensorIndex = serde_json::from_str(
            r#"{
              "schema": "nerve.tensor_index.v1",
              "tensors": {
                "projection.qweight": {
                  "dtype": "I32",
                  "shape": [64, 768],
                  "logical_shape": [768, 512],
                  "layout": "packed_int32"
                }
              }
            }"#,
        )
        .unwrap();

        assert_eq!(
            index.tensor_shape("projection.qweight"),
            Some([768, 512].as_slice())
        );
        assert_eq!(index.tensors["projection.qweight"].shape, vec![64, 768]);
        assert_eq!(
            index.tensors["projection.qweight"].layout.as_deref(),
            Some("packed_int32")
        );
    }

    #[test]
    fn resource_plan_names_fixture_model_mount_resources() {
        let graph = ResolvedLoweredExecutionGraph::from_index_file(fixture_model_index_path()).unwrap();
        let execution_plan = StreamCircuitExecutionPlan::from_graph(&graph).unwrap();

        let resource_plan =
            StreamCircuitResourcePlan::from_graph_and_plan(&graph, &execution_plan).unwrap();

        assert_eq!(resource_plan.circuit_count, 14);
        assert_eq!(resource_plan.node_count, 242);
        assert_eq!(resource_plan.parameter_ref_count, 130);
        assert_eq!(resource_plan.unique_parameter_tensor_count(), 130);
        assert_eq!(resource_plan.transducer_parameter_ref_count, 3);
        assert_eq!(resource_plan.unique_transducer_parameter_tensor_count(), 2);
        assert_eq!(resource_plan.stream_state_count(), 14);
        assert_eq!(resource_plan.temporary_signal_count, 230);
        assert_eq!(resource_plan.state_view_signal_count, 20);
        assert_eq!(resource_plan.layer_local_activation_slot_count, 56);
        assert_eq!(resource_plan.unknown_temporary_shape_count, 172);
        assert_eq!(resource_plan.unknown_state_view_shape_count, 12);
        assert!(!resource_plan.intermediate_activation_shapes_known());

        let conv_in = resource_plan
            .parameters
            .iter()
            .find(|parameter| parameter.tensor == "model.layers.0.conv.in_proj.weight")
            .unwrap();
        assert_eq!(conv_in.uses.len(), 1);
        assert_eq!(conv_in.uses[0].component_id, "layer_00");
        assert_eq!(conv_in.uses[0].param_id, "conv_in_projection");
        assert_eq!(
            conv_in.uses[0].role.as_deref(),
            Some("short_convolution_input_projection")
        );
        assert_eq!(conv_in.uses[0].storage, "source_tensor_refs");

        let embed_tokens = resource_plan
            .transducer_parameters
            .iter()
            .find(|parameter| parameter.tensor == "model.embed_tokens.weight")
            .unwrap();
        assert_eq!(embed_tokens.uses.len(), 2);
        assert_eq!(
            embed_tokens
                .uses
                .iter()
                .map(|parameter_use| parameter_use.component_id.as_str())
                .collect::<Vec<_>>(),
            vec![
                "input_transducer.token_embedding",
                "output_transducer.output_projection",
            ]
        );
        assert_eq!(embed_tokens.uses[0].param_id, "weight");
        assert_eq!(
            embed_tokens.uses[0].role.as_deref(),
            Some("embedding_lookup")
        );
        assert_eq!(embed_tokens.uses[0].layout, "transducer");
        assert_eq!(
            embed_tokens.uses[1].role.as_deref(),
            Some("linear_projection")
        );

        let embedding_norm = resource_plan
            .transducer_parameters
            .iter()
            .find(|parameter| parameter.tensor == "model.embedding_norm.weight")
            .unwrap();
        assert_eq!(embedding_norm.uses.len(), 1);
        assert_eq!(
            embedding_norm.uses[0].component_id,
            "output_transducer.output_norm"
        );
        assert_eq!(embedding_norm.uses[0].role.as_deref(), Some("rms_norm"));

        let rolling_states = resource_plan
            .state_allocations
            .iter()
            .filter(|state| state.state_type == "rolling_frame_memory")
            .count();
        let append_only_states = resource_plan
            .state_allocations
            .iter()
            .filter(|state| state.state_type == "append_only_attention_memory")
            .count();
        assert_eq!(rolling_states, 8);
        assert_eq!(append_only_states, 6);

        let layer_00_state = resource_plan
            .state_allocations
            .iter()
            .find(|state| state.component_id == "layer_00")
            .unwrap();
        assert_eq!(layer_00_state.state_id, "temporal_memory");
        assert_eq!(layer_00_state.shape, Some(vec![3, 1024]));
        assert_eq!(layer_00_state.elements_per_activation, None);
        assert_eq!(layer_00_state.layout.as_deref(), Some("time_hidden"));

        let layer_02_state = resource_plan
            .state_allocations
            .iter()
            .find(|state| state.component_id == "layer_02")
            .unwrap();
        assert_eq!(layer_02_state.state_id, "kv_memory");
        assert_eq!(layer_02_state.shape, None);
        assert_eq!(layer_02_state.elements_per_activation, Some(1024));
        assert_eq!(layer_02_state.layout.as_deref(), Some("append_only_kv"));

        let layer_00_bank = resource_plan
            .activation_banks
            .iter()
            .find(|bank| bank.component_id == "layer_00")
            .unwrap();
        assert_eq!(layer_00_bank.temporary_signal_count, 16);
        assert_eq!(layer_00_bank.slot_count, 4);
        assert_eq!(layer_00_bank.assignments.len(), 16);

        let layer_02_bank = resource_plan
            .activation_banks
            .iter()
            .find(|bank| bank.component_id == "layer_02")
            .unwrap();
        assert_eq!(layer_02_bank.temporary_signal_count, 17);
        assert_eq!(layer_02_bank.slot_count, 4);
    }

    #[test]
    fn resource_plan_rejects_mismatched_execution_plan() {
        let graph = ResolvedLoweredExecutionGraph::from_index_file(fixture_model_index_path()).unwrap();
        let mut execution_plan = StreamCircuitExecutionPlan::from_graph(&graph).unwrap();
        execution_plan.circuits.pop();

        let error =
            StreamCircuitResourcePlan::from_graph_and_plan(&graph, &execution_plan).unwrap_err();

        assert!(error.to_string().contains("graph circuit count 14"));
    }

    #[test]
    fn activation_plan_tracks_signal_producers_and_consumers() {
        let graph = ResolvedLoweredExecutionGraph::from_index_file(fixture_model_index_path()).unwrap();
        let plan = StreamCircuitExecutionPlan::from_graph(&graph).unwrap();
        let layer_00 = &plan.circuits[0];

        let input_frame = layer_00.signal("input_frame").unwrap();
        assert_eq!(input_frame.producer, SignalProducer::BoundaryInput);
        assert_eq!(
            input_frame.consumers,
            vec!["operator_norm".to_string(), "operator_residual".to_string()]
        );

        let norm_out = layer_00.signal("operator_norm_out").unwrap();
        assert_eq!(
            norm_out.producer,
            SignalProducer::Node {
                node_id: "operator_norm".to_string()
            }
        );
        assert_eq!(norm_out.consumers, vec!["conv_in_projection".to_string()]);

        let output_frame = layer_00.signal("output_frame").unwrap();
        assert!(output_frame.is_boundary_output);
        assert_eq!(
            output_frame.consumers,
            vec!["boundary.output:output_frame".to_string()]
        );

        let temporal_window = layer_00.signal("temporal_window").unwrap();
        assert_eq!(temporal_window.storage, SignalStorage::StateView);
        assert_eq!(
            temporal_window.consumers,
            vec!["depthwise_temporal_conv".to_string()]
        );
    }

    #[test]
    fn activation_frame_plan_reuses_temporary_signal_slots_by_liveness() {
        let node = |index: usize, id: &str, outputs: &[&str]| PlannedNode {
            index,
            id: id.to_string(),
            op: "test".to_string(),
            specialization: String::new(),
            inputs: Vec::new(),
            outputs: outputs.iter().map(|output| (*output).to_string()).collect(),
            params: Vec::new(),
            state_reads: Vec::new(),
            state_writes: Vec::new(),
        };
        let signal = |id: &str, producer: &str, consumers: &[&str]| PlannedSignal {
            id: id.to_string(),
            producer: SignalProducer::Node {
                node_id: producer.to_string(),
            },
            consumers: consumers
                .iter()
                .map(|consumer| (*consumer).to_string())
                .collect(),
            shape: Some(vec![8]),
            element_bytes: Some(if id == "d" { 4 } else { 2 }),
            storage: SignalStorage::Activation,
            is_boundary_output: false,
        };
        let plan = CircuitActivationPlan {
            component_id: "component".to_string(),
            circuit_id: "circuit".to_string(),
            input_ports: Vec::new(),
            output_ports: Vec::new(),
            state_ports: Vec::new(),
            parameter_refs: Vec::new(),
            nodes: vec![
                node(0, "node_0", &["a"]),
                node(1, "node_1", &["b"]),
                node(2, "node_2", &["c"]),
                node(3, "node_3", &["d"]),
            ],
            signals: BTreeMap::from([
                ("a".to_string(), signal("a", "node_0", &["node_2"])),
                ("b".to_string(), signal("b", "node_1", &["node_2"])),
                ("c".to_string(), signal("c", "node_2", &["node_3"])),
                ("d".to_string(), signal("d", "node_3", &[])),
            ]),
            temporary_signals: vec![
                "a".to_string(),
                "b".to_string(),
                "c".to_string(),
                "d".to_string(),
            ],
            state_view_signals: Vec::new(),
        };
        let frame = plan.activation_frame_plan();

        assert_eq!(frame.liveness.len(), 4);
        assert_eq!(frame.slot_count, 3);
        assert_eq!(frame.slot_for("a"), Some(0));
        assert_eq!(frame.slot_for("b"), Some(1));
        assert_eq!(frame.slot_for("c"), Some(2));
        assert_eq!(frame.slot_for("d"), Some(0));
        let slots = planned_activation_slots(&plan, &frame);
        assert_eq!(slots[0].signal_ids, vec!["a".to_string(), "d".to_string()]);
        assert_eq!(slots[0].max_elements, Some(8));
        assert_eq!(slots[0].max_bytes, Some(32));
        assert_eq!(slots[1].max_bytes, Some(16));
        assert_eq!(slots[2].max_bytes, Some(16));

        let a = frame.liveness_for("a").unwrap();
        assert_eq!(a.produced_by, "node_0");
        assert_eq!(a.produced_at, 0);
        assert_eq!(a.last_consumed_at, 2);
        assert_eq!(a.consumers, vec!["node_2".to_string()]);
    }

    #[test]
    fn activation_plan_rejects_unscheduled_signal_dependency() {
        let graph = ResolvedLoweredExecutionGraph::from_index_file(fixture_model_index_path()).unwrap();
        let mut circuit = graph.circuits[0].circuit.clone();
        circuit.nodes[0].inputs = vec!["not_available_yet".to_string()];

        let error = CircuitActivationPlan::from_circuit("layer_00", &circuit).unwrap_err();

        assert!(
            error
                .to_string()
                .contains("input \"not_available_yet\" is not available")
        );
    }
}
