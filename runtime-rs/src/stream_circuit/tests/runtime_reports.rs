    fn runtime_device_bindings_treat_cpu_targets_as_direct_runtime_devices() {
        let logical_device_ids = vec!["cpu0".to_string(), "gpu0".to_string()];
        let bindings = RuntimeDeviceBindings::from_vulkan_targets(
            &logical_device_ids,
            &BTreeMap::new(),
            Some(0),
            |target| match target {
                "cpu0" => Ok(Some(6)),
                raw if raw.starts_with("vulkan:") => raw
                    .strip_prefix("vulkan:")
                    .unwrap()
                    .parse::<usize>()
                    .map(Some)
                    .map_err(|error| {
                        format!("invalid Vulkan physical device reference {target:?}: {error}")
                    }),
                _ => Ok(None),
            },
        );

        assert_eq!(bindings.requested_vulkan_device_indices, vec![0, 6]);
        assert!(bindings.can_mount_in_process);
        assert_eq!(
            bindings
                .logical_devices
                .iter()
                .map(|device| (
                    device.device_id.as_str(),
                    device.target.as_deref(),
                    device.binding_source.as_str()
                ))
                .collect::<Vec<_>>(),
            vec![
                ("cpu0", Some("vulkan:6"), "device_id"),
                ("gpu0", Some("vulkan:0"), "process_default"),
            ]
        );
    }

    #[test]
    fn runtime_bound_device_serializes_logical_target_report() {
        let bound = RuntimeBoundDevice {
            device_id: "gpu1".to_string(),
            target: Some("vulkan:5".to_string()),
            physical_device_index: Some(5),
            device_name: "Radeon Test Device".to_string(),
        };

        let payload = serde_json::to_value(&bound).unwrap();

        assert_eq!(payload["device_id"], "gpu1");
        assert_eq!(payload["target"], "vulkan:5");
        assert_eq!(payload["physical_device_index"], 5);
        assert_eq!(payload["device_name"], "Radeon Test Device");
    }

    #[test]
    fn runtime_available_device_serializes_inventory_entries() {
        let available = RuntimeAvailableDevice {
            device_id: "vulkan:5".to_string(),
            backend: "vulkan_compute".to_string(),
            available: true,
            runtime_device_id: None,
            physical_device_id: Some("vulkan:5".to_string()),
            physical_device_index: Some(5),
            device_name: Some("Radeon Test Device".to_string()),
            device_type: Some("discrete_gpu".to_string()),
            vendor_id: Some(4098),
            raw_device_id: Some(29_567),
            api_version: Some(4_203_000),
            driver_version: Some(1_024),
            compute_queue_family_indices: Some(vec![0, 2]),
            memory_heaps: Some(vec![RuntimeAvailableMemoryHeap {
                heap_index: 0,
                size_bytes: 8 * 1024 * 1024 * 1024,
                device_local: true,
            }]),
            selected_by_default: Some(false),
            selected_by_runtime: Some(false),
            runtime_binding: Some("inventory_only".to_string()),
            can_host_runtime_pedals_on_physical_device: Some(true),
            notes: vec![
                "auto-detected by Vulkan inventory; can be selected with --bind-device LOGICAL=vulkan:N"
                    .to_string(),
            ],
            error: None,
        };
        let unavailable = RuntimeAvailableDevice {
            device_id: "runtime_default".to_string(),
            backend: "vulkan_compute".to_string(),
            available: false,
            runtime_device_id: None,
            physical_device_id: None,
            physical_device_index: None,
            device_name: None,
            device_type: None,
            vendor_id: None,
            raw_device_id: None,
            api_version: None,
            driver_version: None,
            compute_queue_family_indices: None,
            memory_heaps: None,
            selected_by_default: None,
            selected_by_runtime: None,
            runtime_binding: None,
            can_host_runtime_pedals_on_physical_device: None,
            notes: vec!["no compute-capable Vulkan physical devices were found".to_string()],
            error: None,
        };

        let available_payload = serde_json::to_value(&available).unwrap();
        assert_eq!(available_payload["device_id"], "vulkan:5");
        assert_eq!(available_payload["physical_device_index"], 5);
        assert_eq!(available_payload["memory_heaps"][0]["device_local"], true);
        assert_eq!(
            available_payload["can_host_runtime_pedals_on_physical_device"],
            true
        );
        assert!(available_payload.get("runtime_device_id").is_none());

        let unavailable_payload = serde_json::to_value(&unavailable).unwrap();
        assert_eq!(unavailable_payload["device_id"], "runtime_default");
        assert_eq!(unavailable_payload["available"], false);
        assert!(unavailable_payload.get("physical_device_id").is_none());
        assert!(unavailable_payload.get("error").is_none());
    }

    #[test]
    fn runtime_source_pedal_serializes_compiled_pedal_summary() {
        let pedal = RuntimeSourcePedal {
            pedal_index: 5,
            pedal_id: "layer_05".to_string(),
            operator_type: "attention".to_string(),
            runtime_role: CircuitRuntimeRole::SignalProcessor,
            implementation: "vulkan_resident".to_string(),
            behavioral_role: "transformer_layer".to_string(),
            source_layer_index: Some(5),
            circuit_id: "layer_05_circuit_v1".to_string(),
            input_ports: vec![RuntimePedalPortSummary {
                id: "input_frame".to_string(),
                signal: "frame".to_string(),
                shape: vec![1024],
                source: Some("hidden_states".to_string()),
                pedal_port: Some("input".to_string()),
            }],
            output_ports: vec![RuntimePedalPortSummary {
                id: "output_frame".to_string(),
                signal: "frame".to_string(),
                shape: vec![1024],
                source: Some("hidden_states".to_string()),
                pedal_port: Some("output".to_string()),
            }],
            state_port_count: 1,
            parameter_ref_count: 12,
            node_count: 7,
            kernel_count: 7,
        };

        let payload = serde_json::to_value(&pedal).unwrap();

        assert_eq!(payload["pedal_index"], 5);
        assert_eq!(payload["pedal_id"], "layer_05");
        assert_eq!(payload["operator_type"], "attention");
        assert_eq!(payload["input_ports"][0]["id"], "input_frame");
        assert_eq!(payload["input_ports"][0]["pedal_port"], "input");
        assert_eq!(payload["output_ports"][0]["pedal_port"], "output");
        assert_eq!(payload["kernel_count"], 7);
    }

    #[test]
    fn runtime_topology_report_serializes_ui_facing_contract() {
        let logical_device_ids = vec!["gpu0".to_string()];
        let bindings = RuntimeDeviceBindings::from_vulkan_targets(
            &logical_device_ids,
            &BTreeMap::new(),
            Some(0),
            |target| {
                if let Some(index) = target.strip_prefix("vulkan:") {
                    return index.parse::<usize>().map(Some).map_err(|error| {
                        format!("invalid Vulkan physical device reference {target:?}: {error}")
                    });
                }
                Ok(None)
            },
        );
        let source_pedal = RuntimeSourcePedal {
            pedal_index: 0,
            pedal_id: "layer_00".to_string(),
            operator_type: "layer".to_string(),
            runtime_role: CircuitRuntimeRole::SignalProcessor,
            implementation: "vulkan_resident".to_string(),
            behavioral_role: "transformer_layer".to_string(),
            source_layer_index: Some(0),
            circuit_id: "layer_00_circuit_v1".to_string(),
            input_ports: Vec::new(),
            output_ports: Vec::new(),
            state_port_count: 0,
            parameter_ref_count: 0,
            node_count: 0,
            kernel_count: 0,
        };
        let report = RuntimeTopologyReport {
            ok: true,
            schema: RUNTIME_TOPOLOGY_SCHEMA.to_string(),
            package_manifest: PathBuf::from("package.json"),
            package_root: PathBuf::from("."),
            package_id: "model-test".to_string(),
            compiled_schema: "nerve.vulkan_resident_model_package.v3".to_string(),
            config_path: "config.json".to_string(),
            tokenizer: serde_json::json!({"path": "tokenizer"}),
            available_devices: vec![RuntimeAvailableDevice {
                device_id: "gpu0".to_string(),
                backend: "vulkan_compute".to_string(),
                available: true,
                runtime_device_id: Some("gpu0".to_string()),
                physical_device_id: Some("vulkan:0".to_string()),
                physical_device_index: Some(0),
                device_name: Some("Radeon Test Device".to_string()),
                device_type: Some("discrete_gpu".to_string()),
                vendor_id: Some(4098),
                raw_device_id: Some(29_567),
                api_version: Some(4_203_000),
                driver_version: Some(1_024),
                compute_queue_family_indices: Some(vec![0]),
                memory_heaps: Some(Vec::new()),
                selected_by_default: Some(true),
                selected_by_runtime: Some(true),
                runtime_binding: Some("default_local_vulkan_target".to_string()),
                can_host_runtime_pedals_on_physical_device: Some(true),
                notes: Vec::new(),
                error: None,
            }],
            compiled: RuntimeCompiledPedalboardSummary {
                wiring: "series".to_string(),
                source_pedal_count: 1,
                source_pedals: vec![source_pedal],
                max_context_activations: 16,
            },
            runtime_patch_controls: RuntimePatchControls {
                default_device_id: Some("gpu0".to_string()),
                pedal_devices: BTreeMap::new(),
                source_chain: None,
                duplicate_after: vec![RuntimePatchDuplicateAfterControl {
                    after_instance_id: "layer_00".to_string(),
                    new_instance_id: "layer_00_repeat".to_string(),
                }],
            },
            runtime_patch: StreamCircuitRuntimePatch {
                schema: STREAM_CIRCUIT_RUNTIME_PATCH_SCHEMA.to_string(),
                wiring: "explicit_graph".to_string(),
                default_device_id: "gpu0".to_string(),
                instances: vec![StreamCircuitPedalInstance {
                    instance_id: "layer_00".to_string(),
                    source_pedal_id: "layer_00".to_string(),
                    device_id: "gpu0".to_string(),
                    enabled: true,
                    control_values: BTreeMap::new(),
                    state_policy: StreamCircuitPedalInstanceStatePolicy::Fresh,
                }],
                cables: Vec::new(),
                boundary: StreamCircuitGraphBoundary {
                    external_inputs: vec![StreamCircuitGraphBoundaryPort {
                        id: "model_input".to_string(),
                        endpoint: StreamCircuitCableEndpoint {
                            pedal_id: "layer_00".to_string(),
                            port_id: "input_frame".to_string(),
                        },
                    }],
                    public_outputs: vec![StreamCircuitGraphBoundaryPort {
                        id: "model_output".to_string(),
                        endpoint: StreamCircuitCableEndpoint {
                            pedal_id: "layer_00".to_string(),
                            port_id: "output_frame".to_string(),
                        },
                    }],
                },
            },
            effective: RuntimeEffectivePedalboardTopology {
                wiring: "series".to_string(),
                pedal_count: 1,
                cable_count: 0,
                local_cable_count: 0,
                cross_device_cable_count: 0,
                device_count: 1,
                device_ids: vec!["gpu0".to_string()],
                device_bindings: bindings,
                cable_routes: RuntimeCableRoutes {
                    schema: RUNTIME_CABLE_ROUTES_SCHEMA.to_string(),
                    cable_count: 0,
                    logical_local_cable_count: 0,
                    logical_cross_device_cable_count: 0,
                    same_physical_target_cable_count: 0,
                    cross_physical_target_cable_count: 0,
                    unresolved_target_cable_count: 0,
                    routes: Vec::new(),
                },
                pedals: vec![PedalPlacement {
                    pedal_index: 0,
                    pedal_id: "layer_00".to_string(),
                    circuit_id: "layer_00_circuit_v1".to_string(),
                    operator_type: "layer".to_string(),
                    device_id: "gpu0".to_string(),
                }],
                cables: Vec::new(),
            },
        };

        let payload = serde_json::to_value(&report).unwrap();

        assert_eq!(payload["schema"], RUNTIME_TOPOLOGY_SCHEMA);
        assert!(payload["compiled"].get("default_device_id").is_none());
        assert_eq!(
            payload["available_devices"][0]["physical_device_id"],
            "vulkan:0"
        );
        assert_eq!(
            payload["runtime_patch_controls"]["duplicate_after"][0]["new_instance_id"],
            "layer_00_repeat"
        );
        assert_eq!(
            payload["effective"]["device_bindings"]["can_mount_in_process"],
            true
        );
        assert_eq!(payload["effective"]["pedals"][0]["pedal_id"], "layer_00");
    }

    #[test]
    fn runtime_package_inspection_report_serializes_box_of_parts_contract() {
        let report = RuntimePackageInspectionReport {
            ok: true,
            package_manifest: PathBuf::from("package.json"),
            package_root: PathBuf::from("."),
            schema: "nerve.vulkan_resident_model_package.v3".to_string(),
            package_id: "model-test".to_string(),
            config_path: "config.json".to_string(),
            tokenizer: serde_json::json!({"path": "tokenizer"}),
            compiled_wiring: "series".to_string(),
            runtime_patch: RuntimePatchControls {
                default_device_id: None,
                pedal_devices: BTreeMap::new(),
                source_chain: None,
                duplicate_after: Vec::new(),
            },
            device_bindings: RuntimeDeviceBindings::from_vulkan_targets(
                &Vec::<String>::new(),
                &BTreeMap::new(),
                Some(0),
                |target| {
                    if let Some(index) = target.strip_prefix("vulkan:") {
                        return index.parse::<usize>().map(Some).map_err(|error| {
                            format!("invalid Vulkan physical device reference {target:?}: {error}")
                        });
                    }
                    Ok(None)
                },
            ),
            max_context_activations: 16,
            source_pedal_count: 0,
            source_pedals: Vec::new(),
            available_devices: Vec::new(),
        };

        let payload = serde_json::to_value(&report).unwrap();

        assert_eq!(payload["package_id"], "model-test");
        assert!(payload.get("compiled_default_device_id").is_none());
        assert_eq!(
            payload["runtime_patch"]["default_device_id"],
            serde_json::Value::Null
        );
        assert_eq!(payload["source_pedal_count"], 0);
    }

    #[test]
    fn runtime_patch_inspection_report_serializes_patch_preview_contract() {
        let report = RuntimePatchInspectionReport {
            ok: true,
            package_manifest: PathBuf::from("package.json"),
            package_root: PathBuf::from("."),
            package_id: "model-test".to_string(),
            compiled_source_pedal_count: 14,
            runtime_patch_controls: RuntimePatchControls {
                default_device_id: Some("gpu0".to_string()),
                pedal_devices: BTreeMap::new(),
                source_chain: Some(vec![RuntimePatchSourceChainEntry {
                    instance_id: "layer_05_repeat".to_string(),
                    source_pedal_id: "layer_05".to_string(),
                }]),
                duplicate_after: Vec::new(),
            },
            runtime_patch: StreamCircuitRuntimePatch {
                schema: STREAM_CIRCUIT_RUNTIME_PATCH_SCHEMA.to_string(),
                wiring: "explicit_graph".to_string(),
                default_device_id: "gpu0".to_string(),
                instances: vec![StreamCircuitPedalInstance {
                    instance_id: "layer_05_repeat".to_string(),
                    source_pedal_id: "layer_05".to_string(),
                    device_id: "vulkan:5".to_string(),
                    enabled: true,
                    control_values: BTreeMap::new(),
                    state_policy: StreamCircuitPedalInstanceStatePolicy::Fresh,
                }],
                cables: Vec::new(),
                boundary: StreamCircuitGraphBoundary {
                    external_inputs: vec![StreamCircuitGraphBoundaryPort {
                        id: "model_input".to_string(),
                        endpoint: StreamCircuitCableEndpoint {
                            pedal_id: "layer_05_repeat".to_string(),
                            port_id: "input_frame".to_string(),
                        },
                    }],
                    public_outputs: vec![StreamCircuitGraphBoundaryPort {
                        id: "model_output".to_string(),
                        endpoint: StreamCircuitCableEndpoint {
                            pedal_id: "layer_05_repeat".to_string(),
                            port_id: "output_frame".to_string(),
                        },
                    }],
                },
            },
            device_bindings: RuntimeDeviceBindings::from_vulkan_targets(
                &["vulkan:5".to_string()],
                &BTreeMap::new(),
                Some(0),
                |target| {
                    if let Some(index) = target.strip_prefix("vulkan:") {
                        return index.parse::<usize>().map(Some).map_err(|error| {
                            format!("invalid Vulkan physical device reference {target:?}: {error}")
                        });
                    }
                    Ok(None)
                },
            ),
            effective_pedal_count: 1,
            effective_cable_count: 0,
            placement: RuntimePatchPlacementReport {
                schema: STREAM_CIRCUIT_PLACEMENT_SCHEMA.to_string(),
                wiring: "series".to_string(),
                local_cable_count: 0,
                cross_device_cable_count: 0,
                runtime_routes: RuntimeCableRoutes {
                    schema: RUNTIME_CABLE_ROUTES_SCHEMA.to_string(),
                    cable_count: 0,
                    logical_local_cable_count: 0,
                    logical_cross_device_cable_count: 0,
                    same_physical_target_cable_count: 0,
                    cross_physical_target_cable_count: 0,
                    unresolved_target_cable_count: 0,
                    routes: Vec::new(),
                },
                pedals: vec![PedalPlacement {
                    pedal_index: 0,
                    pedal_id: "layer_05_repeat".to_string(),
                    circuit_id: "layer_05_circuit_v1".to_string(),
                    operator_type: "layer".to_string(),
                    device_id: "vulkan:5".to_string(),
                }],
                cables: Vec::new(),
            },
        };

        let payload = serde_json::to_value(&report).unwrap();

        assert_eq!(payload["compiled_source_pedal_count"], 14);
        assert_eq!(
            payload["runtime_patch"]["instances"][0]["device_id"],
            "vulkan:5"
        );
        assert_eq!(
            payload["runtime_patch_controls"]["source_chain"][0]["instance_id"],
            "layer_05_repeat"
        );
        assert_eq!(payload["placement"]["pedals"][0]["device_id"], "vulkan:5");
    }

    #[test]
    fn runtime_device_slice_report_serializes_mounted_device_contract() {
        let report = RuntimeDeviceSliceReport {
            ok: true,
            package_manifest: PathBuf::from("package.json"),
            device_name: "Radeon Test Device".to_string(),
            device_id: "gpu1".to_string(),
            context_window_activations: 16,
            hosted_pedals: vec!["layer_05".to_string(), "layer_06".to_string()],
            local_cables: vec![RuntimeLocalCableBufferReport {
                cable_index: 5,
                signal: "hidden_state".to_string(),
                source_pedal_id: "layer_05".to_string(),
                destination_pedal_id: "layer_06".to_string(),
                device_id: "gpu1".to_string(),
                byte_capacity: Some(4096),
            }],
            incoming_cables: vec![RuntimeRemoteCableBufferReport {
                cable_index: 4,
                signal: "hidden_state".to_string(),
                source_device_id: "gpu0".to_string(),
                source_pedal_id: "layer_04".to_string(),
                destination_device_id: "gpu1".to_string(),
                destination_pedal_id: "layer_05".to_string(),
                byte_capacity: Some(4096),
            }],
            outgoing_cables: vec![RuntimeRemoteCableBufferReport {
                cable_index: 6,
                signal: "hidden_state".to_string(),
                source_device_id: "gpu1".to_string(),
                source_pedal_id: "layer_06".to_string(),
                destination_device_id: "gpu2".to_string(),
                destination_pedal_id: "layer_07".to_string(),
                byte_capacity: Some(4096),
            }],
            hosted_pedal_count: 2,
            incoming_cable_count: 1,
            outgoing_cable_count: 1,
            permanent_parameter_count: 12,
            permanent_parameter_bytes: 2048,
            reusable_kernel_word_count: 128,
            loaded_kernel_artifact_count: 4,
            dispatch_count: 8,
            descriptor_count: 24,
            model_boundary_descriptor_count: 2,
            incoming_cable_descriptor_count: 1,
            outgoing_cable_descriptor_count: 1,
            tick_plan: RuntimeDeviceTickPlanReport {
                stage_count: 4,
                receive_stage_count: 1,
                dispatch_stage_count: 2,
                publish_stage_count: 1,
                local_cable_read_count: 1,
                local_cable_write_count: 1,
                incoming_cable_read_count: 1,
                outgoing_cable_write_count: 1,
                model_input_read_count: 0,
                model_output_write_count: 0,
                can_execute: true,
            },
        };

        let payload = serde_json::to_value(&report).unwrap();

        assert_eq!(payload["device_id"], "gpu1");
        assert_eq!(payload["hosted_pedals"][0], "layer_05");
        assert_eq!(payload["local_cables"][0]["byte_capacity"], 4096);
        assert_eq!(payload["incoming_cables"][0]["source_device_id"], "gpu0");
        assert_eq!(
            payload["outgoing_cables"][0]["destination_device_id"],
            "gpu2"
        );
        assert_eq!(payload["tick_plan"]["can_execute"], true);
    }

    #[test]
    fn runtime_placement_report_serializes_device_slice_collection() {
        let report = RuntimePlacementReport {
            ok: true,
            package_manifest: PathBuf::from("package.json"),
            context_window_activations: 16,
            runtime_patch: RuntimePatchControls {
                default_device_id: Some("gpu0".to_string()),
                pedal_devices: BTreeMap::new(),
                source_chain: None,
                duplicate_after: Vec::new(),
            },
            device_bindings: RuntimeDeviceBindings::from_vulkan_targets(
                &["gpu0".to_string()],
                &BTreeMap::new(),
                Some(0),
                |target| {
                    if let Some(index) = target.strip_prefix("vulkan:") {
                        return index.parse::<usize>().map(Some).map_err(|error| {
                            format!("invalid Vulkan physical device reference {target:?}: {error}")
                        });
                    }
                    Ok(None)
                },
            ),
            bound_devices: vec![RuntimeBoundDevice {
                device_id: "gpu0".to_string(),
                target: Some("vulkan:0".to_string()),
                physical_device_index: Some(0),
                device_name: "Radeon Test Device".to_string(),
            }],
            cable_routes: RuntimeCableRoutes {
                schema: RUNTIME_CABLE_ROUTES_SCHEMA.to_string(),
                cable_count: 0,
                logical_local_cable_count: 0,
                logical_cross_device_cable_count: 0,
                same_physical_target_cable_count: 0,
                cross_physical_target_cable_count: 0,
                unresolved_target_cable_count: 0,
                routes: Vec::new(),
            },
            device_count: 1,
            device_ids: vec!["gpu0".to_string()],
            devices: Vec::new(),
        };

        let payload = serde_json::to_value(&report).unwrap();

        assert_eq!(payload["device_count"], 1);
        assert_eq!(payload["device_ids"][0], "gpu0");
        assert_eq!(payload["bound_devices"][0]["target"], "vulkan:0");
        assert_eq!(
            payload["device_bindings"]["logical_devices"][0]["binding_source"],
            "process_default"
        );
    }

    #[test]
    fn runtime_prompt_run_reports_serialize_execution_contracts() {
        let bindings = RuntimeDeviceBindings::from_vulkan_targets(
            &["gpu0".to_string()],
            &BTreeMap::new(),
            Some(0),
            |target| {
                if let Some(index) = target.strip_prefix("vulkan:") {
                    return index.parse::<usize>().map(Some).map_err(|error| {
                        format!("invalid Vulkan physical device reference {target:?}: {error}")
                    });
                }
                Ok(None)
            },
        );
        let tokenizer = RuntimeTokenizerOptionsReport {
            add_special_tokens: true,
            skip_special_tokens: true,
        };
        let timing = RuntimePromptTimingReport {
            setup_time_ns: 10,
            run_time_ns: 90,
            total_time_ns: 100,
            generated_token_count: 1,
            tick_count: 1,
            scheduler_turn_count: 1,
            average_generated_token_time_ns: Some(90),
            average_tick_time_ns: Some(90),
            average_scheduler_turn_time_ns: Some(90),
        };
        let placed = RuntimePlacedPromptRunReport {
            ok: true,
            execution_mode: "placed_in_process".to_string(),
            package_manifest: PathBuf::from("package.json"),
            tokenizer_dir: PathBuf::from("tokenizer"),
            input_device_id: "gpu0".to_string(),
            output_device_id: "gpu1".to_string(),
            device_count: 1,
            device_ids: vec!["gpu0".to_string()],
            bound_devices: Vec::new(),
            cable_routes: RuntimeCableRoutes {
                schema: RUNTIME_CABLE_ROUTES_SCHEMA.to_string(),
                cable_count: 0,
                logical_local_cable_count: 0,
                logical_cross_device_cable_count: 0,
                same_physical_target_cable_count: 0,
                cross_physical_target_cable_count: 0,
                unresolved_target_cable_count: 0,
                routes: Vec::new(),
            },
            runtime_patch: RuntimePatchControls {
                default_device_id: Some("gpu0".to_string()),
                pedal_devices: BTreeMap::new(),
                source_chain: None,
                duplicate_after: Vec::new(),
            },
            device_bindings: bindings,
            hosted_pedal_count: 14,
            context_window_activations: 16,
            scheduled_token_activations: 2,
            tokenizer,
            prompt_text: "Hello".to_string(),
            prompt_ids: vec![1],
            generated_ids: vec![2],
            generated_text: " world".to_string(),
            output_text: "Hello world".to_string(),
            stop_reason: "max_new_tokens".to_string(),
            tick_count: 1,
            scheduler_turns: 1,
            completed_stage_deltas: vec![42],
            transport: RuntimePlacedTransportReport {
                published_packet_count: 0,
                published_byte_count: 0,
                received_packet_count: 0,
                received_byte_count: 0,
                direct_copy_count: 2,
                direct_copy_byte_count: 4096,
                direct_receive_count: 2,
                direct_receive_byte_count: 4096,
                by_tick: vec![RuntimePlacedTransportStatsReport {
                    pending_packet_count: 0,
                    pending_byte_count: 0,
                    pending_direct_cable_count: 0,
                    pending_direct_byte_count: 0,
                    published_packet_count: 0,
                    published_byte_count: 0,
                    received_packet_count: 0,
                    received_byte_count: 0,
                    direct_copy_count: 2,
                    direct_copy_byte_count: 4096,
                    direct_receive_count: 2,
                    direct_receive_byte_count: 4096,
                }],
            },
            timing,
            pedal_timings: vec![RuntimePlacedPedalTimingReport {
                stream_tick: 0,
                device_id: "gpu0".to_string(),
                pedal_id: "layer_00".to_string(),
                dispatch_count: 1,
                run_time_ns: 90,
                average_dispatch_time_ns: Some(90),
                dispatches: vec![RuntimePlacedPedalDispatchTimingReport {
                    dispatch_index: 0,
                    kernel_id: "matmul".to_string(),
                    node_id: "layer_00.matmul".to_string(),
                    op: "linear".to_string(),
                    reusable_family_id: "linear".to_string(),
                    run_time_ns: 90,
                }],
            }],
            pedal_timing_summaries: vec![RuntimePlacedPedalTimingSummaryReport {
                device_id: "gpu0".to_string(),
                pedal_id: "layer_00".to_string(),
                tick_count: 1,
                dispatch_count: 1,
                total_run_time_ns: 90,
                average_tick_time_ns: Some(90),
                average_dispatch_time_ns: Some(90),
            }],
            speculative_cycle_count: 0,
            proposed_draft_token_count: 0,
            accepted_draft_token_count: 0,
            speculative_emitted_token_count: 0,
            speculative_draft_time_ns: 0,
            speculative_target_verification_time_ns: 0,
            speculative_draft_catch_up_time_ns: 0,
        };
        let benchmark_transport = RuntimePromptBenchmarkTransportTotalsReport {
            published_packet_count: 0,
            published_byte_count: 0,
            received_packet_count: 0,
            received_byte_count: 0,
            direct_copy_count: 2,
            direct_copy_byte_count: 4096,
            direct_receive_count: 2,
            direct_receive_byte_count: 4096,
        };
        let benchmark = RuntimePromptBenchmarkReport {
            ok: true,
            execution_mode: "placed_in_process".to_string(),
            package_manifest: PathBuf::from("package.json"),
            tokenizer_dir: PathBuf::from("tokenizer"),
            runtime_patch: placed.runtime_patch.clone(),
            device_bindings: placed.device_bindings.clone(),
            device_count: 1,
            device_ids: vec!["gpu0".to_string()],
            profile_runs: 1,
            prompt_text: "Hello".to_string(),
            prompt_ids: vec![1],
            max_new_tokens: 1,
            setup_time_ns: RuntimePromptBenchmarkU64MetricReport {
                total: 10,
                min: 10,
                max: 10,
                average: 10.0,
            },
            run_time_ns: RuntimePromptBenchmarkU64MetricReport {
                total: 90,
                min: 90,
                max: 90,
                average: 90.0,
            },
            total_time_ns: RuntimePromptBenchmarkU64MetricReport {
                total: 100,
                min: 100,
                max: 100,
                average: 100.0,
            },
            generated_token_count: RuntimePromptBenchmarkUsizeMetricReport {
                total: 1,
                min: 1,
                max: 1,
                average: 1.0,
            },
            tick_count: RuntimePromptBenchmarkUsizeMetricReport {
                total: 1,
                min: 1,
                max: 1,
                average: 1.0,
            },
            scheduler_turn_count: RuntimePromptBenchmarkUsizeMetricReport {
                total: 1,
                min: 1,
                max: 1,
                average: 1.0,
            },
            generated_tokens_per_second: Some(11_111_111.111),
            stop_reasons: BTreeMap::from([("max_new_tokens".to_string(), 1)]),
            transport_totals: Some(benchmark_transport.clone()),
            pedal_timing_summaries: placed.pedal_timing_summaries.clone(),
            runs: vec![RuntimePromptBenchmarkRunReport {
                run_index: 0,
                execution_mode: "placed_in_process".to_string(),
                stop_reason: "max_new_tokens".to_string(),
                generated_token_count: 1,
                tick_count: 1,
                scheduler_turn_count: 1,
                setup_time_ns: 10,
                run_time_ns: 90,
                total_time_ns: 100,
                generated_tokens_per_second: Some(11_111_111.111),
                transport: Some(benchmark_transport),
                pedal_timing_summaries: placed.pedal_timing_summaries.clone(),
            }],
        };

        let placed_payload = serde_json::to_value(&placed).unwrap();
        let benchmark_payload = serde_json::to_value(&benchmark).unwrap();

        assert_eq!(placed_payload["execution_mode"], "placed_in_process");
        assert_eq!(placed_payload["transport"]["direct_copy_count"], 2);
        assert_eq!(placed_payload["completed_stage_deltas"][0], 42);
        assert_eq!(
            placed_payload["timing"]["average_generated_token_time_ns"],
            90
        );
        assert_eq!(placed_payload["pedal_timings"][0]["pedal_id"], "layer_00");
        assert_eq!(placed_payload["pedal_timings"][0]["run_time_ns"], 90);
        assert_eq!(
            placed_payload["pedal_timings"][0]["dispatches"][0]["node_id"],
            "layer_00.matmul"
        );
        assert_eq!(
            placed_payload["pedal_timing_summaries"][0]["total_run_time_ns"],
            90
        );
        assert_eq!(benchmark_payload["profile_runs"], 1);
        assert_eq!(benchmark_payload["run_time_ns"]["average"], 90.0);
        assert_eq!(
            benchmark_payload["transport_totals"]["direct_copy_byte_count"],
            4096
        );
    }
