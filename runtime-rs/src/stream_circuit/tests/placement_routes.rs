    fn placement_plan_keeps_every_stream_entity_as_a_deployable_pedal() {
        let resolved =
            ResolvedLoweredPedalboard::from_index_file(fixture_model_index_path()).unwrap();

        let placement = resolved.single_device_placement_plan("gpu0").unwrap();

        assert_eq!(placement.schema, STREAM_CIRCUIT_PLACEMENT_SCHEMA);
        assert_eq!(placement.wiring, "explicit_graph");
        assert_eq!(placement.pedals.len(), 17);
        assert_eq!(placement.cables.len(), 17);
        assert_eq!(placement.local_cable_count, 17);
        assert_eq!(placement.cross_device_cable_count, 0);
        assert_eq!(
            placement.pedal("layer_00").unwrap(),
            &PedalPlacement {
                pedal_index: 1,
                pedal_id: "layer_00".to_string(),
                circuit_id: "layer_00_shortconv_circuit_v1".to_string(),
                operator_type: "conv".to_string(),
                device_id: "gpu0".to_string(),
            }
        );

        let first_cable = &placement.cables[0];
        assert_eq!(first_cable.source_pedal_id, "input_transducer");
        assert_eq!(first_cable.destination_pedal_id, "layer_00");
        assert_eq!(first_cable.signal, "frame");
        assert_eq!(first_cable.shape, vec![1024]);
        assert_eq!(first_cable.source_port_id, "output_frame");
        assert_eq!(first_cable.destination_port_id, "input_frame");
        assert_eq!(first_cable.source_pedal_port.as_deref(), Some("frame"));
        assert_eq!(first_cable.destination_pedal_port.as_deref(), Some("input"));
        assert_eq!(
            first_cable.transport,
            CableTransport::LocalBuffer {
                device_id: "gpu0".to_string(),
            }
        );
    }

    #[test]
    fn bypassed_instance_is_retained_in_draft_and_removed_from_execution_graph() {
        let source =
            ResolvedLoweredPedalboard::from_index_file(fixture_model_index_path()).unwrap();
        let patch = StreamCircuitRuntimePatch::from_source_series(&source, "gpu0")
            .unwrap()
            .with_instance_enabled("layer_01", false)
            .unwrap();

        let effective = source.instantiate_runtime_patch(&patch).unwrap();
        let placement = effective.placement_plan(&patch.placement_spec()).unwrap();

        assert_eq!(patch.instances.len(), 17);
        assert!(!patch.instances[2].enabled);
        assert_eq!(effective.circuits.len(), 16);
        assert!(
            effective
                .circuits
                .iter()
                .all(|circuit| circuit.pedal.id != "layer_01")
        );
        let bypass = placement
            .cables
            .iter()
            .find(|cable| cable.source_pedal_id == "layer_00")
            .unwrap();
        assert_eq!(bypass.destination_pedal_id, "layer_02");
    }

    #[test]
    fn placement_plan_changes_cables_not_pedalboard_when_devices_differ() {
        let resolved =
            ResolvedLoweredPedalboard::from_index_file(fixture_model_index_path()).unwrap();
        let spec = StreamCircuitPlacementSpec::new("gpu0")
            .with_pedal_device("layer_01", "cpu0")
            .with_pedal_device("layer_02", "gpu1")
            .with_pedal_device("layer_03", "lan:worker-a");

        let placement = resolved.placement_plan(&spec).unwrap();

        assert_eq!(placement.pedals.len(), 17);
        assert_eq!(placement.cables.len(), 17);
        assert_eq!(placement.local_cable_count, 13);
        assert_eq!(placement.cross_device_cable_count, 4);
        assert_eq!(
            placement
                .pedal("layer_01")
                .map(|pedal| pedal.device_id.as_str()),
            Some("cpu0")
        );
        assert_eq!(
            placement
                .pedal("layer_02")
                .map(|pedal| pedal.device_id.as_str()),
            Some("gpu1")
        );
        assert_eq!(
            placement
                .pedal("layer_03")
                .map(|pedal| pedal.device_id.as_str()),
            Some("lan:worker-a")
        );
        assert_eq!(
            placement
                .pedal("layer_04")
                .map(|pedal| pedal.device_id.as_str()),
            Some("gpu0")
        );

        let cross = placement.cross_device_cables();
        assert_eq!(cross.len(), 4);
        assert_eq!(
            cross
                .iter()
                .map(|cable| (
                    cable.source_pedal_id.as_str(),
                    cable.source_device_id.as_str(),
                    cable.destination_pedal_id.as_str(),
                    cable.destination_device_id.as_str()
                ))
                .collect::<Vec<_>>(),
            vec![
                ("layer_00", "gpu0", "layer_01", "cpu0"),
                ("layer_01", "cpu0", "layer_02", "gpu1"),
                ("layer_02", "gpu1", "layer_03", "lan:worker-a"),
                ("layer_03", "lan:worker-a", "layer_04", "gpu0"),
            ]
        );
        assert_eq!(
            cross[2].transport,
            CableTransport::CrossDevice {
                from_device_id: "gpu1".to_string(),
                to_device_id: "lan:worker-a".to_string(),
            }
        );
    }

    #[test]
    fn runtime_cable_routes_classify_logical_and_physical_routes() {
        let resolved =
            ResolvedLoweredPedalboard::from_index_file(fixture_model_index_path()).unwrap();
        let spec = StreamCircuitPlacementSpec::new("gpu0")
            .with_pedal_device("layer_01", "gpu1")
            .with_pedal_device("layer_02", "gpu2");
        let placement = resolved.placement_plan(&spec).unwrap();

        let routes = placement.runtime_cable_routes(|device_id| {
            let (target, physical_device_index) = match device_id {
                "gpu0" | "gpu1" => (Some("vulkan:0".to_string()), Some(0)),
                "gpu2" => (Some("vulkan:1".to_string()), Some(1)),
                _ => (None, None),
            };
            RuntimeCableRouteTarget {
                target,
                physical_device_index,
                binding_source: "test".to_string(),
            }
        });

        assert_eq!(routes.schema, RUNTIME_CABLE_ROUTES_SCHEMA);
        assert_eq!(routes.cable_count, 17);
        assert_eq!(routes.logical_local_cable_count, 14);
        assert_eq!(routes.logical_cross_device_cable_count, 3);
        assert_eq!(routes.same_physical_target_cable_count, 1);
        assert_eq!(routes.cross_physical_target_cable_count, 2);
        assert_eq!(routes.unresolved_target_cable_count, 0);
        assert_eq!(
            routes.routes[1].route_kind,
            RuntimeCableRouteKind::SamePhysicalTarget
        );
        assert_eq!(
            routes.routes[2].route_kind,
            RuntimeCableRouteKind::CrossPhysicalTarget
        );
        assert_eq!(
            routes.routes[0].route_kind,
            RuntimeCableRouteKind::LogicalLocal
        );

        let payload = serde_json::to_value(&routes).unwrap();
        assert_eq!(payload["routes"][1]["route_kind"], "same_physical_target");
        assert_eq!(payload["routes"][2]["route_kind"], "cross_physical_target");
        assert_eq!(payload["routes"][0]["route_kind"], "logical_local");
    }

    #[test]
    fn runtime_device_bindings_capture_runtime_target_contract() {
        let logical_device_ids = vec![
            "gpu0".to_string(),
            "gpu1".to_string(),
            "vulkan:7".to_string(),
        ];
        let mut explicit_bindings = BTreeMap::new();
        explicit_bindings.insert("gpu1".to_string(), "vulkan:5".to_string());
        explicit_bindings.insert("remote0".to_string(), "lan:worker-a".to_string());

        let bindings = RuntimeDeviceBindings::from_vulkan_targets(
            &logical_device_ids,
            &explicit_bindings,
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

        assert_eq!(bindings.schema, RUNTIME_DEVICE_BINDINGS_SCHEMA);
        assert_eq!(bindings.process_vulkan_device_index, Some(0));
        assert_eq!(bindings.default_vulkan_device_index, Some(0));
        assert_eq!(bindings.requested_vulkan_device_indices, vec![0, 5, 7]);
        assert!(!bindings.can_mount_in_process);
        assert_eq!(bindings.mounting_model, "unsupported_targets");
        assert_eq!(bindings.unsupported_targets, vec!["remote0=lan:worker-a"]);
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
                ("gpu0", Some("vulkan:0"), "process_default"),
                ("gpu1", Some("vulkan:5"), "explicit"),
                ("remote0", Some("lan:worker-a"), "explicit"),
                ("vulkan:7", Some("vulkan:7"), "device_id"),
            ]
        );

        let payload = serde_json::to_value(&bindings).unwrap();
        assert_eq!(payload["schema"], RUNTIME_DEVICE_BINDINGS_SCHEMA);
        assert_eq!(payload["logical_devices"][0]["device_id"], "gpu0");
        assert_eq!(
            payload["logical_devices"][0]["binding_source"],
            "process_default"
        );
        assert_eq!(payload["unsupported_targets"][0], "remote0=lan:worker-a");
    }

