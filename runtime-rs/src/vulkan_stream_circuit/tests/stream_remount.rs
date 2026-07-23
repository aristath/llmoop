#[test]
fn placed_stream_remount_clones_live_component_state_without_sharing_it() {
    let device = match VulkanComputeDevice::new() {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping placed live state clone test: {error}");
            return;
        }
    };
    let manifest = fixture_model_package_manifest();
    let manifest_path = fixture_model_package_manifest_path();
    let manifest_dir = manifest_path.parent().unwrap();
    let source_graph = manifest
        .circuit_graph
        .to_resolved_lowered_execution_graph(manifest_dir)
        .unwrap();
    let source_runtime_graph =
        StreamCircuitRuntimeGraph::from_source_series(&source_graph, "gpu0").unwrap();
    let source_model = manifest.clone().mount_runtime_graph(&source_runtime_graph).unwrap();
    let device = Rc::new(device);
    let devices = BTreeMap::from([("gpu0".to_string(), device)]);
    let source_package = Arc::new(
        VulkanResidentInProcessPlacedModelPackage::from_runtime_model_for_bound_devices(
            &devices,
            manifest_dir,
            source_model,
            Some(4),
            false,
        )
        .unwrap(),
    );
    let mut stream =
        VulkanResidentInProcessPlacedPromptStream::new(source_package, devices.clone(), 0).unwrap();
    let source_state = stream
        .processor
        .device("gpu0")
        .unwrap()
        .mounted
        .buffers
        .state_buffers
        .iter()
        .find(|state| state.component_id == "layer_05")
        .unwrap();
    let state_id = source_state.state_id.clone();
    source_state.buffer.write_bytes(&[0xa5; 16]).unwrap();

    let mut clone_runtime_graph = source_runtime_graph
        .duplicate_after_instance(&source_graph, "layer_05", "layer_05_repeat")
        .unwrap();
    clone_runtime_graph
        .instances
        .iter_mut()
        .find(|instance| instance.instance_id == "layer_05_repeat")
        .unwrap()
        .state_policy = StreamCircuitNodeInstanceStatePolicy::CloneFrom {
        instance_id: "layer_05".to_string(),
    };
    clone_runtime_graph.validate_against_graph(&source_graph).unwrap();
    let clone_model = manifest.mount_runtime_graph(&clone_runtime_graph).unwrap();
    let clone_package = Arc::new(
        VulkanResidentInProcessPlacedModelPackage::from_runtime_model_for_bound_devices(
            &devices,
            manifest_dir,
            clone_model,
            Some(4),
            false,
        )
        .unwrap(),
    );

    stream
        .remount_model_preserving_state(clone_package.clone(), 0)
        .unwrap();
    let mounted = &stream.processor.device("gpu0").unwrap().mounted;
    let inherited = mounted.buffers.state_buffer("layer_05", &state_id).unwrap();
    let cloned = mounted
        .buffers
        .state_buffer("layer_05_repeat", &state_id)
        .unwrap();
    assert_eq!(inherited.buffer.read_bytes(16).unwrap(), vec![0xa5; 16]);
    assert_eq!(cloned.buffer.read_bytes(16).unwrap(), vec![0xa5; 16]);
    assert!(!std::ptr::eq(&inherited.buffer, &cloned.buffer));

    cloned.buffer.write_bytes(&[0x5a; 16]).unwrap();
    stream
        .remount_model_preserving_state(clone_package, 0)
        .unwrap();
    let remounted = &stream.processor.device("gpu0").unwrap().mounted;
    assert_eq!(
        remounted
            .buffers
            .state_buffer("layer_05", &state_id)
            .unwrap()
            .buffer
            .read_bytes(16)
            .unwrap(),
        vec![0xa5; 16]
    );
    assert_eq!(
        remounted
            .buffers
            .state_buffer("layer_05_repeat", &state_id)
            .unwrap()
            .buffer
            .read_bytes(16)
            .unwrap(),
        vec![0x5a; 16]
    );
}

