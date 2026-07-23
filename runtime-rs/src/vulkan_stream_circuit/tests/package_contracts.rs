#[test]
fn package_loader_rejects_shallow_and_stale_behavioral_evidence() {
    let source_manifest_path = fixture_model_package_manifest_path();
    let source_manifest = fixture_model_package_manifest();
    let source_root = source_manifest_path.parent().unwrap();
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = std::env::temp_dir().join(format!(
        "nerve-behavioral-evidence-{}-{unique}",
        std::process::id()
    ));
    std::fs::create_dir_all(&root).unwrap();
    let manifest_path = root.join("vulkan_resident_package.json");
    let evidence_path = root.join("behavioral_validation.json");
    std::fs::copy(&source_manifest_path, &manifest_path).unwrap();

    std::fs::write(
        &evidence_path,
        serde_json::to_vec_pretty(&serde_json::json!({
            "schema": "nerve.behavioral_validation.v1",
            "status": "passed",
            "candidate_kind": "exact_reference",
            "candidate_contract_digest_algorithm": CONTRACT_DIGEST_ALGORITHM
        }))
        .unwrap(),
    )
    .unwrap();
    let shallow_error = VulkanResidentModelPackageManifest::from_json_file(&manifest_path)
        .unwrap_err()
        .to_string();
    assert!(shallow_error.contains("source oracle contract"));

    let mut evidence: Value = serde_json::from_slice(
        &std::fs::read(source_root.join(&source_manifest.behavioral_validation_path)).unwrap(),
    )
    .unwrap();
    evidence["circuits"][0]["candidate_node_count"] = Value::from(u64::MAX);
    std::fs::write(
        &evidence_path,
        serde_json::to_vec_pretty(&evidence).unwrap(),
    )
    .unwrap();
    let stale_error = VulkanResidentModelPackageManifest::from_json_file(&manifest_path)
        .unwrap_err()
        .to_string();
    assert!(stale_error.contains("incomplete or stale proof"));

    let original_evidence =
        std::fs::read(source_root.join(&source_manifest.behavioral_validation_path)).unwrap();
    std::fs::write(&evidence_path, original_evidence).unwrap();
    let mut raw_manifest: Value =
        serde_json::from_slice(&std::fs::read(&source_manifest_path).unwrap()).unwrap();
    raw_manifest["circuit_graph"]["pedals"][0]["circuit"]["nodes"][0]["attrs"]["adversarial_drift"] =
        Value::from(true);
    std::fs::write(
        &manifest_path,
        serde_json::to_vec_pretty(&raw_manifest).unwrap(),
    )
    .unwrap();
    let same_size_stale_error = VulkanResidentModelPackageManifest::from_json_file(&manifest_path)
        .unwrap_err()
        .to_string();
    assert!(same_size_stale_error.contains("incomplete or stale proof"));

    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn package_loader_rejects_compiler_owned_runtime_placement() {
    let source_manifest_path = fixture_model_package_manifest_path();
    let raw_manifest: Value =
        serde_json::from_slice(&std::fs::read(source_manifest_path).unwrap()).unwrap();
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = std::env::temp_dir().join(format!(
        "nerve-compiler-placement-{}-{unique}",
        std::process::id()
    ));
    std::fs::create_dir_all(&root).unwrap();
    let manifest_path = root.join("vulkan_resident_package.json");

    for (field, value) in [
        ("device_id", Value::from("gpu0")),
        (
            "placement",
            serde_json::json!({
                "schema": "nerve.stream_circuit_placement.v1",
                "default_device_id": "gpu0",
                "pedal_devices": {}
            }),
        ),
    ] {
        let mut invalid_manifest = raw_manifest.clone();
        invalid_manifest[field] = value;
        std::fs::write(
            &manifest_path,
            serde_json::to_vec_pretty(&invalid_manifest).unwrap(),
        )
        .unwrap();

        let error = VulkanResidentModelPackageManifest::from_json_file(&manifest_path)
            .unwrap_err()
            .to_string();
        assert!(error.contains("must not contain runtime placement fields"));
        assert!(error.contains(field));
    }

    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn package_loader_rejects_paths_outside_the_package() {
    let mut manifest = fixture_model_package_manifest();
    manifest.config_path = "../config.json".to_string();
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = std::env::temp_dir().join(format!(
        "nerve-package-path-{}-{unique}",
        std::process::id()
    ));
    std::fs::create_dir_all(&root).unwrap();
    let manifest_path = root.join("vulkan_resident_package.json");
    manifest.write_json_file(&manifest_path).unwrap();

    let error = VulkanResidentModelPackageManifest::from_json_file(&manifest_path)
        .unwrap_err()
        .to_string();

    assert!(error.contains("config path"));
    assert!(error.contains("must stay inside the package"));
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn package_loader_rejects_same_length_shader_corruption() {
    let source_manifest_path = fixture_model_package_manifest_path();
    let source_manifest = fixture_model_package_manifest();
    let source_root = source_manifest_path.parent().unwrap();
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = std::env::temp_dir().join(format!(
        "nerve-package-integrity-{}-{unique}",
        std::process::id()
    ));
    std::fs::create_dir_all(&root).unwrap();
    copy_package_integrity_artifacts(source_root, &root, &source_manifest);
    let manifest_path = root.join("vulkan_resident_package.json");
    std::fs::copy(&source_manifest_path, &manifest_path).unwrap();
    let shader_path = root.join(&source_manifest.sampler.kernels[0].shader_path);
    let mut shader = std::fs::read(&shader_path).unwrap();
    *shader.last_mut().unwrap() ^= 0x01;
    std::fs::write(&shader_path, shader).unwrap();

    let error = VulkanResidentModelPackageManifest::from_json_file(&manifest_path)
        .unwrap_err()
        .to_string();

    assert!(error.contains("does not match its integrity contract"));
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn package_loader_accepts_mixed_exact_and_approximate_pedal_proofs() {
    let source_manifest_path = fixture_model_package_manifest_path();
    let source_manifest = fixture_model_package_manifest();
    let source_root = source_manifest_path.parent().unwrap();
    let mut evidence: Value = serde_json::from_slice(
        &std::fs::read(source_root.join(&source_manifest.behavioral_validation_path)).unwrap(),
    )
    .unwrap();
    evidence["candidate_kind"] = Value::from("approximate");
    evidence["teacher_forced"] = serde_json::json!({
        "status": "passed",
        "sample_count": 128,
        "metrics": {"maximum_logit_error": 0.01}
    });
    evidence["free_running"] = serde_json::json!({
        "status": "passed",
        "sample_count": 64,
        "metrics": {"distribution_similarity": 0.99}
    });
    evidence["circuits"][0]["candidate_kind"] = Value::from("approximate");

    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = std::env::temp_dir().join(format!(
        "nerve-mixed-behavioral-evidence-{}-{unique}",
        std::process::id()
    ));
    std::fs::create_dir_all(&root).unwrap();
    let manifest_path = root.join("vulkan_resident_package.json");
    copy_package_integrity_artifacts(source_root, &root, &source_manifest);
    let evidence_payload = serde_json::to_vec_pretty(&evidence).unwrap();
    std::fs::write(root.join("behavioral_validation.json"), &evidence_payload).unwrap();
    let mut raw_manifest: Value =
        serde_json::from_slice(&std::fs::read(&source_manifest_path).unwrap()).unwrap();
    raw_manifest["artifact_integrity"]["files"]["behavioral_validation.json"] = serde_json::json!({
        "byte_count": evidence_payload.len(),
        "sha256": Sha256::digest(&evidence_payload)
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>()
    });
    std::fs::write(
        &manifest_path,
        serde_json::to_vec_pretty(&raw_manifest).unwrap(),
    )
    .unwrap();

    VulkanResidentModelPackageManifest::from_json_file(&manifest_path).unwrap();

    evidence["circuits"][0]["candidate_kind"] = Value::from("exact_reference");
    std::fs::write(
        root.join("behavioral_validation.json"),
        serde_json::to_vec_pretty(&evidence).unwrap(),
    )
    .unwrap();
    let error = VulkanResidentModelPackageManifest::from_json_file(&manifest_path)
        .unwrap_err()
        .to_string();
    assert!(error.contains("no approximate pedal proof"));

    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn runtime_patch_duplicates_package_pedals_in_memory() {
    let manifest = fixture_model_package_manifest();
    let source_graph = manifest
        .circuit_graph
        .to_resolved_lowered_pedalboard(PathBuf::from("."))
        .unwrap();
    let patch = StreamCircuitRuntimePatch::from_source_series(&source_graph, "gpu0")
        .unwrap()
        .duplicate_after_instance(&source_graph, "layer_05", "layer_05_repeat")
        .unwrap()
        .with_instance_device("layer_05_repeat", "gpu1")
        .unwrap();

    let patched = manifest.clone().mount_runtime_patch(&patch).unwrap();

    assert_eq!(manifest.circuit_graph.pedals.len(), 17);
    assert_eq!(manifest.pedal_executions.len(), 14);
    assert!(
        manifest
            .circuit_graph
            .pedals
            .iter()
            .all(|pedal| pedal.pedal_id != "layer_05_repeat")
    );

    assert_eq!(patched.circuit_graph.pedals.len(), 18);
    assert_eq!(patched.pedal_executions.len(), 15);
    assert_eq!(patched.package, manifest);
    assert!(
        patched
            .patch
            .instances
            .iter()
            .any(|instance| instance.instance_id == "layer_05_repeat")
    );
    assert_eq!(
        patched.placement.device_for_pedal("layer_05_repeat"),
        "gpu1"
    );
    let repeated_pedal = patched
        .circuit_graph
        .pedals
        .iter()
        .find(|pedal| pedal.pedal_id == "layer_05_repeat")
        .unwrap();
    let source_pedal = manifest
        .circuit_graph
        .pedals
        .iter()
        .find(|pedal| pedal.pedal_id == "layer_05")
        .unwrap();
    assert_eq!(repeated_pedal.operator_type, source_pedal.operator_type);
    assert_eq!(repeated_pedal.circuit.id, source_pedal.circuit.id);
    assert_eq!(repeated_pedal.circuit.source.pedal_id, "layer_05_repeat");
    assert_eq!(
        repeated_pedal.params.refs.keys().collect::<Vec<_>>(),
        source_pedal.params.refs.keys().collect::<Vec<_>>()
    );

    let repeated_execution = patched
        .pedal_executions
        .iter()
        .find(|execution| execution.pedal_id == "layer_05_repeat")
        .unwrap();
    let source_execution = manifest
        .pedal_executions
        .iter()
        .find(|execution| execution.pedal_id == "layer_05")
        .unwrap();
    assert_eq!(repeated_execution.kernels, source_execution.kernels);
}

#[test]
fn runtime_model_coalesces_execution_placement_without_rewriting_the_patch() {
    let manifest = fixture_model_package_manifest();
    let source_graph = manifest
        .circuit_graph
        .to_resolved_lowered_pedalboard(PathBuf::from("."))
        .unwrap();
    let patch = StreamCircuitRuntimePatch::from_source_series(&source_graph, "gpu0")
        .unwrap()
        .with_instance_device("layer_05", "gpu1")
        .unwrap();
    let runtime_model = manifest.mount_runtime_patch(&patch).unwrap();
    assert_eq!(
        runtime_model.placement_device_ids(),
        vec!["gpu0".to_string(), "gpu1".to_string()]
    );

    let coalesced = runtime_model
        .clone()
        .coalesce_placement_to_device("physical_gpu0");

    assert_eq!(
        coalesced.placement_device_ids(),
        vec!["physical_gpu0".to_string()]
    );
    assert_eq!(coalesced.patch, runtime_model.patch);
    assert_eq!(coalesced.circuit_graph, runtime_model.circuit_graph);
    assert_eq!(coalesced.pedal_executions, runtime_model.pedal_executions);
}

#[test]
fn vulkan_lowering_extracts_signal_processors_from_the_full_pedalboard() {
    let manifest = fixture_model_package_manifest();
    let graph = manifest
        .circuit_graph
        .to_signal_processor_graph(PathBuf::from("."))
        .unwrap();

    assert_eq!(graph.circuits.len(), 14);
    assert!(
        graph
            .circuits
            .iter()
            .all(|artifact| artifact.circuit.runtime_role.is_signal_processor())
    );
    assert_eq!(graph.index.graph.cables.len(), 13);
    assert_eq!(
        graph.index.graph.boundary.external_inputs[0]
            .endpoint
            .pedal_id,
        "layer_00"
    );
    assert_eq!(
        graph.index.graph.boundary.public_outputs[0]
            .endpoint
            .pedal_id,
        "layer_13"
    );
}

#[test]
fn placed_generation_endpoints_follow_wiring_not_system_pedal_order() {
    let manifest = fixture_model_package_manifest();
    let (input_pedal, output_pedal) = manifest
        .circuit_graph
        .signal_processor_endpoint_pedal_ids()
        .unwrap();
    let placement = StreamCircuitPlacementSpec::new(RUNTIME_DEFAULT_LOGICAL_DEVICE_ID)
        .with_pedal_device(&input_pedal, "gpu-input")
        .with_pedal_device(&output_pedal, "gpu-output");

    assert_eq!(input_pedal, "layer_00");
    assert_eq!(output_pedal, "layer_13");
    assert_eq!(
        manifest
            .circuit_graph
            .signal_processor_device_ids(&placement),
        vec![
            "gpu-input".to_string(),
            "gpu-output".to_string(),
            "runtime_default".to_string(),
        ]
    );
}

#[test]
fn fused_generation_pedals_follow_connected_processor_devices() {
    let resolved = fixture_model_package_manifest()
        .circuit_graph
        .to_resolved_lowered_pedalboard(PathBuf::from("."))
        .unwrap();
    let patch = resolved
        .default_runtime_patch("gpu0")
        .unwrap()
        .with_instance_device("layer_00", "gpu-input")
        .unwrap()
        .with_instance_device("layer_13", "gpu-output")
        .unwrap();
    let patch = attach_generation_pedal_devices_for_vulkan(patch, &resolved).unwrap();
    let device_for = |instance_id: &str| {
        patch
            .instances
            .iter()
            .find(|instance| instance.instance_id == instance_id)
            .unwrap()
            .device_id
            .as_str()
    };

    assert_eq!(device_for("input_transducer"), "gpu-input");
    assert_eq!(device_for("output_transducer"), "gpu-output");
    assert_eq!(device_for("sampler"), "gpu-output");
}

#[test]
fn runtime_chain_control_preserves_generation_pedals_and_feedback() {
    let manifest = fixture_model_package_manifest();
    let chain = vec![
        ("first".to_string(), "layer_00".to_string()),
        ("repeat".to_string(), "layer_00".to_string()),
        ("last".to_string(), "layer_13".to_string()),
    ];

    let patch = manifest
        .runtime_patch_from_controls(None, &BTreeMap::new(), &[], Some(&chain))
        .unwrap();

    assert_eq!(patch.instances.len(), 6);
    for system_pedal in ["input_transducer", "output_transducer", "sampler"] {
        assert!(
            patch
                .instances
                .iter()
                .any(|instance| instance.instance_id == system_pedal)
        );
    }
    assert!(patch.cables.iter().any(|cable| {
        cable.source.pedal_id == "sampler"
            && cable.destination.pedal_id == "input_transducer"
            && !cable.connection.is_forward()
    }));
}

#[test]
fn generation_contract_rejects_execution_and_graph_drift() {
    let manifest = fixture_model_package_manifest();

    let mut sampler_drift = manifest.clone();
    sampler_drift.sampler.spec.top_k += 1;
    let sampler_error =
        validate_generation_execution_contract(&sampler_drift, &sampler_drift.circuit_graph)
            .unwrap_err()
            .to_string();
    assert!(sampler_error.contains("sampler execution does not match"));

    let mut extension_drift = manifest.clone();
    extension_drift.required_vulkan_device_extensions = vec![
        "VK_EXT_shader_float8".to_string(),
        "VK_EXT_shader_float8".to_string(),
    ];
    let extension_error =
        validate_generation_execution_contract(&extension_drift, &extension_drift.circuit_graph)
            .unwrap_err()
            .to_string();
    assert!(extension_error.contains("invalid required Vulkan device extensions"));

    let mut wiring_drift = manifest;
    wiring_drift
        .circuit_graph
        .cables
        .retain(|cable| cable.connection.is_forward());
    let wiring_error =
        validate_generation_execution_contract(&wiring_drift, &wiring_drift.circuit_graph)
            .unwrap_err()
            .to_string();
    assert!(wiring_error.contains("delayed sampler feedback"));

    let mut boundary_drift = fixture_model_package_manifest();
    boundary_drift.circuit_graph.boundary.public_outputs[0]
        .endpoint
        .port_id = "random_seed".to_string();
    let boundary_error =
        validate_generation_execution_contract(&boundary_drift, &boundary_drift.circuit_graph)
            .unwrap_err()
            .to_string();
    assert!(boundary_error.contains("sampler public output"));
}

