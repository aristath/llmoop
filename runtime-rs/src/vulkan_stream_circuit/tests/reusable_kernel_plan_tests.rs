#[test]
fn reusable_kernel_plan_keeps_compile_time_specializations_distinct() {
    let command =
        |dispatch_index: usize, component_id: &str, specialization: &str| VulkanKernelDispatchCommand {
            dispatch_index,
            circuit_index: dispatch_index,
            kernel_id: format!("{component_id}.per_layer_embedding"),
            component_id: component_id.to_string(),
            circuit_id: format!("{component_id}_circuit"),
            node_index: 0,
            node_id: "per_layer_embedding".to_string(),
            op: "per_layer_embedding".to_string(),
            specialization: specialization.to_string(),
            descriptor_bindings: Vec::new(),
            push_constants: Vec::new(),
            uses_stream_tick: true,
        };
    let dispatch_plan = VulkanKernelDispatchPlan {
        backend_id: VULKAN_STREAM_CIRCUIT_BACKEND_ID.to_string(),
        commands: vec![
            command(0, "layer_00", r#"{"layer_index":0}"#),
            command(1, "layer_01", r#"{"layer_index":1}"#),
        ],
    };

    let reusable_plan = VulkanReusableKernelPlan::from_dispatch_plan(&dispatch_plan);

    assert_eq!(reusable_plan.total_family_count(), 2);
    assert_eq!(reusable_plan.reusable_family_count(), 0);
    assert!(
        reusable_plan
            .families
            .iter()
            .all(|family| family.command_refs.len() == 1)
    );

    let reversed_dispatch_plan = VulkanKernelDispatchPlan {
        backend_id: VULKAN_STREAM_CIRCUIT_BACKEND_ID.to_string(),
        commands: dispatch_plan.commands.iter().rev().cloned().collect(),
    };
    let reversed_reusable_plan =
        VulkanReusableKernelPlan::from_dispatch_plan(&reversed_dispatch_plan);
    assert_eq!(
        reusable_plan
            .families
            .iter()
            .map(|family| family.family_id.as_str())
            .collect::<BTreeSet<_>>(),
        reversed_reusable_plan
            .families
            .iter()
            .map(|family| family.family_id.as_str())
            .collect::<BTreeSet<_>>()
    );
}

#[test]
fn parses_explicit_shared_state_sources() {
    assert_eq!(
        shared_state_source("shared_from:layer_22.kv_memory").unwrap(),
        Some(("layer_22".to_string(), "kv_memory".to_string()))
    );
    assert_eq!(shared_state_source("private").unwrap(), None);
    assert!(shared_state_source("shared_from:kv_memory").is_err());
}

#[test]
fn reusable_kernel_plan_collapses_fixture_model_dispatches_into_op_families() {
    let graph = fixture_model_execution_graph();
    let tensor_index = TensorIndex::from_json_file(fixture_model_tensor_index_path()).unwrap();
    let execution_plan =
        StreamCircuitExecutionPlan::from_graph_with_tensor_index(&graph, &tensor_index).unwrap();
    let resource_plan =
        StreamCircuitResourcePlan::from_graph_and_plan(&graph, &execution_plan).unwrap();
    let resident_plan = VulkanStreamCircuitResidentPlan::from_resource_plan(
        &resource_plan,
        Some(&tensor_index),
        Some(2),
    )
    .unwrap();
    let binding_plan =
        VulkanStreamCircuitBindingPlan::from_plans(&execution_plan, &resource_plan, &resident_plan)
            .unwrap();
    let dispatch_plan = VulkanKernelDispatchPlan::from_binding_plan(&binding_plan);

    let reusable_plan = VulkanReusableKernelPlan::from_dispatch_plan(&dispatch_plan);

    assert_eq!(reusable_plan.backend_id, VULKAN_STREAM_CIRCUIT_BACKEND_ID);
    assert_eq!(reusable_plan.total_command_count, 242);
    assert_eq!(reusable_plan.total_family_count(), 26);
    assert_eq!(reusable_plan.reusable_family_count(), 26);
    assert_eq!(reusable_plan.families_for_op("rms_norm").len(), 4);
    assert_eq!(reusable_plan.families_for_op("linear").len(), 6);

    let linear = reusable_family_with_kernel(&reusable_plan, "layer_00.conv_in_projection");
    assert_eq!(linear.op, "linear");
    assert_eq!(linear.command_refs.len(), 8);
    assert!(!linear.uses_stream_tick);
    assert_eq!(
        linear.descriptor_signature,
        vec![
            VulkanKernelDescriptorSlotSignature {
                binding: 0,
                usage: VulkanKernelDescriptorUsage::InputSignal,
                resource_class: VulkanKernelDescriptorResourceClass::SignalBuffer,
                byte_capacity: Some(5_120),
                shape: None,
            },
            VulkanKernelDescriptorSlotSignature {
                binding: 1,
                usage: VulkanKernelDescriptorUsage::OutputSignal,
                resource_class: VulkanKernelDescriptorResourceClass::SignalBuffer,
                byte_capacity: Some(6_144),
                shape: None,
            },
            VulkanKernelDescriptorSlotSignature {
                binding: 2,
                usage: VulkanKernelDescriptorUsage::Parameter,
                resource_class: VulkanKernelDescriptorResourceClass::ParameterBuffer,
                byte_capacity: Some(6_291_456),
                shape: Some(vec![3072, 1024]),
            },
        ]
    );
    assert_eq!(linear.command_refs[0].dispatch_index, 1);
    assert_eq!(
        linear.command_refs[0].kernel_id,
        "layer_00.conv_in_projection"
    );
    assert_eq!(
        linear.command_refs.last().unwrap().kernel_id,
        "layer_13.conv_in_projection"
    );

    let rope = reusable_plan
        .families_for_op("rotary_position_embedding")
        .into_iter()
        .find(|family| family.command_refs.len() == 6)
        .unwrap();
    assert_eq!(rope.command_refs.len(), 6);
    assert_eq!(
        reusable_plan
            .families_for_op("rotary_position_embedding")
            .iter()
            .map(|family| family.command_refs.len())
            .sum::<usize>(),
        12
    );
    assert!(rope.uses_stream_tick);
    assert!(rope.push_constants.is_empty());

    let append = reusable_plan
        .families_for_op("append_state_update")
        .into_iter()
        .find(|family| family.command_refs.len() == 6)
        .unwrap();
    assert_eq!(append.command_refs.len(), 6);
    assert!(append.uses_stream_tick);
    assert_eq!(
        append
            .descriptor_signature
            .iter()
            .map(|slot| (
                slot.binding,
                slot.usage.clone(),
                slot.resource_class.clone()
            ))
            .collect::<Vec<_>>(),
        vec![
            (
                0,
                VulkanKernelDescriptorUsage::InputSignal,
                VulkanKernelDescriptorResourceClass::SignalBuffer,
            ),
            (
                1,
                VulkanKernelDescriptorUsage::InputSignal,
                VulkanKernelDescriptorResourceClass::SignalBuffer,
            ),
            (
                2,
                VulkanKernelDescriptorUsage::InputSignal,
                VulkanKernelDescriptorResourceClass::SignalBuffer,
            ),
            (
                3,
                VulkanKernelDescriptorUsage::OutputSignal,
                VulkanKernelDescriptorResourceClass::SignalBuffer,
            ),
            (
                4,
                VulkanKernelDescriptorUsage::OutputSignal,
                VulkanKernelDescriptorResourceClass::SignalBuffer,
            ),
            (
                5,
                VulkanKernelDescriptorUsage::StateRead,
                VulkanKernelDescriptorResourceClass::StateBuffer,
            ),
            (
                6,
                VulkanKernelDescriptorUsage::StateWrite,
                VulkanKernelDescriptorResourceClass::StateBuffer,
            ),
            (
                7,
                VulkanKernelDescriptorUsage::StateView,
                VulkanKernelDescriptorResourceClass::SignalBuffer,
            ),
            (
                8,
                VulkanKernelDescriptorUsage::StateView,
                VulkanKernelDescriptorResourceClass::SignalBuffer,
            ),
        ]
    );

    let split = reusable_plan
        .families_for_op("split")
        .into_iter()
        .find(|family| family.command_refs.len() == 8)
        .unwrap();
    assert_eq!(split.command_refs.len(), 8);
    assert_eq!(split.descriptor_signature.len(), 4);
}

#[test]
fn reusable_kernel_coverage_reports_missing_gpu_component_circuits() {
    let graph = fixture_model_execution_graph();
    let tensor_index = TensorIndex::from_json_file(fixture_model_tensor_index_path()).unwrap();
    let execution_plan =
        StreamCircuitExecutionPlan::from_graph_with_tensor_index(&graph, &tensor_index).unwrap();
    let resource_plan =
        StreamCircuitResourcePlan::from_graph_and_plan(&graph, &execution_plan).unwrap();
    let resident_plan = VulkanStreamCircuitResidentPlan::from_resource_plan(
        &resource_plan,
        Some(&tensor_index),
        Some(2),
    )
    .unwrap();
    let binding_plan =
        VulkanStreamCircuitBindingPlan::from_plans(&execution_plan, &resource_plan, &resident_plan)
            .unwrap();
    let dispatch_plan = VulkanKernelDispatchPlan::from_binding_plan(&binding_plan);
    let reusable_plan = VulkanReusableKernelPlan::from_dispatch_plan(&dispatch_plan);
    let conv_in_family = reusable_family_with_kernel(&reusable_plan, "layer_00.conv_in_projection");
    let conv_in_family_id = conv_in_family.family_id.as_str();

    let empty = reusable_plan.coverage_report(std::iter::empty::<&str>());
    assert!(!empty.all_available());
    assert_eq!(empty.required_family_count, 26);
    assert_eq!(empty.available_family_count, 0);
    assert_eq!(empty.missing_family_count, 26);
    assert_eq!(empty.required_command_count, 242);
    assert_eq!(empty.covered_command_count, 0);
    assert_eq!(empty.missing_command_count, 242);
    assert!(
        empty
            .missing_families()
            .iter()
            .any(|family| family.family_id == conv_in_family_id && family.command_count == 8)
    );

    let rms_norm_family_id = reusable_plan.families_for_op("rms_norm")[1]
        .family_id
        .as_str();
    let partial_family_ids = [conv_in_family_id, rms_norm_family_id];
    let partial_covered_command_count = partial_family_ids
        .iter()
        .map(|family_id| reusable_plan.family(family_id).unwrap().command_refs.len())
        .sum::<usize>();
    let partial = reusable_plan.coverage_report(partial_family_ids);
    assert!(!partial.all_available());
    assert_eq!(partial.available_family_count, 2);
    assert_eq!(partial.missing_family_count, 24);
    assert_eq!(partial.covered_command_count, partial_covered_command_count);
    assert_eq!(
        partial.missing_command_count,
        242 - partial_covered_command_count
    );
    assert_eq!(partial.missing_families().len(), 24);

    let full = reusable_plan.coverage_report(
        reusable_plan
            .families
            .iter()
            .map(|family| family.family_id.as_str()),
    );
    assert!(full.all_available());
    assert_eq!(full.available_family_count, 26);
    assert_eq!(full.missing_family_count, 0);
    assert_eq!(full.covered_command_count, 242);
    assert_eq!(full.missing_command_count, 0);
}

#[test]
fn reusable_kernel_artifact_manifest_links_fixture_model_kernel_families() {
    let graph = fixture_model_execution_graph();
    let tensor_index = TensorIndex::from_json_file(fixture_model_tensor_index_path()).unwrap();
    let execution_plan =
        StreamCircuitExecutionPlan::from_graph_with_tensor_index(&graph, &tensor_index).unwrap();
    let resource_plan =
        StreamCircuitResourcePlan::from_graph_and_plan(&graph, &execution_plan).unwrap();
    let resident_plan = VulkanStreamCircuitResidentPlan::from_resource_plan(
        &resource_plan,
        Some(&tensor_index),
        Some(2),
    )
    .unwrap();
    let binding_plan =
        VulkanStreamCircuitBindingPlan::from_plans(&execution_plan, &resource_plan, &resident_plan)
            .unwrap();
    let dispatch_plan = VulkanKernelDispatchPlan::from_binding_plan(&binding_plan);
    let reusable_plan = VulkanReusableKernelPlan::from_dispatch_plan(&dispatch_plan);
    let conv_in_family = reusable_family_with_kernel(&reusable_plan, "layer_00.conv_in_projection");
    let conv_in_family_id = conv_in_family.family_id.as_str();
    let conv_in_artifact_path = artifact_path_for_family(conv_in_family);
    let manifest = VulkanReusableKernelArtifactManifest::new(
        reusable_plan
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

    let link_plan = reusable_plan.link_artifacts(&manifest);

    assert_eq!(
        manifest.schema,
        VULKAN_REUSABLE_KERNEL_ARTIFACT_MANIFEST_SCHEMA
    );
    assert_eq!(manifest.backend_id, VULKAN_STREAM_CIRCUIT_BACKEND_ID);
    assert_eq!(manifest.artifacts.len(), 26);
    assert!(link_plan.is_fully_linked());
    assert_eq!(link_plan.required_family_count, 26);
    assert_eq!(link_plan.linked_family_count, 26);
    assert_eq!(link_plan.missing_family_count, 0);
    assert_eq!(link_plan.incompatible_family_count, 0);
    assert_eq!(link_plan.required_command_count, 242);
    assert_eq!(link_plan.linked_command_count, 242);
    assert_eq!(link_plan.missing_command_count, 0);
    assert_eq!(link_plan.incompatible_command_count, 0);
    assert!(link_plan.issues.is_empty());

    let linear = link_plan.family(conv_in_family_id).unwrap();
    assert_eq!(linear.status, VulkanReusableKernelLinkStatus::Linked);
    assert_eq!(linear.command_count, 8);
    assert_eq!(
        linear.artifact_path.as_deref(),
        Some(conv_in_artifact_path.as_str())
    );

    let manifest_path = std::env::temp_dir().join(format!(
        "nerve-reusable-kernel-manifest-{}.json",
        std::process::id()
    ));
    manifest.write_json_file(&manifest_path).unwrap();
    let read = VulkanReusableKernelArtifactManifest::from_json_file(&manifest_path).unwrap();
    std::fs::remove_file(&manifest_path).unwrap();
    assert_eq!(read, manifest);
    assert_eq!(read.family_ids().len(), 26);

    let artifact_root = std::env::temp_dir().join(format!(
        "nerve-reusable-kernel-artifacts-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(artifact_root.join("kernels")).unwrap();
    for (index, artifact) in manifest.artifacts.iter().enumerate() {
        crate::vulkan::write_spirv_words(
            artifact_root.join(&artifact.path),
            &[0x0723_0203, index as u32],
        )
        .unwrap();
    }

    let loaded = manifest.load_artifacts(&artifact_root).unwrap();

    assert_eq!(
        loaded.schema,
        VULKAN_REUSABLE_KERNEL_ARTIFACT_MANIFEST_SCHEMA
    );
    assert_eq!(loaded.backend_id, VULKAN_STREAM_CIRCUIT_BACKEND_ID);
    assert_eq!(loaded.artifacts.len(), 26);
    assert_eq!(loaded.family_ids().len(), 26);
    assert_eq!(loaded.total_word_count, 52);
    let loaded_linear = loaded.artifact(conv_in_family_id).unwrap();
    assert_eq!(loaded_linear.artifact.family_id, conv_in_family_id);
    assert_eq!(
        loaded_linear.resolved_path,
        artifact_root.join(&conv_in_artifact_path)
    );
    assert_eq!(loaded_linear.words[0], 0x0723_0203);
    std::fs::remove_dir_all(&artifact_root).unwrap();
}

#[test]
fn reusable_kernel_link_plan_reports_partial_and_incompatible_artifacts() {
    let graph = fixture_model_execution_graph();
    let tensor_index = TensorIndex::from_json_file(fixture_model_tensor_index_path()).unwrap();
    let execution_plan =
        StreamCircuitExecutionPlan::from_graph_with_tensor_index(&graph, &tensor_index).unwrap();
    let resource_plan =
        StreamCircuitResourcePlan::from_graph_and_plan(&graph, &execution_plan).unwrap();
    let resident_plan = VulkanStreamCircuitResidentPlan::from_resource_plan(
        &resource_plan,
        Some(&tensor_index),
        Some(2),
    )
    .unwrap();
    let binding_plan =
        VulkanStreamCircuitBindingPlan::from_plans(&execution_plan, &resource_plan, &resident_plan)
            .unwrap();
    let dispatch_plan = VulkanKernelDispatchPlan::from_binding_plan(&binding_plan);
    let reusable_plan = VulkanReusableKernelPlan::from_dispatch_plan(&dispatch_plan);
    let linear = reusable_family_with_kernel(&reusable_plan, "layer_00.conv_in_projection");
    let linear_family_id = linear.family_id.as_str();
    let linear_artifact_path = artifact_path_for_family(linear);

    let partial_manifest = VulkanReusableKernelArtifactManifest::empty().with_artifact(
        VulkanReusableKernelArtifact::from_family(linear, linear_artifact_path),
    );
    let partial_link = reusable_plan.link_artifacts(&partial_manifest);

    assert!(!partial_link.is_fully_linked());
    assert_eq!(partial_link.linked_family_count, 1);
    assert_eq!(partial_link.missing_family_count, 25);
    assert_eq!(partial_link.incompatible_family_count, 0);
    assert_eq!(partial_link.linked_command_count, 8);
    assert_eq!(partial_link.missing_command_count, 242 - 8);
    assert_eq!(
        partial_link.family(linear_family_id).unwrap().status,
        VulkanReusableKernelLinkStatus::Linked
    );
    assert!(
        partial_link
            .missing_families()
            .iter()
            .any(|family| family.op == "append_state_update")
    );

    let mut bad_linear = VulkanReusableKernelArtifact::from_family(linear, "")
        .with_entry_point("not_main")
        .with_local_size_x(0);
    bad_linear.op = "multiply".to_string();
    bad_linear.descriptor_signature.pop();
    let incompatible_manifest =
        VulkanReusableKernelArtifactManifest::empty().with_artifact(bad_linear);
    let incompatible_link = reusable_plan.link_artifacts(&incompatible_manifest);

    assert!(!incompatible_link.is_fully_linked());
    assert_eq!(incompatible_link.linked_family_count, 0);
    assert_eq!(incompatible_link.missing_family_count, 25);
    assert_eq!(incompatible_link.incompatible_family_count, 1);
    assert_eq!(incompatible_link.incompatible_command_count, 8);
    assert_eq!(incompatible_link.missing_command_count, 242 - 8);
    let linear_link = incompatible_link.family(linear_family_id).unwrap();
    assert_eq!(
        linear_link.status,
        VulkanReusableKernelLinkStatus::Incompatible
    );
    assert!(linear_link.issues.iter().any(|issue| matches!(
        issue.problem,
        VulkanReusableKernelLinkProblem::OpMismatch { .. }
    )));
    assert!(linear_link.issues.iter().any(|issue| matches!(
        issue.problem,
        VulkanReusableKernelLinkProblem::DescriptorSignatureMismatch
    )));
    assert!(linear_link.issues.iter().any(|issue| matches!(
        issue.problem,
        VulkanReusableKernelLinkProblem::EmptySpirvPath
    )));
    assert!(linear_link.issues.iter().any(|issue| matches!(
        issue.problem,
        VulkanReusableKernelLinkProblem::UnsupportedEntryPoint { .. }
    )));
    assert!(linear_link.issues.iter().any(|issue| matches!(
        issue.problem,
        VulkanReusableKernelLinkProblem::InvalidLocalSizeX { .. }
    )));
    assert_eq!(incompatible_link.incompatible_families().len(), 1);
}

