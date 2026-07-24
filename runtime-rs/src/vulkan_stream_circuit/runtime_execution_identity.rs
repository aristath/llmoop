fn canonical_runtime_execution_identity(
    runtime_model: &VulkanResidentRuntimeModel,
    dynamic_state_capacity_activations: usize,
    speculative_decoders_enabled: bool,
) -> Result<String, VulkanResidentTokenModelPackageError> {
    let mut instances = runtime_model.runtime_graph.instances.clone();
    instances.sort_by(|left, right| left.instance_id.cmp(&right.instance_id));
    let mut edges = runtime_model.runtime_graph.edges.clone();
    edges.sort_by(|left, right| {
        (
            left.source.component_id.as_str(),
            left.source.port_id.as_str(),
            left.destination.component_id.as_str(),
            left.destination.port_id.as_str(),
            left.id.as_str(),
        )
            .cmp(&(
                right.source.component_id.as_str(),
                right.source.port_id.as_str(),
                right.destination.component_id.as_str(),
                right.destination.port_id.as_str(),
                right.id.as_str(),
            ))
    });
    let mut external_inputs = runtime_model.runtime_graph.boundary.external_inputs.clone();
    external_inputs.sort_by(|left, right| left.id.cmp(&right.id));
    let mut public_outputs = runtime_model.runtime_graph.boundary.public_outputs.clone();
    public_outputs.sort_by(|left, right| left.id.cmp(&right.id));
    let mut component_executions = runtime_model.component_executions.clone();
    component_executions.sort_by(|left, right| left.component_id.cmp(&right.component_id));

    let identity = serde_json::json!({
        "schema": "nerve.runtime_execution_identity.v1",
        "package": {
            "id": runtime_model.package.package_id,
            "compiler_fingerprint": runtime_model.package.compiler_fingerprint,
            "artifact_integrity": runtime_model.package.artifact_integrity,
        },
        "graph": {
            "schema": runtime_model.runtime_graph.schema,
            "topology": runtime_model.runtime_graph.topology,
            "default_device_id": runtime_model.runtime_graph.default_device_id,
            "instances": instances,
            "edges": edges,
            "boundary": {
                "external_inputs": external_inputs,
                "public_outputs": public_outputs,
            },
        },
        "component_executions": component_executions,
        "state_capacity_activations": dynamic_state_capacity_activations,
        "speculative_decoders_enabled": speculative_decoders_enabled,
    });
    let bytes = serde_json::to_vec(&identity).map_err(|error| {
        VulkanResidentTokenModelPackageError::new(format!(
            "failed to serialize canonical runtime execution identity: {error}"
        ))
    })?;
    let digest = Sha256::digest(bytes);
    Ok(digest.iter().map(|byte| format!("{byte:02x}")).collect())
}

#[cfg(test)]
mod runtime_execution_identity_tests {
    use super::*;

    #[test]
    fn canonical_execution_identity_ignores_non_semantic_graph_storage_order() {
        let mut left = tests::tiny_fixture_model_runtime_model_with_placement(
            StreamCircuitPlacementSpec::new("gpu0"),
        );
        let mut right = left.clone();
        right.runtime_graph.instances.reverse();
        right.runtime_graph.edges.reverse();
        right.runtime_graph.boundary.external_inputs.reverse();
        right.runtime_graph.boundary.public_outputs.reverse();
        right.component_executions.reverse();

        assert_eq!(
            canonical_runtime_execution_identity(&left, 4096, false).unwrap(),
            canonical_runtime_execution_identity(&right, 4096, false).unwrap()
        );

        left.runtime_graph.instances[0].device_id = "gpu1".to_string();
        assert_ne!(
            canonical_runtime_execution_identity(&left, 4096, false).unwrap(),
            canonical_runtime_execution_identity(&right, 4096, false).unwrap()
        );
    }

    #[test]
    fn canonical_execution_identity_includes_capacity_and_kernel_selection() {
        let model = tests::tiny_fixture_model_runtime_model_with_placement(
            StreamCircuitPlacementSpec::new("gpu0"),
        );
        let base = canonical_runtime_execution_identity(&model, 4096, false).unwrap();
        assert_ne!(
            base,
            canonical_runtime_execution_identity(&model, 8192, false).unwrap()
        );

        let mut changed_kernel = model.clone();
        changed_kernel.component_executions[0].kernels[0].local_size_x += 1;
        assert_ne!(
            base,
            canonical_runtime_execution_identity(&changed_kernel, 4096, false).unwrap()
        );
    }
}
