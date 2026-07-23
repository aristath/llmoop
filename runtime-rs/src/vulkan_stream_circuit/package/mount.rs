impl VulkanResidentModelPackageManifest {
    pub fn from_json_file(path: impl AsRef<Path>) -> io::Result<Self> {
        let path = path.as_ref();
        let bytes = fs::read(path)?;
        let raw_manifest: Value = serde_json::from_slice(&bytes)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        let schema = raw_manifest
            .get("schema")
            .and_then(Value::as_str)
            .unwrap_or("<missing>");
        if schema != VULKAN_RESIDENT_MODEL_PACKAGE_MANIFEST_SCHEMA {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "unsupported resident model package manifest schema {schema:?}; recompile the model"
                ),
            ));
        }
        let compiler_fingerprint = raw_manifest
            .get("compiler_fingerprint")
            .and_then(Value::as_str)
            .unwrap_or("<missing>");
        if compiler_fingerprint != VULKAN_PACKAGE_COMPILER_FINGERPRINT {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "compiled package fingerprint {compiler_fingerprint:?} does not match runtime fingerprint {VULKAN_PACKAGE_COMPILER_FINGERPRINT:?}; recompile the model"
                ),
            ));
        }
        let compiler_owned_placement = ["device_id", "placement"]
            .into_iter()
            .filter(|field| raw_manifest.get(field).is_some())
            .collect::<Vec<_>>();
        if !compiler_owned_placement.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "compiled model package must not contain runtime placement fields {compiler_owned_placement:?}"
                ),
            ));
        }
        let manifest: Self = serde_json::from_value(raw_manifest.clone())
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        validate_resident_package_paths(&manifest)?;
        validate_behavioral_validation_artifact(path, &manifest, &raw_manifest)?;
        validate_resident_package_artifact_integrity(path, &manifest)?;
        let package_root = path.parent().unwrap_or_else(|| Path::new("."));
        validate_resident_package_spirv_requirements(package_root, &manifest)?;
        let source_graph = manifest
            .resolved_source_graph(package_root)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error.to_string()))?;
        validate_component_executions_against_graph(
            &manifest.package_id,
            &manifest.component_executions,
            &source_graph,
        )
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error.to_string()))?;
        validate_generation_execution_contract(&manifest, &manifest.circuit_graph)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error.to_string()))?;
        Ok(manifest)
    }

    pub fn write_json_file(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let bytes = serde_json::to_vec_pretty(self)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        fs::write(path, bytes)
    }

    pub fn mount_runtime_graph_controls(
        self,
        default_device_id: Option<&str>,
        node_devices: &BTreeMap<String, String>,
        duplicate_after: &[(String, String)],
        source_chain: Option<&[(String, String)]>,
    ) -> Result<VulkanResidentRuntimeModel, VulkanResidentTokenModelPackageError> {
        let runtime_graph = self.runtime_graph_from_controls(
            default_device_id,
            node_devices,
            duplicate_after,
            source_chain,
        )?;

        self.mount_runtime_graph(&runtime_graph)
    }

    pub fn runtime_graph_from_controls(
        &self,
        default_device_id: Option<&str>,
        node_devices: &BTreeMap<String, String>,
        duplicate_after: &[(String, String)],
        source_chain: Option<&[(String, String)]>,
    ) -> Result<StreamCircuitRuntimeGraph, VulkanResidentTokenModelPackageError> {
        let source_graph = self
            .circuit_graph
            .to_resolved_lowered_execution_graph(PathBuf::from("."))?;
        let default_device_id = default_device_id
            .unwrap_or(RUNTIME_DEFAULT_LOGICAL_DEVICE_ID)
            .to_string();
        let mut runtime_graph =
            StreamCircuitRuntimeGraph::from_source_series(&source_graph, default_device_id)
                .map_err(|error| VulkanResidentTokenModelPackageError::new(error.to_string()))?;
        if let Some(source_chain) = source_chain {
            runtime_graph = runtime_graph
                .with_signal_processor_chain(&source_graph, source_chain)
                .map_err(|error| VulkanResidentTokenModelPackageError::new(error.to_string()))?;
        }
        for (after_instance_id, new_instance_id) in duplicate_after {
            runtime_graph = runtime_graph
                .duplicate_after_instance(&source_graph, after_instance_id, new_instance_id)
                .map_err(|error| VulkanResidentTokenModelPackageError::new(error.to_string()))?;
        }
        for (instance_id, device_id) in node_devices {
            let instance = runtime_graph
                .instances
                .iter()
                .find(|instance| instance.instance_id == *instance_id)
                .ok_or_else(|| {
                    VulkanResidentTokenModelPackageError::new(format!(
                        "runtime graph has no node instance {instance_id:?}"
                    ))
                })?;
            let source = source_graph
                .circuits
                .iter()
                .find(|artifact| artifact.component.id == instance.source_component_id)
                .expect("validated runtime graph source must exist");
            if !source.component.runtime_role.is_signal_processor() {
                return Err(VulkanResidentTokenModelPackageError::new(format!(
                    "component {instance_id:?} is attached to the processor boundary and cannot be placed independently by the Vulkan backend"
                )));
            }
            runtime_graph = runtime_graph
                .with_instance_device(instance_id, device_id)
                .map_err(|error| VulkanResidentTokenModelPackageError::new(error.to_string()))?;
        }
        attach_generation_node_devices_for_vulkan(runtime_graph, &source_graph)
            .map_err(|error| VulkanResidentTokenModelPackageError::new(error.to_string()))
    }

    pub fn resolved_source_graph(
        &self,
        package_root: impl Into<PathBuf>,
    ) -> Result<ResolvedLoweredExecutionGraph, VulkanResidentTokenModelPackageError> {
        self.circuit_graph
            .to_resolved_lowered_execution_graph(package_root)
    }

    pub fn mount_runtime_graph(
        self,
        runtime_graph: &StreamCircuitRuntimeGraph,
    ) -> Result<VulkanResidentRuntimeModel, VulkanResidentTokenModelPackageError> {
        let source_graph = self
            .circuit_graph
            .to_resolved_lowered_execution_graph(PathBuf::from("."))?;
        let runtime_graph = attach_generation_node_devices_for_vulkan(runtime_graph.clone(), &source_graph)
            .map_err(|error| VulkanResidentTokenModelPackageError::new(error.to_string()))?;
        runtime_graph
            .validate_against_graph(&source_graph)
            .map_err(|error| VulkanResidentTokenModelPackageError::new(error.to_string()))?;

        let source_components = self
            .circuit_graph
            .components
            .iter()
            .map(|component| (component.component_id.as_str(), component))
            .collect::<BTreeMap<_, _>>();
        let source_executions = self
            .component_executions
            .iter()
            .map(|execution| (execution.component_id.as_str(), execution))
            .collect::<BTreeMap<_, _>>();

        let ordered_instance_ids = runtime_graph
            .topological_instance_ids(&source_graph)
            .map_err(|error| VulkanResidentTokenModelPackageError::new(error.to_string()))?;
        let enabled_instance_count = ordered_instance_ids.len();
        let mut components = Vec::with_capacity(enabled_instance_count);
        let mut component_executions = Vec::with_capacity(enabled_instance_count);
        let mut placement = StreamCircuitPlacementSpec::new(runtime_graph.default_device_id.clone());

        for instance_id in ordered_instance_ids {
            let instance = runtime_graph
                .instances
                .iter()
                .find(|instance| instance.instance_id == instance_id)
                .expect("validated topological instance id must exist");
            let source_component = source_components
                .get(instance.source_component_id.as_str())
                .ok_or_else(|| {
                    VulkanResidentTokenModelPackageError::new(format!(
                        "runtime graph instance {} references unknown source component {}",
                        instance.instance_id, instance.source_component_id
                    ))
                })?;
            let mut component = (*source_component).clone();
            component.component_id = instance.instance_id.clone();
            component.circuit.source.component_id = instance.instance_id.clone();
            apply_runtime_graph_state_policy(&mut component, &runtime_graph, instance);
            components.push(component);

            if source_component.runtime_role.is_signal_processor() {
                let source_execution = source_executions
                    .get(instance.source_component_id.as_str())
                    .ok_or_else(|| {
                        VulkanResidentTokenModelPackageError::new(format!(
                            "runtime graph signal processor {} has no execution spec",
                            instance.source_component_id
                        ))
                    })?;
                let mut execution = (*source_execution).clone();
                execution.component_id = instance.instance_id.clone();
                component_executions.push(execution);
            }

            if instance.device_id != runtime_graph.default_device_id {
                placement = placement.with_component_device(&instance.instance_id, &instance.device_id);
            }
        }

        let mut circuit_graph = self.circuit_graph.clone();
        circuit_graph.topology = runtime_graph.topology.clone();
        circuit_graph.edges = runtime_graph
            .effective_edges()
            .map_err(|error| VulkanResidentTokenModelPackageError::new(error.to_string()))?;
        circuit_graph.boundary = runtime_graph.boundary.clone();
        circuit_graph.components = components;
        validate_generation_execution_contract(&self, &circuit_graph)?;
        Ok(VulkanResidentRuntimeModel {
            package: self,
            runtime_graph,
            placement,
            circuit_graph,
            component_executions,
        })
    }
}

impl VulkanResidentRuntimeModel {
    pub fn placement_device_ids(&self) -> Vec<String> {
        self.circuit_graph
            .components
            .iter()
            .map(|component| self.placement.device_for_component(&component.component_id).to_string())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }

    pub fn resolved_graph(
        &self,
        package_root: impl Into<PathBuf>,
    ) -> Result<ResolvedLoweredExecutionGraph, VulkanResidentTokenModelPackageError> {
        self.circuit_graph
            .to_resolved_lowered_execution_graph(package_root)
    }

    pub fn coalesce_placement_to_device(mut self, device_id: impl Into<String>) -> Self {
        self.placement = StreamCircuitPlacementSpec::new(device_id);
        self
    }
}

pub(crate) fn attach_generation_node_devices_for_vulkan(
    mut runtime_graph: StreamCircuitRuntimeGraph,
    graph: &ResolvedLoweredExecutionGraph,
) -> Result<StreamCircuitRuntimeGraph, crate::stream_circuit::CircuitPlacementError> {
    runtime_graph.validate_against_graph(graph)?;
    let source_by_id = graph
        .circuits
        .iter()
        .map(|artifact| (artifact.component.id.as_str(), artifact))
        .collect::<BTreeMap<_, _>>();
    let role_for = |instance: &crate::stream_circuit::StreamCircuitNodeInstance| {
        source_by_id[instance.source_component_id.as_str()]
            .component
            .runtime_role
    };
    let instances_with_role = |role: CircuitRuntimeRole| {
        runtime_graph
            .instances
            .iter()
            .filter(|instance| role_for(instance) == role)
            .map(|instance| instance.instance_id.clone())
            .collect::<Vec<_>>()
    };
    let input_transducer_ids = instances_with_role(CircuitRuntimeRole::InputTransducer);
    let [input_transducer_id] = input_transducer_ids.as_slice() else {
        return Err(crate::stream_circuit::CircuitPlacementError(
            "Vulkan generation placement requires exactly one input transducer".to_string(),
        ));
    };
    let input_transducer_id = input_transducer_id.clone();
    let output_transducer_ids = instances_with_role(CircuitRuntimeRole::OutputTransducer);
    let [output_transducer_id] = output_transducer_ids.as_slice() else {
        return Err(crate::stream_circuit::CircuitPlacementError(
            "Vulkan generation placement requires exactly one output transducer".to_string(),
        ));
    };
    let output_transducer_id = output_transducer_id.clone();
    let sampler_ids = instances_with_role(CircuitRuntimeRole::Sampler);
    let [sampler_id] = sampler_ids.as_slice() else {
        return Err(crate::stream_circuit::CircuitPlacementError(
            "Vulkan generation placement requires exactly one sampler".to_string(),
        ));
    };
    let sampler_id = sampler_id.clone();
    let processor_ids = runtime_graph
        .instances
        .iter()
        .filter(|instance| role_for(instance).is_signal_processor())
        .map(|instance| instance.instance_id.as_str())
        .collect::<BTreeSet<_>>();
    let input_edges = runtime_graph
        .edges
        .iter()
        .filter(|edge| {
            edge.connection.is_forward()
                && edge.source.component_id == input_transducer_id
                && processor_ids.contains(edge.destination.component_id.as_str())
        })
        .collect::<Vec<_>>();
    let output_edges = runtime_graph
        .edges
        .iter()
        .filter(|edge| {
            edge.connection.is_forward()
                && processor_ids.contains(edge.source.component_id.as_str())
                && edge.destination.component_id == output_transducer_id
        })
        .collect::<Vec<_>>();
    let ([input_edge], [output_edge]) = (input_edges.as_slice(), output_edges.as_slice()) else {
        return Err(crate::stream_circuit::CircuitPlacementError(
            "Vulkan generation placement requires one processor input and output boundary"
                .to_string(),
        ));
    };
    let device_by_instance = runtime_graph
        .instances
        .iter()
        .map(|instance| (instance.instance_id.as_str(), instance.device_id.clone()))
        .collect::<BTreeMap<_, _>>();
    let input_device = device_by_instance[input_edge.destination.component_id.as_str()].clone();
    let output_device = device_by_instance[output_edge.source.component_id.as_str()].clone();
    for instance in &mut runtime_graph.instances {
        if instance.instance_id == input_transducer_id {
            instance.device_id = input_device.clone();
        } else if instance.instance_id == output_transducer_id || instance.instance_id == sampler_id
        {
            instance.device_id = output_device.clone();
        }
    }
    runtime_graph.validate_against_graph(graph)?;
    Ok(runtime_graph)
}

fn apply_runtime_graph_state_policy(
    component: &mut VulkanResidentPackageComponentCircuit,
    runtime_graph: &StreamCircuitRuntimeGraph,
    instance: &crate::stream_circuit::StreamCircuitNodeInstance,
) {
    let (mode, mut source_instance_id) = match &instance.state_policy {
        StreamCircuitNodeInstanceStatePolicy::Fresh => return,
        StreamCircuitNodeInstanceStatePolicy::CloneFrom { instance_id } => {
            ("clone_from", instance_id.as_str())
        }
        StreamCircuitNodeInstanceStatePolicy::ShareWith { instance_id } => {
            ("shared_from", instance_id.as_str())
        }
    };
    if mode == "clone_from" {
        loop {
            let source = runtime_graph
                .instances
                .iter()
                .find(|candidate| candidate.instance_id == source_instance_id)
                .expect("validated state source instance must exist");
            match &source.state_policy {
                StreamCircuitNodeInstanceStatePolicy::ShareWith { instance_id } => {
                    source_instance_id = instance_id;
                }
                StreamCircuitNodeInstanceStatePolicy::Fresh
                | StreamCircuitNodeInstanceStatePolicy::CloneFrom { .. } => break,
            }
        }
    }
    let source_prefix = format!("{mode}:{source_instance_id}.");
    for state in &mut component.state.state_ports {
        state.sharing = Some(format!("{source_prefix}{}", state.id));
    }
    for state in &mut component.circuit.state_ports {
        state.sharing = Some(format!("{source_prefix}{}", state.id));
    }
}

