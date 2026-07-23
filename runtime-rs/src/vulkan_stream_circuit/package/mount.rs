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
        validate_pedal_executions_against_graph(
            &manifest.package_id,
            &manifest.pedal_executions,
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

    pub fn mount_runtime_patch_controls(
        self,
        default_device_id: Option<&str>,
        pedal_devices: &BTreeMap<String, String>,
        duplicate_after: &[(String, String)],
        source_chain: Option<&[(String, String)]>,
    ) -> Result<VulkanResidentRuntimeModel, VulkanResidentTokenModelPackageError> {
        let patch = self.runtime_patch_from_controls(
            default_device_id,
            pedal_devices,
            duplicate_after,
            source_chain,
        )?;

        self.mount_runtime_patch(&patch)
    }

    pub fn runtime_patch_from_controls(
        &self,
        default_device_id: Option<&str>,
        pedal_devices: &BTreeMap<String, String>,
        duplicate_after: &[(String, String)],
        source_chain: Option<&[(String, String)]>,
    ) -> Result<StreamCircuitRuntimePatch, VulkanResidentTokenModelPackageError> {
        let source_graph = self
            .circuit_graph
            .to_resolved_lowered_pedalboard(PathBuf::from("."))?;
        let default_device_id = default_device_id
            .unwrap_or(RUNTIME_DEFAULT_LOGICAL_DEVICE_ID)
            .to_string();
        let mut patch =
            StreamCircuitRuntimePatch::from_source_series(&source_graph, default_device_id)
                .map_err(|error| VulkanResidentTokenModelPackageError::new(error.to_string()))?;
        if let Some(source_chain) = source_chain {
            patch = patch
                .with_signal_processor_chain(&source_graph, source_chain)
                .map_err(|error| VulkanResidentTokenModelPackageError::new(error.to_string()))?;
        }
        for (after_instance_id, new_instance_id) in duplicate_after {
            patch = patch
                .duplicate_after_instance(&source_graph, after_instance_id, new_instance_id)
                .map_err(|error| VulkanResidentTokenModelPackageError::new(error.to_string()))?;
        }
        for (instance_id, device_id) in pedal_devices {
            let instance = patch
                .instances
                .iter()
                .find(|instance| instance.instance_id == *instance_id)
                .ok_or_else(|| {
                    VulkanResidentTokenModelPackageError::new(format!(
                        "runtime patch has no pedal instance {instance_id:?}"
                    ))
                })?;
            let source = source_graph
                .circuits
                .iter()
                .find(|artifact| artifact.pedal.id == instance.source_pedal_id)
                .expect("validated runtime patch source must exist");
            if !source.pedal.runtime_role.is_signal_processor() {
                return Err(VulkanResidentTokenModelPackageError::new(format!(
                    "pedal {instance_id:?} is attached to the processor boundary and cannot be placed independently by the Vulkan backend"
                )));
            }
            patch = patch
                .with_instance_device(instance_id, device_id)
                .map_err(|error| VulkanResidentTokenModelPackageError::new(error.to_string()))?;
        }
        attach_generation_pedal_devices_for_vulkan(patch, &source_graph)
            .map_err(|error| VulkanResidentTokenModelPackageError::new(error.to_string()))
    }

    pub fn resolved_source_graph(
        &self,
        package_root: impl Into<PathBuf>,
    ) -> Result<ResolvedLoweredPedalboard, VulkanResidentTokenModelPackageError> {
        self.circuit_graph
            .to_resolved_lowered_pedalboard(package_root)
    }

    pub fn mount_runtime_patch(
        self,
        patch: &StreamCircuitRuntimePatch,
    ) -> Result<VulkanResidentRuntimeModel, VulkanResidentTokenModelPackageError> {
        let source_graph = self
            .circuit_graph
            .to_resolved_lowered_pedalboard(PathBuf::from("."))?;
        let patch = attach_generation_pedal_devices_for_vulkan(patch.clone(), &source_graph)
            .map_err(|error| VulkanResidentTokenModelPackageError::new(error.to_string()))?;
        patch
            .validate_against_graph(&source_graph)
            .map_err(|error| VulkanResidentTokenModelPackageError::new(error.to_string()))?;

        let source_pedals = self
            .circuit_graph
            .pedals
            .iter()
            .map(|pedal| (pedal.pedal_id.as_str(), pedal))
            .collect::<BTreeMap<_, _>>();
        let source_executions = self
            .pedal_executions
            .iter()
            .map(|execution| (execution.pedal_id.as_str(), execution))
            .collect::<BTreeMap<_, _>>();

        let ordered_instance_ids = patch
            .topological_instance_ids(&source_graph)
            .map_err(|error| VulkanResidentTokenModelPackageError::new(error.to_string()))?;
        let enabled_instance_count = ordered_instance_ids.len();
        let mut pedals = Vec::with_capacity(enabled_instance_count);
        let mut pedal_executions = Vec::with_capacity(enabled_instance_count);
        let mut placement = StreamCircuitPlacementSpec::new(patch.default_device_id.clone());

        for instance_id in ordered_instance_ids {
            let instance = patch
                .instances
                .iter()
                .find(|instance| instance.instance_id == instance_id)
                .expect("validated topological instance id must exist");
            let source_pedal = source_pedals
                .get(instance.source_pedal_id.as_str())
                .ok_or_else(|| {
                    VulkanResidentTokenModelPackageError::new(format!(
                        "runtime patch instance {} references unknown source pedal {}",
                        instance.instance_id, instance.source_pedal_id
                    ))
                })?;
            let mut pedal = (*source_pedal).clone();
            pedal.pedal_id = instance.instance_id.clone();
            pedal.circuit.source.pedal_id = instance.instance_id.clone();
            apply_runtime_patch_state_policy(&mut pedal, &patch, instance);
            pedals.push(pedal);

            if source_pedal.runtime_role.is_signal_processor() {
                let source_execution = source_executions
                    .get(instance.source_pedal_id.as_str())
                    .ok_or_else(|| {
                        VulkanResidentTokenModelPackageError::new(format!(
                            "runtime patch signal processor {} has no execution spec",
                            instance.source_pedal_id
                        ))
                    })?;
                let mut execution = (*source_execution).clone();
                execution.pedal_id = instance.instance_id.clone();
                pedal_executions.push(execution);
            }

            if instance.device_id != patch.default_device_id {
                placement = placement.with_pedal_device(&instance.instance_id, &instance.device_id);
            }
        }

        let mut circuit_graph = self.circuit_graph.clone();
        circuit_graph.wiring = patch.wiring.clone();
        circuit_graph.cables = patch
            .effective_cables()
            .map_err(|error| VulkanResidentTokenModelPackageError::new(error.to_string()))?;
        circuit_graph.boundary = patch.boundary.clone();
        circuit_graph.pedals = pedals;
        validate_generation_execution_contract(&self, &circuit_graph)?;
        Ok(VulkanResidentRuntimeModel {
            package: self,
            patch,
            placement,
            circuit_graph,
            pedal_executions,
        })
    }
}

impl VulkanResidentRuntimeModel {
    pub fn placement_device_ids(&self) -> Vec<String> {
        self.circuit_graph
            .pedals
            .iter()
            .map(|pedal| self.placement.device_for_pedal(&pedal.pedal_id).to_string())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }

    pub fn resolved_graph(
        &self,
        package_root: impl Into<PathBuf>,
    ) -> Result<ResolvedLoweredPedalboard, VulkanResidentTokenModelPackageError> {
        self.circuit_graph
            .to_resolved_lowered_pedalboard(package_root)
    }

    pub fn coalesce_placement_to_device(mut self, device_id: impl Into<String>) -> Self {
        self.placement = StreamCircuitPlacementSpec::new(device_id);
        self
    }
}

pub(crate) fn attach_generation_pedal_devices_for_vulkan(
    mut patch: StreamCircuitRuntimePatch,
    graph: &ResolvedLoweredPedalboard,
) -> Result<StreamCircuitRuntimePatch, crate::stream_circuit::CircuitPlacementError> {
    patch.validate_against_graph(graph)?;
    let source_by_id = graph
        .circuits
        .iter()
        .map(|artifact| (artifact.pedal.id.as_str(), artifact))
        .collect::<BTreeMap<_, _>>();
    let role_for = |instance: &crate::stream_circuit::StreamCircuitPedalInstance| {
        source_by_id[instance.source_pedal_id.as_str()]
            .pedal
            .runtime_role
    };
    let instances_with_role = |role: CircuitRuntimeRole| {
        patch
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
    let processor_ids = patch
        .instances
        .iter()
        .filter(|instance| role_for(instance).is_signal_processor())
        .map(|instance| instance.instance_id.as_str())
        .collect::<BTreeSet<_>>();
    let input_edges = patch
        .cables
        .iter()
        .filter(|cable| {
            cable.connection.is_forward()
                && cable.source.pedal_id == input_transducer_id
                && processor_ids.contains(cable.destination.pedal_id.as_str())
        })
        .collect::<Vec<_>>();
    let output_edges = patch
        .cables
        .iter()
        .filter(|cable| {
            cable.connection.is_forward()
                && processor_ids.contains(cable.source.pedal_id.as_str())
                && cable.destination.pedal_id == output_transducer_id
        })
        .collect::<Vec<_>>();
    let ([input_edge], [output_edge]) = (input_edges.as_slice(), output_edges.as_slice()) else {
        return Err(crate::stream_circuit::CircuitPlacementError(
            "Vulkan generation placement requires one processor input and output boundary"
                .to_string(),
        ));
    };
    let device_by_instance = patch
        .instances
        .iter()
        .map(|instance| (instance.instance_id.as_str(), instance.device_id.clone()))
        .collect::<BTreeMap<_, _>>();
    let input_device = device_by_instance[input_edge.destination.pedal_id.as_str()].clone();
    let output_device = device_by_instance[output_edge.source.pedal_id.as_str()].clone();
    for instance in &mut patch.instances {
        if instance.instance_id == input_transducer_id {
            instance.device_id = input_device.clone();
        } else if instance.instance_id == output_transducer_id || instance.instance_id == sampler_id
        {
            instance.device_id = output_device.clone();
        }
    }
    patch.validate_against_graph(graph)?;
    Ok(patch)
}

fn apply_runtime_patch_state_policy(
    pedal: &mut VulkanResidentPackagePedalCircuit,
    patch: &StreamCircuitRuntimePatch,
    instance: &crate::stream_circuit::StreamCircuitPedalInstance,
) {
    let (mode, mut source_instance_id) = match &instance.state_policy {
        StreamCircuitPedalInstanceStatePolicy::Fresh => return,
        StreamCircuitPedalInstanceStatePolicy::CloneFrom { instance_id } => {
            ("clone_from", instance_id.as_str())
        }
        StreamCircuitPedalInstanceStatePolicy::ShareWith { instance_id } => {
            ("shared_from", instance_id.as_str())
        }
    };
    if mode == "clone_from" {
        loop {
            let source = patch
                .instances
                .iter()
                .find(|candidate| candidate.instance_id == source_instance_id)
                .expect("validated state source instance must exist");
            match &source.state_policy {
                StreamCircuitPedalInstanceStatePolicy::ShareWith { instance_id } => {
                    source_instance_id = instance_id;
                }
                StreamCircuitPedalInstanceStatePolicy::Fresh
                | StreamCircuitPedalInstanceStatePolicy::CloneFrom { .. } => break,
            }
        }
    }
    let source_prefix = format!("{mode}:{source_instance_id}.");
    for state in &mut pedal.state.state_ports {
        state.sharing = Some(format!("{source_prefix}{}", state.id));
    }
    for state in &mut pedal.circuit.state_ports {
        state.sharing = Some(format!("{source_prefix}{}", state.id));
    }
}

