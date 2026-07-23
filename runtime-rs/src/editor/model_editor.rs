impl RuntimeModelEditor {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, RuntimeEditorError> {
        Self::load_with_device_provider(path, |default_device_id| {
            discover_runtime_devices(default_device_id, None)
        })
    }

    pub fn load_with_available_devices(
        path: impl AsRef<Path>,
        available_devices: Vec<RuntimeAvailableDevice>,
    ) -> Result<Self, RuntimeEditorError> {
        Self::load_with_device_provider(path, |_| available_devices)
    }

    fn load_with_device_provider(
        path: impl AsRef<Path>,
        devices: impl FnOnce(&str) -> Vec<RuntimeAvailableDevice>,
    ) -> Result<Self, RuntimeEditorError> {
        let manifest_path = match classify_runtime_model_path(path)? {
            RuntimeModelPathKind::CompiledPackage { manifest } => manifest,
            RuntimeModelPathKind::SafetensorsSource { .. } => {
                return Err(RuntimeEditorError(
                    "Safetensors sources must be compiled before loading the runtime editor"
                        .to_string(),
                ));
            }
        };
        let package_root = manifest_path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));
        let manifest = VulkanResidentModelPackageManifest::from_json_file(&manifest_path)?;
        let source_graph = manifest
            .resolved_source_graph(package_root.clone())
            .map_err(|error| RuntimeEditorError(error.to_string()))?;
        let draft = manifest
            .runtime_patch_from_controls(None, &BTreeMap::new(), &[], None)
            .map_err(|error| RuntimeEditorError(error.to_string()))?;
        let source_pedals = source_pedals(&manifest);
        let source_by_layer = source_pedals
            .iter()
            .filter_map(|pedal| {
                pedal
                    .layer_index
                    .map(|layer_index| (layer_index, pedal.source_id.clone()))
            })
            .fold(
                BTreeMap::<usize, Vec<String>>::new(),
                |mut by_layer, entry| {
                    by_layer.entry(entry.0).or_default().push(entry.1);
                    by_layer
                },
            );
        let source_ids = source_pedals
            .iter()
            .map(|pedal| pedal.source_id.clone())
            .collect();
        let available_devices = devices(RUNTIME_DEFAULT_LOGICAL_DEVICE_ID);
        Ok(Self {
            package_manifest_path: manifest_path,
            package_root,
            manifest,
            source_graph,
            source_pedals,
            source_by_layer,
            source_ids,
            available_devices,
            draft,
        })
    }

    pub fn package_manifest_path(&self) -> &Path {
        &self.package_manifest_path
    }

    pub fn package_root(&self) -> &Path {
        &self.package_root
    }

    pub fn package_id(&self) -> &str {
        &self.manifest.package_id
    }

    pub fn max_context_activations(&self) -> usize {
        self.manifest.max_context_activations
    }

    pub fn source_pedals(&self) -> &[RuntimeEditorSourcePedal] {
        &self.source_pedals
    }

    pub fn available_devices(&self) -> &[RuntimeAvailableDevice] {
        &self.available_devices
    }

    pub fn refresh_devices(&mut self) {
        self.available_devices = discover_runtime_devices(&self.draft.default_device_id, None);
    }

    pub fn draft(&self) -> &StreamCircuitRuntimePatch {
        &self.draft
    }

    pub fn layer_sequence(&self) -> Vec<usize> {
        let layer_by_source = self
            .source_pedals
            .iter()
            .filter_map(|pedal| {
                pedal
                    .layer_index
                    .map(|layer_index| (pedal.source_id.as_str(), layer_index))
            })
            .collect::<BTreeMap<_, _>>();
        self.draft
            .instances
            .iter()
            .filter_map(|instance| {
                layer_by_source
                    .get(instance.source_pedal_id.as_str())
                    .copied()
            })
            .collect()
    }

    pub fn source_sequence(&self) -> Vec<String> {
        self.draft
            .instances
            .iter()
            .filter(|instance| instance.enabled)
            .map(|instance| instance.source_pedal_id.clone())
            .collect()
    }

    pub fn instances(&self) -> Vec<RuntimeEditorInstance> {
        let layer_by_source = self
            .source_pedals
            .iter()
            .map(|pedal| (pedal.source_id.as_str(), pedal.layer_index))
            .collect::<BTreeMap<_, _>>();
        let mut occurrences = BTreeMap::<&str, usize>::new();
        self.draft
            .instances
            .iter()
            .filter_map(|instance| {
                let layer_index = *layer_by_source.get(instance.source_pedal_id.as_str())?;
                let occurrence = occurrences
                    .entry(instance.source_pedal_id.as_str())
                    .and_modify(|value| *value += 1)
                    .or_insert(1);
                Some(RuntimeEditorInstance {
                    instance_id: instance.instance_id.clone(),
                    source_id: instance.source_pedal_id.clone(),
                    layer_index,
                    occurrence: *occurrence,
                    device_id: instance.device_id.clone(),
                    enabled: instance.enabled,
                    control_values: instance.control_values.clone(),
                    state_policy: instance.state_policy.clone(),
                })
            })
            .collect()
    }

    pub fn replace_layer_sequence(
        &mut self,
        layer_sequence: &[usize],
    ) -> Result<(), RuntimeEditorError> {
        if layer_sequence.is_empty() {
            return Err(RuntimeEditorError(
                "layer sequence must contain at least one layer".to_string(),
            ));
        }
        let source_sequence = layer_sequence
            .iter()
            .map(|layer_index| {
                let sources = self.source_by_layer.get(layer_index).ok_or_else(|| {
                    RuntimeEditorError(format!(
                        "unknown layer {layer_index}; available layers: {}",
                        available_layer_range(&self.source_by_layer)
                    ))
                })?;
                if sources.len() != 1 {
                    return Err(RuntimeEditorError(format!(
                        "layer {layer_index} has {} source pedals; edit the source sequence by id",
                        sources.len()
                    )));
                }
                Ok(sources[0].clone())
            })
            .collect::<Result<Vec<_>, RuntimeEditorError>>()?;
        self.replace_signal_processor_sequence(&source_sequence)
    }

    pub fn replace_signal_processor_sequence(
        &mut self,
        source_sequence: &[String],
    ) -> Result<(), RuntimeEditorError> {
        let processor_instances = self.instances_for_source_sequence(source_sequence)?;
        let chain = processor_instances
            .iter()
            .map(|instance| {
                (
                    instance.instance_id.clone(),
                    instance.source_pedal_id.clone(),
                )
            })
            .collect::<Vec<_>>();
        self.draft = self
            .draft
            .clone()
            .with_signal_processor_chain(&self.source_graph, &chain)?;
        Ok(())
    }

    fn instances_for_source_sequence(
        &self,
        source_sequence: &[String],
    ) -> Result<Vec<StreamCircuitPedalInstance>, RuntimeEditorError> {
        let mut previous_by_source =
            BTreeMap::<String, VecDeque<StreamCircuitPedalInstance>>::new();
        for instance in &self.draft.instances {
            previous_by_source
                .entry(instance.source_pedal_id.clone())
                .or_default()
                .push_back(instance.clone());
        }
        let mut occurrence_by_source = BTreeMap::<String, usize>::new();
        let mut used_instance_ids = BTreeSet::new();
        let mut instances = Vec::with_capacity(source_sequence.len());
        for source_id in source_sequence {
            if !self.source_ids.contains(source_id) {
                return Err(RuntimeEditorError(format!(
                    "unknown source pedal {source_id:?}"
                )));
            }
            let occurrence = occurrence_by_source
                .entry(source_id.clone())
                .and_modify(|value| *value += 1)
                .or_insert(1);
            let previous = previous_by_source
                .get_mut(source_id)
                .and_then(VecDeque::pop_front);
            let instance = if let Some(previous) = previous {
                used_instance_ids.insert(previous.instance_id.clone());
                previous
            } else {
                let instance_id = allocate_instance_id(source_id, *occurrence, &used_instance_ids);
                used_instance_ids.insert(instance_id.clone());
                StreamCircuitPedalInstance {
                    instance_id,
                    source_pedal_id: source_id.clone(),
                    device_id: self.draft.default_device_id.clone(),
                    enabled: true,
                    control_values: BTreeMap::new(),
                    state_policy: StreamCircuitPedalInstanceStatePolicy::Fresh,
                }
            };
            instances.push(instance);
        }
        Ok(instances)
    }

    pub fn set_instance_device(
        &mut self,
        instance_id: &str,
        device_id: &str,
    ) -> Result<(), RuntimeEditorError> {
        let available = self.available_devices.iter().any(|device| {
            device.device_id == device_id
                && device.available
                && device.can_host_runtime_pedals_on_physical_device != Some(false)
        });
        if !available {
            return Err(RuntimeEditorError(format!(
                "runtime device {device_id:?} is unavailable or cannot host this pedal"
            )));
        }
        self.draft = self
            .draft
            .clone()
            .with_instance_device(instance_id, device_id)?;
        Ok(())
    }

    pub fn set_instance_enabled(
        &mut self,
        instance_id: &str,
        enabled: bool,
    ) -> Result<(), RuntimeEditorError> {
        let candidate = self
            .draft
            .clone()
            .with_instance_enabled(instance_id, enabled)?;
        candidate.validate_against_graph(&self.source_graph)?;
        self.draft = candidate;
        Ok(())
    }

    pub fn set_instance_control_value(
        &mut self,
        instance_id: &str,
        control_id: &str,
        value: Value,
    ) -> Result<(), RuntimeEditorError> {
        let source = self.source_pedal_for_instance(instance_id).ok_or_else(|| {
            RuntimeEditorError(format!(
                "runtime patch has no pedal instance {instance_id:?}"
            ))
        })?;
        let schema = source
            .control_schemas
            .iter()
            .find(|schema| schema.id == control_id)
            .ok_or_else(|| {
                RuntimeEditorError(format!(
                    "source pedal {} declares no control {control_id:?}",
                    source.source_id
                ))
            })?;
        validate_runtime_editor_control_value(schema, &value)?;
        let instance = self
            .draft
            .instances
            .iter_mut()
            .find(|instance| instance.instance_id == instance_id)
            .ok_or_else(|| {
                RuntimeEditorError(format!(
                    "runtime patch has no pedal instance {instance_id:?}"
                ))
            })?;
        instance
            .control_values
            .insert(control_id.to_string(), value);
        Ok(())
    }

    pub fn effective_instance_control_value(
        &self,
        instance_id: &str,
        control_id: &str,
    ) -> Option<Value> {
        let instance = self
            .draft
            .instances
            .iter()
            .find(|instance| instance.instance_id == instance_id)?;
        if let Some(value) = instance.control_values.get(control_id) {
            return Some(value.clone());
        }
        self.source_pedal_for_instance(instance_id)?
            .control_schemas
            .iter()
            .find(|schema| schema.id == control_id)
            .and_then(|schema| {
                schema
                    .current_value
                    .clone()
                    .or_else(|| schema.default_value.clone())
            })
    }

    pub fn set_instance_state_policy(
        &mut self,
        instance_id: &str,
        state_policy: StreamCircuitPedalInstanceStatePolicy,
    ) -> Result<(), RuntimeEditorError> {
        let instance = self
            .draft
            .instances
            .iter_mut()
            .find(|instance| instance.instance_id == instance_id)
            .ok_or_else(|| {
                RuntimeEditorError(format!(
                    "runtime patch has no pedal instance {instance_id:?}"
                ))
            })?;
        instance.state_policy = state_policy;
        Ok(())
    }

    pub fn validation(&self) -> RuntimeEditorValidation {
        let mut errors = Vec::new();
        for instance in &self.draft.instances {
            if !self.available_devices.iter().any(|device| {
                device.device_id == instance.device_id
                    && device.available
                    && device.can_host_runtime_pedals_on_physical_device != Some(false)
            }) {
                errors.push(format!(
                    "instance {} is assigned to unavailable device {}",
                    instance.instance_id, instance.device_id
                ));
            }
            if let Some(source) = self
                .source_pedals
                .iter()
                .find(|source| source.source_id == instance.source_pedal_id)
            {
                for (control_id, value) in &instance.control_values {
                    match source
                        .control_schemas
                        .iter()
                        .find(|schema| schema.id == *control_id)
                    {
                        Some(schema) => {
                            if let Err(error) = validate_runtime_editor_control_value(schema, value)
                            {
                                errors.push(format!(
                                    "instance {} control {}: {}",
                                    instance.instance_id, control_id, error
                                ));
                            }
                        }
                        None => errors.push(format!(
                            "instance {} has undeclared control {}",
                            instance.instance_id, control_id
                        )),
                    }
                }
            }
        }
        if let Err(error) = self.draft.validate_against_graph(&self.source_graph) {
            errors.push(error.to_string());
        }
        let placement = if errors.is_empty() {
            self.source_graph
                .instantiate_runtime_patch(&self.draft)
                .and_then(|graph| graph.placement_plan(&self.draft.placement_spec()))
                .map_err(|error| errors.push(error.to_string()))
                .ok()
        } else {
            None
        };
        RuntimeEditorValidation {
            valid: errors.is_empty(),
            errors,
            warnings: Vec::new(),
            placement,
        }
    }

    pub fn source_pedal_for_instance(
        &self,
        instance_id: &str,
    ) -> Option<&RuntimeEditorSourcePedal> {
        let source_id = self
            .draft
            .instances
            .iter()
            .find(|instance| instance.instance_id == instance_id)?
            .source_pedal_id
            .as_str();
        self.source_pedals
            .iter()
            .find(|pedal| pedal.source_id == source_id)
    }
}

#[cfg(test)]
pub(crate) fn load_runtime_model_editor_without_hardware(
    path: impl AsRef<Path>,
) -> Result<RuntimeModelEditor, RuntimeEditorError> {
    let path = path.as_ref();
    let manifest_path = match classify_runtime_model_path(path)? {
        RuntimeModelPathKind::CompiledPackage { manifest } => manifest,
        RuntimeModelPathKind::SafetensorsSource { .. } => {
            return Err(RuntimeEditorError(
                "test editor requires a compiled package".to_string(),
            ));
        }
    };
    let device_id = RUNTIME_DEFAULT_LOGICAL_DEVICE_ID.to_string();
    RuntimeModelEditor::load_with_available_devices(
        manifest_path,
        vec![RuntimeAvailableDevice {
            device_id: device_id.clone(),
            backend: "test".to_string(),
            available: true,
            runtime_device_id: Some(device_id),
            physical_device_id: Some("test:0".to_string()),
            physical_device_index: Some(0),
            device_name: Some("Deterministic test device".to_string()),
            device_type: Some("test".to_string()),
            vendor_id: None,
            raw_device_id: None,
            api_version: None,
            driver_version: None,
            compute_queue_family_indices: Some(vec![0]),
            memory_heaps: Some(Vec::new()),
            selected_by_default: Some(true),
            selected_by_runtime: Some(true),
            runtime_binding: Some("test_only".to_string()),
            can_host_runtime_pedals_on_physical_device: Some(true),
            notes: vec!["hardware discovery disabled for this test".to_string()],
            error: None,
        }],
    )
}
