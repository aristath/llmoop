#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StreamCircuitRuntimePatch {
    pub schema: String,
    pub wiring: String,
    pub default_device_id: String,
    pub instances: Vec<StreamCircuitPedalInstance>,
    pub cables: Vec<StreamCircuitGraphCable>,
    pub boundary: StreamCircuitGraphBoundary,
}

impl StreamCircuitRuntimePatch {
    pub fn from_source_series(
        graph: &ResolvedLoweredPedalboard,
        default_device_id: impl Into<String>,
    ) -> Result<Self, CircuitPlacementError> {
        let spec = StreamCircuitPlacementSpec::new(default_device_id);
        Self::from_placement_spec(graph, &spec)
    }

    pub fn from_source_chain(
        graph: &ResolvedLoweredPedalboard,
        default_device_id: impl Into<String>,
        chain: &[(String, String)],
    ) -> Result<Self, CircuitPlacementError> {
        validate_runtime_patch_source_graph(graph)?;
        let instances = chain
            .iter()
            .map(
                |(instance_id, source_pedal_id)| StreamCircuitPedalInstance {
                    instance_id: instance_id.clone(),
                    source_pedal_id: source_pedal_id.clone(),
                    device_id: String::new(),
                    enabled: true,
                    control_values: BTreeMap::new(),
                    state_policy: StreamCircuitPedalInstanceStatePolicy::Fresh,
                },
            )
            .collect::<Vec<_>>();
        let cables = series_cables_for_instances(graph, &instances)?;
        let boundary = series_boundary_for_instances(graph, &instances)?;
        let patch = Self {
            schema: STREAM_CIRCUIT_RUNTIME_PATCH_SCHEMA.to_string(),
            wiring: "explicit_graph".to_string(),
            default_device_id: default_device_id.into(),
            instances,
            cables,
            boundary,
        };
        patch.with_default_devices().and_then(|patch| {
            patch.validate_against_graph(graph)?;
            Ok(patch)
        })
    }

    pub fn from_placement_spec(
        graph: &ResolvedLoweredPedalboard,
        spec: &StreamCircuitPlacementSpec,
    ) -> Result<Self, CircuitPlacementError> {
        validate_runtime_patch_source_graph(graph)?;
        validate_placement_spec_against_graph(graph, spec)?;
        Ok(Self {
            schema: STREAM_CIRCUIT_RUNTIME_PATCH_SCHEMA.to_string(),
            wiring: "explicit_graph".to_string(),
            default_device_id: spec.default_device_id.clone(),
            instances: graph
                .circuits
                .iter()
                .map(|artifact| StreamCircuitPedalInstance {
                    instance_id: artifact.pedal.id.clone(),
                    source_pedal_id: artifact.pedal.id.clone(),
                    device_id: spec.device_for_pedal(&artifact.pedal.id).to_string(),
                    enabled: true,
                    control_values: BTreeMap::new(),
                    state_policy: StreamCircuitPedalInstanceStatePolicy::Fresh,
                })
                .collect(),
            cables: graph.index.graph.cables.clone(),
            boundary: graph.index.graph.boundary.clone(),
        })
    }

    pub fn placement_spec(&self) -> StreamCircuitPlacementSpec {
        let mut spec = StreamCircuitPlacementSpec::new(self.default_device_id.clone());
        for instance in self.instances.iter().filter(|instance| instance.enabled) {
            if instance.device_id != self.default_device_id {
                spec = spec.with_pedal_device(&instance.instance_id, &instance.device_id);
            }
        }
        spec
    }

    pub fn duplicate_after_instance(
        mut self,
        graph: &ResolvedLoweredPedalboard,
        after_instance_id: &str,
        new_instance_id: impl Into<String>,
    ) -> Result<Self, CircuitPlacementError> {
        let new_instance_id = new_instance_id.into();
        if new_instance_id.is_empty() {
            return Err(CircuitPlacementError(
                "runtime patch duplicate instance id must not be empty".to_string(),
            ));
        }
        if self
            .instances
            .iter()
            .any(|instance| instance.instance_id == new_instance_id)
        {
            return Err(CircuitPlacementError(format!(
                "runtime patch already has pedal instance {new_instance_id:?}"
            )));
        }
        let after_index = self
            .instances
            .iter()
            .position(|instance| instance.instance_id == after_instance_id)
            .ok_or_else(|| {
                CircuitPlacementError(format!(
                    "runtime patch has no pedal instance {after_instance_id:?}"
                ))
            })?;
        let source = self.instances[after_index].clone();
        let duplicate = StreamCircuitPedalInstance {
            instance_id: new_instance_id.clone(),
            source_pedal_id: source.source_pedal_id.clone(),
            device_id: source.device_id.clone(),
            enabled: source.enabled,
            control_values: BTreeMap::new(),
            state_policy: StreamCircuitPedalInstanceStatePolicy::Fresh,
        };
        let source_by_id = graph
            .circuits
            .iter()
            .map(|artifact| (artifact.pedal.id.as_str(), artifact))
            .collect::<BTreeMap<_, _>>();
        let source_artifact = source_by_id
            .get(source.source_pedal_id.as_str())
            .ok_or_else(|| {
                CircuitPlacementError(format!(
                    "runtime patch instance {} references unknown source pedal {}",
                    source.instance_id, source.source_pedal_id
                ))
            })?;
        let source_output = single_series_port(
            &source_artifact.circuit.boundary.outputs,
            &source.instance_id,
            "output",
        )?;
        let duplicate_input = single_series_port(
            &source_artifact.circuit.boundary.inputs,
            &new_instance_id,
            "input",
        )?;
        let outgoing = self
            .cables
            .iter()
            .enumerate()
            .filter(|(_, cable)| {
                cable.source.pedal_id == after_instance_id && cable.connection.is_forward()
            })
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        if outgoing.len() > 1 {
            return Err(CircuitPlacementError(format!(
                "cannot insert duplicate after branching pedal {after_instance_id:?}; wire the explicit graph instead"
            )));
        }
        if outgoing
            .first()
            .is_some_and(|index| self.cables[*index].source.port_id != source_output.id)
        {
            return Err(CircuitPlacementError(format!(
                "cannot insert duplicate after pedal {after_instance_id:?}: its outgoing cable does not use the sole series output"
            )));
        }
        let inserted_cable = StreamCircuitGraphCable {
            id: allocate_cable_id(&self.cables, after_instance_id, &new_instance_id),
            source: StreamCircuitCableEndpoint {
                pedal_id: after_instance_id.to_string(),
                port_id: source_output.id.clone(),
            },
            destination: StreamCircuitCableEndpoint {
                pedal_id: new_instance_id.clone(),
                port_id: duplicate_input.id.clone(),
            },
            connection: StreamCircuitConnection::Forward,
        };
        if let Some(outgoing_index) = outgoing.first().copied() {
            self.cables[outgoing_index].source = StreamCircuitCableEndpoint {
                pedal_id: new_instance_id.clone(),
                port_id: source_output.id.clone(),
            };
            self.cables.insert(outgoing_index, inserted_cable);
        } else {
            for output in &mut self.boundary.public_outputs {
                if output.endpoint.pedal_id == source.instance_id
                    && output.endpoint.port_id == source_output.id
                {
                    output.endpoint = StreamCircuitCableEndpoint {
                        pedal_id: new_instance_id.clone(),
                        port_id: source_output.id.clone(),
                    };
                }
            }
            self.cables.push(inserted_cable);
        }
        self.instances.insert(after_index + 1, duplicate);
        Ok(self)
    }

    pub fn with_source_chain(
        self,
        graph: &ResolvedLoweredPedalboard,
        chain: &[(String, String)],
    ) -> Result<Self, CircuitPlacementError> {
        let previous_instances = self
            .instances
            .iter()
            .map(|instance| (instance.instance_id.clone(), instance.clone()))
            .collect::<BTreeMap<_, _>>();
        let mut patch = Self::from_source_chain(graph, self.default_device_id, chain)?;
        for instance in &mut patch.instances {
            if let Some(previous) = previous_instances.get(&instance.instance_id) {
                instance.device_id = previous.device_id.clone();
                instance.enabled = previous.enabled;
                instance.control_values = previous.control_values.clone();
                instance.state_policy = previous.state_policy.clone();
            }
        }
        patch.validate_against_graph(graph)?;
        Ok(patch)
    }

    pub fn with_signal_processor_chain(
        self,
        graph: &ResolvedLoweredPedalboard,
        chain: &[(String, String)],
    ) -> Result<Self, CircuitPlacementError> {
        self.validate_against_graph(graph)?;
        if chain.is_empty() {
            return Err(CircuitPlacementError(
                "signal-processor chain must contain at least one pedal".to_string(),
            ));
        }

        let source_by_id = graph
            .circuits
            .iter()
            .map(|artifact| (artifact.pedal.id.as_str(), artifact))
            .collect::<BTreeMap<_, _>>();
        let old_processor_ids = self
            .instances
            .iter()
            .filter(|instance| {
                source_by_id
                    .get(instance.source_pedal_id.as_str())
                    .is_some_and(|source| source.pedal.runtime_role.is_signal_processor())
            })
            .map(|instance| instance.instance_id.as_str())
            .collect::<BTreeSet<_>>();
        if old_processor_ids.is_empty() {
            return Err(CircuitPlacementError(
                "runtime patch contains no signal-processing pedals to replace".to_string(),
            ));
        }

        let preserved_instance_ids = self
            .instances
            .iter()
            .filter(|instance| !old_processor_ids.contains(instance.instance_id.as_str()))
            .map(|instance| instance.instance_id.as_str())
            .collect::<BTreeSet<_>>();
        let previous_by_id = self
            .instances
            .iter()
            .map(|instance| (instance.instance_id.as_str(), instance))
            .collect::<BTreeMap<_, _>>();
        let mut chain_instance_ids = BTreeSet::new();
        let mut processor_instances = Vec::with_capacity(chain.len());
        for (instance_id, source_pedal_id) in chain {
            if instance_id.is_empty() || !chain_instance_ids.insert(instance_id.as_str()) {
                return Err(CircuitPlacementError(format!(
                    "signal-processor chain contains an empty or duplicate instance id {instance_id:?}"
                )));
            }
            if preserved_instance_ids.contains(instance_id.as_str()) {
                return Err(CircuitPlacementError(format!(
                    "signal-processor instance id {instance_id:?} collides with a non-processor pedal"
                )));
            }
            let source = source_by_id.get(source_pedal_id.as_str()).ok_or_else(|| {
                CircuitPlacementError(format!(
                    "signal-processor chain references unknown source pedal {source_pedal_id:?}"
                ))
            })?;
            if !source.pedal.runtime_role.is_signal_processor() {
                return Err(CircuitPlacementError(format!(
                    "source pedal {source_pedal_id:?} has runtime role {:?}, not signal_processor",
                    source.pedal.runtime_role
                )));
            }
            let mut instance = StreamCircuitPedalInstance {
                instance_id: instance_id.clone(),
                source_pedal_id: source_pedal_id.clone(),
                device_id: self.default_device_id.clone(),
                enabled: true,
                control_values: BTreeMap::new(),
                state_policy: StreamCircuitPedalInstanceStatePolicy::Fresh,
            };
            if let Some(previous) = previous_by_id.get(instance_id.as_str()) {
                instance.device_id = previous.device_id.clone();
                if previous.source_pedal_id == *source_pedal_id {
                    instance.enabled = previous.enabled;
                    instance.control_values = previous.control_values.clone();
                    instance.state_policy = previous.state_policy.clone();
                }
            }
            processor_instances.push(instance);
        }

        if old_processor_ids.len() == self.instances.len() {
            let mut patch = Self::from_source_chain(graph, self.default_device_id.clone(), chain)?;
            patch.instances = processor_instances;
            patch.validate_against_graph(graph)?;
            return Ok(patch);
        }

        let crossing_inputs = self
            .cables
            .iter()
            .filter(|cable| {
                cable.connection.is_forward()
                    && !old_processor_ids.contains(cable.source.pedal_id.as_str())
                    && old_processor_ids.contains(cable.destination.pedal_id.as_str())
            })
            .cloned()
            .collect::<Vec<_>>();
        let crossing_outputs = self
            .cables
            .iter()
            .filter(|cable| {
                cable.connection.is_forward()
                    && old_processor_ids.contains(cable.source.pedal_id.as_str())
                    && !old_processor_ids.contains(cable.destination.pedal_id.as_str())
            })
            .cloned()
            .collect::<Vec<_>>();
        if crossing_inputs.len() != 1 || crossing_outputs.len() != 1 {
            return Err(CircuitPlacementError(format!(
                "signal-processor chain replacement requires one forward input and output cable; found {} inputs and {} outputs",
                crossing_inputs.len(),
                crossing_outputs.len()
            )));
        }
        if self.cables.iter().any(|cable| {
            !cable.connection.is_forward()
                && (old_processor_ids.contains(cable.source.pedal_id.as_str())
                    || old_processor_ids.contains(cable.destination.pedal_id.as_str()))
        }) {
            return Err(CircuitPlacementError(
                "signal-processor chain replacement cannot discard processor-local temporal wiring"
                    .to_string(),
            ));
        }
        if self
            .boundary
            .external_inputs
            .iter()
            .chain(self.boundary.public_outputs.iter())
            .any(|port| old_processor_ids.contains(port.endpoint.pedal_id.as_str()))
        {
            return Err(CircuitPlacementError(
                "signal-processor chain replacement requires graph boundaries outside the processor chain"
                    .to_string(),
            ));
        }

        let first = processor_instances
            .first()
            .expect("non-empty processor chain must have a first instance");
        let last = processor_instances
            .last()
            .expect("non-empty processor chain must have a last instance");
        let first_source = source_by_id[first.source_pedal_id.as_str()];
        let last_source = source_by_id[last.source_pedal_id.as_str()];
        let first_input = single_series_port(
            &first_source.circuit.boundary.inputs,
            &first.instance_id,
            "input",
        )?;
        let last_output = single_series_port(
            &last_source.circuit.boundary.outputs,
            &last.instance_id,
            "output",
        )?;

        let mut instances = Vec::with_capacity(
            self.instances.len() - old_processor_ids.len() + processor_instances.len(),
        );
        let mut inserted = false;
        for instance in &self.instances {
            if old_processor_ids.contains(instance.instance_id.as_str()) {
                if !inserted {
                    instances.extend(processor_instances.iter().cloned());
                    inserted = true;
                }
            } else {
                instances.push(instance.clone());
            }
        }

        let mut cables = self
            .cables
            .iter()
            .filter(|cable| {
                !old_processor_ids.contains(cable.source.pedal_id.as_str())
                    && !old_processor_ids.contains(cable.destination.pedal_id.as_str())
            })
            .cloned()
            .collect::<Vec<_>>();
        let mut input_cable = crossing_inputs[0].clone();
        input_cable.destination = StreamCircuitCableEndpoint {
            pedal_id: first.instance_id.clone(),
            port_id: first_input.id.clone(),
        };
        cables.push(input_cable);
        for pair in processor_instances.windows(2) {
            let source = source_by_id[pair[0].source_pedal_id.as_str()];
            let destination = source_by_id[pair[1].source_pedal_id.as_str()];
            let source_output = single_series_port(
                &source.circuit.boundary.outputs,
                &pair[0].instance_id,
                "output",
            )?;
            let destination_input = single_series_port(
                &destination.circuit.boundary.inputs,
                &pair[1].instance_id,
                "input",
            )?;
            let cable_id = allocate_cable_id(&cables, &pair[0].instance_id, &pair[1].instance_id);
            cables.push(StreamCircuitGraphCable {
                id: cable_id,
                source: StreamCircuitCableEndpoint {
                    pedal_id: pair[0].instance_id.clone(),
                    port_id: source_output.id.clone(),
                },
                destination: StreamCircuitCableEndpoint {
                    pedal_id: pair[1].instance_id.clone(),
                    port_id: destination_input.id.clone(),
                },
                connection: StreamCircuitConnection::Forward,
            });
        }
        let mut output_cable = crossing_outputs[0].clone();
        output_cable.source = StreamCircuitCableEndpoint {
            pedal_id: last.instance_id.clone(),
            port_id: last_output.id.clone(),
        };
        cables.push(output_cable);

        let mut patch = self;
        patch.instances = instances;
        patch.cables = cables;
        patch.validate_against_graph(graph)?;
        Ok(patch)
    }

    pub fn with_instance_device(
        mut self,
        instance_id: &str,
        device_id: impl Into<String>,
    ) -> Result<Self, CircuitPlacementError> {
        let device_id = device_id.into();
        if device_id.is_empty() {
            return Err(CircuitPlacementError(format!(
                "runtime patch device id for instance {instance_id:?} must not be empty"
            )));
        }
        let instance = self
            .instances
            .iter_mut()
            .find(|instance| instance.instance_id == instance_id)
            .ok_or_else(|| {
                CircuitPlacementError(format!(
                    "runtime patch has no pedal instance {instance_id:?}"
                ))
            })?;
        instance.device_id = device_id;
        Ok(self)
    }

    pub fn with_instance_enabled(
        mut self,
        instance_id: &str,
        enabled: bool,
    ) -> Result<Self, CircuitPlacementError> {
        let instance = self
            .instances
            .iter_mut()
            .find(|instance| instance.instance_id == instance_id)
            .ok_or_else(|| {
                CircuitPlacementError(format!(
                    "runtime patch has no pedal instance {instance_id:?}"
                ))
            })?;
        instance.enabled = enabled;
        Ok(self)
    }

    pub fn instantiate_graph(
        &self,
        graph: &ResolvedLoweredPedalboard,
    ) -> Result<ResolvedLoweredPedalboard, CircuitPlacementError> {
        self.validate_against_graph(graph)?;
        let source_by_id = graph
            .circuits
            .iter()
            .map(|artifact| (artifact.pedal.id.as_str(), artifact))
            .collect::<BTreeMap<_, _>>();
        let mut circuits = Vec::with_capacity(self.instances.len());
        let mut circuit_refs = Vec::with_capacity(self.instances.len());
        let mut operator_counts = BTreeMap::new();

        let ordered_instance_ids = self.topological_instance_ids(graph)?;
        for instance_id in ordered_instance_ids {
            let instance = self
                .instances
                .iter()
                .find(|instance| instance.instance_id == instance_id)
                .expect("validated topological instance id must exist");
            let source = source_by_id
                .get(instance.source_pedal_id.as_str())
                .ok_or_else(|| {
                    CircuitPlacementError(format!(
                        "runtime patch instance {} references unknown source pedal {}",
                        instance.instance_id, instance.source_pedal_id
                    ))
                })?;
            let mut resolved = (*source).clone();
            resolved.pedal.id = instance.instance_id.clone();
            circuit_refs.push(resolved.pedal.clone());
            *operator_counts
                .entry(resolved.pedal.operator_type.clone())
                .or_insert(0) += 1;
            circuits.push(resolved);
        }

        let mut index = graph.index.clone();
        index.graph.wiring = self.wiring.clone();
        index.graph.circuits = circuit_refs;
        index.graph.cables = self.effective_cables()?;
        index.graph.boundary = self.boundary.clone();
        index.summary = LoweredPedalboardSummary {
            circuit_count: circuits.len(),
            operator_counts,
        };

        Ok(ResolvedLoweredPedalboard {
            artifact_root: graph.artifact_root.clone(),
            index,
            circuits,
        })
    }

    pub fn validate_against_graph(
        &self,
        graph: &ResolvedLoweredPedalboard,
    ) -> Result<(), CircuitPlacementError> {
        validate_runtime_patch_source_graph(graph)?;
        if self.schema != STREAM_CIRCUIT_RUNTIME_PATCH_SCHEMA {
            return Err(CircuitPlacementError(format!(
                "unsupported runtime patch schema {:?}",
                self.schema
            )));
        }
        if self.wiring != "explicit_graph" {
            return Err(CircuitPlacementError(format!(
                "runtime patch wiring must be explicit_graph, got {:?}",
                self.wiring
            )));
        }
        if self.default_device_id.is_empty() {
            return Err(CircuitPlacementError(
                "runtime patch default_device_id must not be empty".to_string(),
            ));
        }
        if self.instances.is_empty() {
            return Err(CircuitPlacementError(
                "runtime patch must contain at least one pedal instance".to_string(),
            ));
        }
        if !self.instances.iter().any(|instance| instance.enabled) {
            return Err(CircuitPlacementError(
                "runtime patch must contain at least one enabled pedal instance".to_string(),
            ));
        }

        let source_by_id = graph
            .circuits
            .iter()
            .map(|artifact| (artifact.pedal.id.as_str(), artifact))
            .collect::<BTreeMap<_, _>>();
        let mut instance_ids = BTreeSet::new();
        for instance in &self.instances {
            if instance.instance_id.is_empty() {
                return Err(CircuitPlacementError(
                    "runtime patch contains an instance with an empty id".to_string(),
                ));
            }
            if !instance_ids.insert(instance.instance_id.as_str()) {
                return Err(CircuitPlacementError(format!(
                    "runtime patch contains duplicate pedal instance {:?}",
                    instance.instance_id
                )));
            }
            if instance.device_id.is_empty() {
                return Err(CircuitPlacementError(format!(
                    "runtime patch instance {} has an empty device id",
                    instance.instance_id
                )));
            }
            if !source_by_id.contains_key(instance.source_pedal_id.as_str()) {
                return Err(CircuitPlacementError(format!(
                    "runtime patch instance {} references unknown source pedal {}",
                    instance.instance_id, instance.source_pedal_id
                )));
            }
            if !instance.control_values.is_empty() {
                return Err(CircuitPlacementError(format!(
                    "runtime patch instance {} supplies control values, but executable pedal controls are not implemented",
                    instance.instance_id
                )));
            }
            validate_instance_state_policy(instance, &self.instances, &source_by_id)?;
        }
        validate_state_policy_dependencies(&self.instances)?;

        validate_explicit_cables(self, &source_by_id)?;
        self.topological_instance_ids(graph)?;

        Ok(())
    }

    fn with_default_devices(mut self) -> Result<Self, CircuitPlacementError> {
        if self.default_device_id.is_empty() {
            return Err(CircuitPlacementError(
                "runtime patch default_device_id must not be empty".to_string(),
            ));
        }
        for instance in &mut self.instances {
            if instance.device_id.is_empty() {
                instance.device_id = self.default_device_id.clone();
            }
        }
        Ok(self)
    }

    pub fn effective_cables(&self) -> Result<Vec<StreamCircuitGraphCable>, CircuitPlacementError> {
        effective_runtime_patch_cables(&self.instances, &self.cables)
    }

    pub fn topological_instance_ids(
        &self,
        _graph: &ResolvedLoweredPedalboard,
    ) -> Result<Vec<String>, CircuitPlacementError> {
        topological_runtime_patch_order(&self.instances, &self.effective_cables()?)
    }
}
