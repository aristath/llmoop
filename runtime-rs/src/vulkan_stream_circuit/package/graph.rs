#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VulkanResidentPackageCircuitGraph {
    pub wiring: String,
    pub cables: Vec<crate::stream_circuit::StreamCircuitGraphCable>,
    pub boundary: StreamCircuitGraphBoundary,
    #[serde(default)]
    pub architecture: Value,
    #[serde(default)]
    pub dimensions: Value,
    #[serde(default)]
    pub input_transducer: Value,
    #[serde(default)]
    pub output_transducer: Value,
    #[serde(default)]
    pub pedals: Vec<VulkanResidentPackagePedalCircuit>,
}

impl VulkanResidentPackageCircuitGraph {
    fn to_resolved_lowered_pedalboard(
        &self,
        package_root: impl Into<PathBuf>,
    ) -> Result<ResolvedLoweredPedalboard, VulkanResidentTokenModelPackageError> {
        let mut operator_counts = BTreeMap::new();
        let mut circuit_refs = Vec::with_capacity(self.pedals.len());
        let mut circuits = Vec::with_capacity(self.pedals.len());

        for pedal in &self.pedals {
            *operator_counts
                .entry(pedal.operator_type.clone())
                .or_insert(0) += 1;
            let circuit_ref = LoweredCircuitRef {
                id: pedal.pedal_id.clone(),
                operator_type: pedal.operator_type.clone(),
                runtime_role: pedal.runtime_role,
                circuit: format!("package://{}/circuit", pedal.pedal_id),
                params: format!("package://{}/params", pedal.pedal_id),
                state: format!("package://{}/state", pedal.pedal_id),
                implementation: pedal.implementation.clone(),
                behavioral_role: pedal.behavioral_role.clone(),
            };
            let resolved = ResolvedCircuitArtifact {
                pedal: circuit_ref.clone(),
                circuit: pedal.circuit.clone(),
                params: pedal.params.clone(),
                state: pedal.state.clone(),
            };
            resolved.validate().map_err(|error| {
                VulkanResidentTokenModelPackageError::new(format!(
                    "resident package circuit graph pedal {} is invalid: {error}",
                    pedal.pedal_id
                ))
            })?;
            circuit_refs.push(circuit_ref);
            circuits.push(resolved);
        }

        let index = LoweredPedalboard {
            schema: LOWERED_PEDALBOARD_SCHEMA.to_string(),
            source: LoweredPedalboardSource {
                format: VULKAN_RESIDENT_MODEL_PACKAGE_MANIFEST_SCHEMA.to_string(),
                artifact_root: "package".to_string(),
            },
            architecture: self.architecture.clone(),
            dimensions: self.dimensions.clone(),
            graph: LoweredPedalboardGraph {
                wiring: self.wiring.clone(),
                circuits: circuit_refs,
                cables: self.cables.clone(),
                boundary: self.boundary.clone(),
                input_transducer: self.input_transducer.clone(),
                output_transducer: self.output_transducer.clone(),
            },
            summary: LoweredPedalboardSummary {
                circuit_count: self.pedals.len(),
                operator_counts,
            },
            notes: vec!["resolved from resident model package manifest".to_string()],
        };
        index.validate_index().map_err(|error| {
            VulkanResidentTokenModelPackageError::new(format!(
                "resident package circuit graph is invalid: {error}"
            ))
        })?;

        Ok(ResolvedLoweredPedalboard {
            artifact_root: package_root.into(),
            index,
            circuits,
        })
    }

    pub(super) fn to_signal_processor_graph(
        &self,
        package_root: impl Into<PathBuf>,
    ) -> Result<ResolvedLoweredPedalboard, VulkanResidentTokenModelPackageError> {
        let full = self.to_resolved_lowered_pedalboard(package_root)?;
        let processor_ids = full
            .circuits
            .iter()
            .filter(|artifact| artifact.circuit.runtime_role.is_signal_processor())
            .map(|artifact| artifact.pedal.id.as_str())
            .collect::<BTreeSet<_>>();
        if processor_ids.is_empty() {
            return Err(VulkanResidentTokenModelPackageError::new(
                "resident package pedalboard contains no signal processors",
            ));
        }
        let circuits = full
            .circuits
            .iter()
            .filter(|artifact| processor_ids.contains(artifact.pedal.id.as_str()))
            .cloned()
            .collect::<Vec<_>>();
        let circuit_refs = circuits
            .iter()
            .map(|artifact| artifact.pedal.clone())
            .collect::<Vec<_>>();
        let cables = full
            .index
            .graph
            .cables
            .iter()
            .filter(|cable| {
                cable.connection.is_forward()
                    && processor_ids.contains(cable.source.pedal_id.as_str())
                    && processor_ids.contains(cable.destination.pedal_id.as_str())
            })
            .cloned()
            .collect::<Vec<_>>();
        let external_inputs = execution_boundary_inputs(&full, &processor_ids);
        let public_outputs = execution_boundary_outputs(&full, &processor_ids);
        let mut operator_counts = BTreeMap::new();
        for artifact in &circuits {
            *operator_counts
                .entry(artifact.pedal.operator_type.clone())
                .or_insert(0) += 1;
        }
        let mut index = full.index.clone();
        index.graph.circuits = circuit_refs;
        index.graph.cables = cables;
        index.graph.boundary = StreamCircuitGraphBoundary {
            external_inputs,
            public_outputs,
        };
        index.summary = LoweredPedalboardSummary {
            circuit_count: circuits.len(),
            operator_counts,
        };
        index.validate_index().map_err(|error| {
            VulkanResidentTokenModelPackageError::new(format!(
                "resident package signal-processor graph is invalid: {error}"
            ))
        })?;
        Ok(ResolvedLoweredPedalboard {
            artifact_root: full.artifact_root,
            index,
            circuits,
        })
    }

    pub(super) fn signal_processor_placement(
        &self,
        placement: &StreamCircuitPlacementSpec,
    ) -> StreamCircuitPlacementSpec {
        let processor_ids = self
            .pedals
            .iter()
            .filter(|pedal| pedal.runtime_role.is_signal_processor())
            .map(|pedal| pedal.pedal_id.as_str())
            .collect::<BTreeSet<_>>();
        StreamCircuitPlacementSpec {
            schema: placement.schema.clone(),
            default_device_id: placement.default_device_id.clone(),
            pedal_devices: placement
                .pedal_devices
                .iter()
                .filter(|(pedal_id, _)| processor_ids.contains(pedal_id.as_str()))
                .map(|(pedal_id, device_id)| (pedal_id.clone(), device_id.clone()))
                .collect(),
        }
    }

    pub(super) fn signal_processor_endpoint_pedal_ids(
        &self,
    ) -> Result<(String, String), VulkanResidentTokenModelPackageError> {
        let graph = self.to_signal_processor_graph(PathBuf::from("."))?;
        let [input] = graph.index.graph.boundary.external_inputs.as_slice() else {
            return Err(VulkanResidentTokenModelPackageError::new(format!(
                "Vulkan generation currently requires exactly one signal-processor input boundary, found {}",
                graph.index.graph.boundary.external_inputs.len()
            )));
        };
        let [output] = graph.index.graph.boundary.public_outputs.as_slice() else {
            return Err(VulkanResidentTokenModelPackageError::new(format!(
                "Vulkan generation currently requires exactly one signal-processor output boundary, found {}",
                graph.index.graph.boundary.public_outputs.len()
            )));
        };
        Ok((
            input.endpoint.pedal_id.clone(),
            output.endpoint.pedal_id.clone(),
        ))
    }

    pub(super) fn signal_processor_device_ids(
        &self,
        placement: &StreamCircuitPlacementSpec,
    ) -> Vec<String> {
        self.pedals
            .iter()
            .filter(|pedal| pedal.runtime_role.is_signal_processor())
            .map(|pedal| placement.device_for_pedal(&pedal.pedal_id).to_string())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }
}

fn execution_boundary_inputs(
    graph: &ResolvedLoweredPedalboard,
    processor_ids: &BTreeSet<&str>,
) -> Vec<crate::stream_circuit::StreamCircuitGraphBoundaryPort> {
    let mut ports = graph
        .index
        .graph
        .boundary
        .external_inputs
        .iter()
        .filter(|port| processor_ids.contains(port.endpoint.pedal_id.as_str()))
        .cloned()
        .collect::<Vec<_>>();
    ports.extend(
        graph
            .index
            .graph
            .cables
            .iter()
            .filter(|cable| {
                cable.connection.is_forward()
                    && !processor_ids.contains(cable.source.pedal_id.as_str())
                    && processor_ids.contains(cable.destination.pedal_id.as_str())
            })
            .map(
                |cable| crate::stream_circuit::StreamCircuitGraphBoundaryPort {
                    id: format!("{}_input", cable.id),
                    endpoint: cable.destination.clone(),
                },
            ),
    );
    ports
}

fn execution_boundary_outputs(
    graph: &ResolvedLoweredPedalboard,
    processor_ids: &BTreeSet<&str>,
) -> Vec<crate::stream_circuit::StreamCircuitGraphBoundaryPort> {
    let mut ports = graph
        .index
        .graph
        .boundary
        .public_outputs
        .iter()
        .filter(|port| processor_ids.contains(port.endpoint.pedal_id.as_str()))
        .cloned()
        .collect::<Vec<_>>();
    ports.extend(
        graph
            .index
            .graph
            .cables
            .iter()
            .filter(|cable| {
                cable.connection.is_forward()
                    && processor_ids.contains(cable.source.pedal_id.as_str())
                    && !processor_ids.contains(cable.destination.pedal_id.as_str())
            })
            .map(
                |cable| crate::stream_circuit::StreamCircuitGraphBoundaryPort {
                    id: format!("{}_output", cable.id),
                    endpoint: cable.source.clone(),
                },
            ),
    );
    ports
}

