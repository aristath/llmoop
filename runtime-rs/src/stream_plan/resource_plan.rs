#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlannedParameterResource {
    pub tensor: String,
    pub uses: Vec<PlannedParameterUse>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlannedParameterUse {
    pub pedal_id: String,
    pub circuit_id: String,
    pub param_id: String,
    pub role: Option<String>,
    pub layout: String,
    pub storage: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlannedStateResource {
    pub pedal_id: String,
    pub circuit_id: String,
    pub state_id: String,
    pub state_type: String,
    pub shape: Option<Vec<usize>>,
    pub elements_per_activation: Option<usize>,
    pub max_dynamic_activations: Option<usize>,
    pub update: Option<String>,
    pub growth: Option<String>,
    pub sharing: Option<String>,
    pub owner: Option<String>,
    pub layout: Option<String>,
    pub source_layout: Option<String>,
    pub element_bytes: Option<usize>,
}

fn state_dtype_bytes(dtype: &str) -> Result<usize, CircuitPlanError> {
    match dtype {
        "BF16" | "F16" => Ok(2),
        "F32" => Ok(4),
        unsupported => Err(CircuitPlanError(format!(
            "unsupported circuit state dtype {unsupported:?}"
        ))),
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlannedActivationSlotBank {
    pub pedal_id: String,
    pub circuit_id: String,
    pub temporary_signal_count: usize,
    pub state_view_signal_count: usize,
    pub slot_count: usize,
    pub slots: Vec<PlannedActivationSlot>,
    pub assignments: Vec<SignalSlotAssignment>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlannedActivationSlot {
    pub slot: usize,
    pub signal_ids: Vec<String>,
    pub max_elements: Option<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CircuitActivationPlan {
    pub pedal_id: String,
    pub circuit_id: String,
    pub input_ports: Vec<PlannedPort>,
    pub output_ports: Vec<PlannedPort>,
    pub state_ports: Vec<PlannedStatePort>,
    pub parameter_refs: Vec<String>,
    pub nodes: Vec<PlannedNode>,
    pub signals: BTreeMap<String, PlannedSignal>,
    pub temporary_signals: Vec<String>,
    pub state_view_signals: Vec<String>,
}

impl CircuitActivationPlan {
    pub fn from_artifact(artifact: &ResolvedCircuitArtifact) -> Result<Self, CircuitPlanError> {
        Self::from_circuit(&artifact.pedal.id, &artifact.circuit)
    }

    pub fn from_artifact_with_tensor_index(
        artifact: &ResolvedCircuitArtifact,
        tensor_index: &TensorIndex,
    ) -> Result<Self, CircuitPlanError> {
        Self::from_circuit_with_tensor_index(&artifact.pedal.id, &artifact.circuit, tensor_index)
    }

    pub fn from_circuit(
        pedal_id: impl Into<String>,
        circuit: &StreamCircuit,
    ) -> Result<Self, CircuitPlanError> {
        Self::from_circuit_with_optional_tensor_index(pedal_id, circuit, None)
    }

    pub fn from_circuit_with_tensor_index(
        pedal_id: impl Into<String>,
        circuit: &StreamCircuit,
        tensor_index: &TensorIndex,
    ) -> Result<Self, CircuitPlanError> {
        Self::from_circuit_with_optional_tensor_index(pedal_id, circuit, Some(tensor_index))
    }

    fn from_circuit_with_optional_tensor_index(
        pedal_id: impl Into<String>,
        circuit: &StreamCircuit,
        tensor_index: Option<&TensorIndex>,
    ) -> Result<Self, CircuitPlanError> {
        let pedal_id = pedal_id.into();
        let state_ids: BTreeSet<_> = circuit.state_ports.iter().map(|state| &state.id).collect();
        let param_ids: BTreeSet<_> = circuit.parameters.refs.keys().collect();
        let boundary_output_sources: BTreeSet<_> = circuit
            .boundary
            .outputs
            .iter()
            .map(|port| port.source.as_ref().unwrap_or(&port.id).clone())
            .collect();

        let mut available = BTreeSet::new();
        let mut signals = BTreeMap::new();
        for input in &circuit.boundary.inputs {
            available.insert(input.id.clone());
            signals.insert(
                input.id.clone(),
                PlannedSignal {
                    id: input.id.clone(),
                    producer: SignalProducer::BoundaryInput,
                    consumers: Vec::new(),
                    shape: Some(input.shape.clone()),
                    storage: SignalStorage::Boundary,
                    is_boundary_output: false,
                },
            );
        }
        for state in &circuit.state_ports {
            available.insert(state.id.clone());
            signals.insert(
                state.id.clone(),
                PlannedSignal {
                    id: state.id.clone(),
                    producer: SignalProducer::StatePort,
                    consumers: Vec::new(),
                    shape: state.shape.clone(),
                    storage: SignalStorage::State,
                    is_boundary_output: false,
                },
            );
        }

        let mut planned_nodes = Vec::with_capacity(circuit.nodes.len());
        for (index, node) in circuit.nodes.iter().enumerate() {
            validate_node_dependencies(&pedal_id, node, &available, &state_ids, &param_ids)?;
            let output_shapes = infer_node_output_shapes(
                &pedal_id,
                node,
                &signals,
                &circuit.parameters.refs,
                tensor_index,
            )?;

            for input in &node.inputs {
                let signal = signals.get_mut(input).ok_or_else(|| {
                    CircuitPlanError(format!(
                        "{} node {} input {:?} is not in the planned signal table",
                        pedal_id, node.id, input
                    ))
                })?;
                signal.consumers.push(node.id.clone());
            }

            for (output_index, output) in node.outputs.iter().enumerate() {
                if available.contains(output) {
                    return Err(CircuitPlanError(format!(
                        "{} node {} output {:?} is already available",
                        pedal_id, node.id, output
                    )));
                }
                available.insert(output.clone());
                signals.insert(
                    output.clone(),
                    PlannedSignal {
                        id: output.clone(),
                        producer: SignalProducer::Node {
                            node_id: node.id.clone(),
                        },
                        consumers: Vec::new(),
                        shape: output_shapes.get(output_index).cloned().unwrap_or(None),
                        storage: node_output_storage(node),
                        is_boundary_output: boundary_output_sources.contains(output),
                    },
                );
            }

            planned_nodes.push(PlannedNode::from_node(index, node));
        }

        for output in &circuit.boundary.outputs {
            let source = output.source.as_ref().unwrap_or(&output.id);
            let signal = signals.get_mut(source).ok_or_else(|| {
                CircuitPlanError(format!(
                    "{} boundary output {} source {:?} is not planned",
                    pedal_id, output.id, source
                ))
            })?;
            signal.is_boundary_output = true;
            signal
                .consumers
                .push(format!("boundary.output:{}", output.id));
        }

        let temporary_signals = signals
            .values()
            .filter(|signal| {
                matches!(signal.producer, SignalProducer::Node { .. })
                    && signal.storage == SignalStorage::Activation
                    && !signal.is_boundary_output
            })
            .map(|signal| signal.id.clone())
            .collect();
        let state_view_signals = signals
            .values()
            .filter(|signal| {
                matches!(signal.producer, SignalProducer::Node { .. })
                    && signal.storage == SignalStorage::StateView
            })
            .map(|signal| signal.id.clone())
            .collect();

        Ok(Self {
            pedal_id,
            circuit_id: circuit.id.clone(),
            input_ports: circuit
                .boundary
                .inputs
                .iter()
                .map(PlannedPort::from_port)
                .collect(),
            output_ports: circuit
                .boundary
                .outputs
                .iter()
                .map(PlannedPort::from_port)
                .collect(),
            state_ports: circuit
                .state_ports
                .iter()
                .map(PlannedStatePort::from_state_port)
                .collect(),
            parameter_refs: circuit.parameters.refs.keys().cloned().collect(),
            nodes: planned_nodes,
            signals,
            temporary_signals,
            state_view_signals,
        })
    }

    pub fn produced_signal_count(&self) -> usize {
        self.signals
            .values()
            .filter(|signal| matches!(signal.producer, SignalProducer::Node { .. }))
            .count()
    }

    pub fn signal(&self, signal_id: &str) -> Option<&PlannedSignal> {
        self.signals.get(signal_id)
    }

    pub fn activation_frame_plan(&self) -> ActivationFramePlan {
        let liveness = self.signal_liveness();
        let mut slot_free_after: Vec<usize> = Vec::new();
        let mut assignments = Vec::with_capacity(liveness.len());

        for live in &liveness {
            let reusable_slot = slot_free_after
                .iter()
                .position(|free_after| *free_after < live.produced_at);
            let slot = if let Some(slot) = reusable_slot {
                slot_free_after[slot] = live.last_consumed_at;
                slot
            } else {
                let slot = slot_free_after.len();
                slot_free_after.push(live.last_consumed_at);
                slot
            };
            assignments.push(SignalSlotAssignment {
                signal_id: live.signal_id.clone(),
                slot,
                produced_at: live.produced_at,
                last_consumed_at: live.last_consumed_at,
            });
        }

        ActivationFramePlan {
            liveness,
            assignments,
            slot_count: slot_free_after.len(),
        }
    }

    fn signal_liveness(&self) -> Vec<SignalLiveness> {
        let temporary_signals: BTreeSet<_> = self.temporary_signals.iter().cloned().collect();
        let node_indices: BTreeMap<_, _> = self
            .nodes
            .iter()
            .map(|node| (node.id.as_str(), node.index))
            .collect();
        let mut liveness = Vec::new();

        for node in &self.nodes {
            for output in &node.outputs {
                if !temporary_signals.contains(output) {
                    continue;
                }
                let signal = self
                    .signals
                    .get(output)
                    .expect("temporary signal is in the planned signal table");
                let consumer_indices: Vec<_> = signal
                    .consumers
                    .iter()
                    .map(|consumer| {
                        if consumer.starts_with("boundary.output:") {
                            self.nodes.len()
                        } else {
                            *node_indices.get(consumer.as_str()).unwrap_or_else(|| {
                                panic!("unknown consumer {consumer:?} for signal {output:?}")
                            })
                        }
                    })
                    .collect();
                let last_consumed_at = consumer_indices.iter().copied().max().unwrap_or(node.index);
                liveness.push(SignalLiveness {
                    signal_id: output.clone(),
                    produced_by: node.id.clone(),
                    produced_at: node.index,
                    consumers: signal.consumers.clone(),
                    consumer_indices,
                    last_consumed_at,
                });
            }
        }

        liveness
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlannedPort {
    pub id: String,
    pub signal: String,
    pub shape: Vec<usize>,
    pub source: Option<String>,
}

impl PlannedPort {
    fn from_port(port: &CircuitPort) -> Self {
        Self {
            id: port.id.clone(),
            signal: port.signal.clone(),
            shape: port.shape.clone(),
            source: port.source.clone(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlannedStatePort {
    pub id: String,
    pub state_type: String,
    pub shape: Option<Vec<usize>>,
    pub elements_per_activation: Option<usize>,
    pub max_dynamic_activations: Option<usize>,
}

impl PlannedStatePort {
    fn from_state_port(state: &StatePort) -> Self {
        Self {
            id: state.id.clone(),
            state_type: state.state_type.clone(),
            shape: state.shape.clone(),
            elements_per_activation: state.elements_per_activation(),
            max_dynamic_activations: state.max_dynamic_activations,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlannedNode {
    pub index: usize,
    pub id: String,
    pub op: String,
    pub specialization: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub params: Vec<String>,
    pub state_reads: Vec<String>,
    pub state_writes: Vec<String>,
}

impl PlannedNode {
    fn from_node(index: usize, node: &CircuitNode) -> Self {
        Self {
            index,
            id: node.id.clone(),
            op: node.op.clone(),
            specialization: if node.attrs.is_null()
                || node
                    .attrs
                    .as_object()
                    .is_some_and(serde_json::Map::is_empty)
            {
                String::new()
            } else {
                serde_json::to_string(&node.attrs)
                    .expect("circuit node attributes must serialize as JSON")
            },
            inputs: node.inputs.clone(),
            outputs: node.outputs.clone(),
            params: node.params.clone(),
            state_reads: node.state_reads.clone(),
            state_writes: node.state_writes.clone(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlannedSignal {
    pub id: String,
    pub producer: SignalProducer,
    pub consumers: Vec<String>,
    pub shape: Option<Vec<usize>>,
    pub storage: SignalStorage,
    pub is_boundary_output: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SignalProducer {
    BoundaryInput,
    StatePort,
    Node { node_id: String },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SignalStorage {
    Boundary,
    State,
    Activation,
    StateView,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ActivationFramePlan {
    pub liveness: Vec<SignalLiveness>,
    pub assignments: Vec<SignalSlotAssignment>,
    pub slot_count: usize,
}

impl ActivationFramePlan {
    pub fn slot_for(&self, signal_id: &str) -> Option<usize> {
        self.assignments
            .iter()
            .find(|assignment| assignment.signal_id == signal_id)
            .map(|assignment| assignment.slot)
    }

    pub fn liveness_for(&self, signal_id: &str) -> Option<&SignalLiveness> {
        self.liveness
            .iter()
            .find(|liveness| liveness.signal_id == signal_id)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SignalLiveness {
    pub signal_id: String,
    pub produced_by: String,
    pub produced_at: usize,
    pub consumers: Vec<String>,
    pub consumer_indices: Vec<usize>,
    pub last_consumed_at: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SignalSlotAssignment {
    pub signal_id: String,
    pub slot: usize,
    pub produced_at: usize,
    pub last_consumed_at: usize,
}

