use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::fs;
use std::path::Path;

use serde::Deserialize;

use crate::stream_circuit::{
    CircuitNode, CircuitPort, ParameterRef, ResolvedCircuitArtifact, ResolvedLoweredPedalboard,
    StatePort, StreamCircuit,
};

pub const TENSOR_INDEX_SCHEMA: &str = "llmoop.tensor_index.v1";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CircuitPlanError(pub String);

impl Display for CircuitPlanError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for CircuitPlanError {}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
pub struct TensorIndex {
    pub schema: String,
    #[serde(default)]
    pub tensors: BTreeMap<String, TensorMetadata>,
}

impl TensorIndex {
    pub fn from_json_file(path: impl AsRef<Path>) -> Result<Self, CircuitPlanError> {
        let bytes = fs::read(path).map_err(|error| CircuitPlanError(error.to_string()))?;
        let index: Self =
            serde_json::from_slice(&bytes).map_err(|error| CircuitPlanError(error.to_string()))?;
        if index.schema != TENSOR_INDEX_SCHEMA {
            return Err(CircuitPlanError(format!(
                "unsupported tensor index schema {:?}",
                index.schema
            )));
        }
        Ok(index)
    }

    pub fn tensor_shape(&self, tensor: &str) -> Option<&[usize]> {
        self.tensors
            .get(tensor)
            .map(|metadata| metadata.shape.as_slice())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
pub struct TensorMetadata {
    pub dtype: String,
    pub shape: Vec<usize>,
    #[serde(default)]
    pub parameter_count: Option<usize>,
    #[serde(default)]
    pub byte_count: Option<usize>,
    #[serde(default)]
    pub source_file: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StreamCircuitExecutionPlan {
    pub wiring: String,
    pub circuits: Vec<CircuitActivationPlan>,
}

impl StreamCircuitExecutionPlan {
    pub fn from_graph(graph: &ResolvedLoweredPedalboard) -> Result<Self, CircuitPlanError> {
        if graph.index.graph.wiring != "series" {
            return Err(CircuitPlanError(format!(
                "only series wiring can be planned currently, got {:?}",
                graph.index.graph.wiring
            )));
        }
        let mut circuits = Vec::with_capacity(graph.circuits.len());
        for artifact in &graph.circuits {
            circuits.push(CircuitActivationPlan::from_artifact(artifact)?);
        }
        Ok(Self {
            wiring: graph.index.graph.wiring.clone(),
            circuits,
        })
    }

    pub fn from_graph_with_tensor_index(
        graph: &ResolvedLoweredPedalboard,
        tensor_index: &TensorIndex,
    ) -> Result<Self, CircuitPlanError> {
        if graph.index.graph.wiring != "series" {
            return Err(CircuitPlanError(format!(
                "only series wiring can be planned currently, got {:?}",
                graph.index.graph.wiring
            )));
        }
        let mut circuits = Vec::with_capacity(graph.circuits.len());
        for artifact in &graph.circuits {
            circuits.push(CircuitActivationPlan::from_artifact_with_tensor_index(
                artifact,
                tensor_index,
            )?);
        }
        Ok(Self {
            wiring: graph.index.graph.wiring.clone(),
            circuits,
        })
    }

    pub fn total_node_count(&self) -> usize {
        self.circuits
            .iter()
            .map(|circuit| circuit.nodes.len())
            .sum()
    }

    pub fn temporary_signal_count(&self) -> usize {
        self.circuits
            .iter()
            .map(|circuit| circuit.temporary_signals.len())
            .sum()
    }

    pub fn state_view_signal_count(&self) -> usize {
        self.circuits
            .iter()
            .map(|circuit| circuit.state_view_signals.len())
            .sum()
    }

    pub fn produced_signal_count(&self) -> usize {
        self.circuits
            .iter()
            .map(|circuit| circuit.produced_signal_count())
            .sum()
    }

    pub fn operator_counts(&self) -> BTreeMap<String, usize> {
        let mut counts = BTreeMap::new();
        for circuit in &self.circuits {
            for node in &circuit.nodes {
                *counts.entry(node.op.clone()).or_insert(0) += 1;
            }
        }
        counts
    }

    pub fn state_type_counts(&self) -> BTreeMap<String, usize> {
        let mut counts = BTreeMap::new();
        for circuit in &self.circuits {
            for state in &circuit.state_ports {
                *counts.entry(state.state_type.clone()).or_insert(0) += 1;
            }
        }
        counts
    }

    pub fn layer_local_activation_slot_count(&self) -> usize {
        self.circuits
            .iter()
            .map(|circuit| circuit.activation_frame_plan().slot_count)
            .sum()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StreamCircuitResourcePlan {
    pub circuit_count: usize,
    pub node_count: usize,
    pub parameter_ref_count: usize,
    pub parameters: Vec<PlannedParameterResource>,
    pub state_allocations: Vec<PlannedStateResource>,
    pub activation_banks: Vec<PlannedActivationSlotBank>,
    pub temporary_signal_count: usize,
    pub state_view_signal_count: usize,
    pub layer_local_activation_slot_count: usize,
    pub unknown_temporary_shape_count: usize,
    pub unknown_state_view_shape_count: usize,
}

impl StreamCircuitResourcePlan {
    pub fn from_graph(graph: &ResolvedLoweredPedalboard) -> Result<Self, CircuitPlanError> {
        let execution_plan = StreamCircuitExecutionPlan::from_graph(graph)?;
        Self::from_graph_and_plan(graph, &execution_plan)
    }

    pub fn from_graph_with_tensor_index(
        graph: &ResolvedLoweredPedalboard,
        tensor_index: &TensorIndex,
    ) -> Result<Self, CircuitPlanError> {
        let execution_plan =
            StreamCircuitExecutionPlan::from_graph_with_tensor_index(graph, tensor_index)?;
        Self::from_graph_and_plan(graph, &execution_plan)
    }

    pub fn from_graph_and_plan(
        graph: &ResolvedLoweredPedalboard,
        execution_plan: &StreamCircuitExecutionPlan,
    ) -> Result<Self, CircuitPlanError> {
        if graph.circuits.len() != execution_plan.circuits.len() {
            return Err(CircuitPlanError(format!(
                "graph circuit count {} does not match plan circuit count {}",
                graph.circuits.len(),
                execution_plan.circuits.len()
            )));
        }

        let mut parameter_ref_count = 0;
        let mut parameters_by_tensor: BTreeMap<String, Vec<PlannedParameterUse>> = BTreeMap::new();
        let mut state_allocations = Vec::new();
        let mut activation_banks = Vec::new();
        let mut unknown_temporary_shape_count = 0;
        let mut unknown_state_view_shape_count = 0;

        for (artifact, activation_plan) in graph.circuits.iter().zip(&execution_plan.circuits) {
            if artifact.pedal.id != activation_plan.pedal_id {
                return Err(CircuitPlanError(format!(
                    "graph pedal {:?} does not match activation plan pedal {:?}",
                    artifact.pedal.id, activation_plan.pedal_id
                )));
            }
            if artifact.circuit.id != activation_plan.circuit_id {
                return Err(CircuitPlanError(format!(
                    "graph circuit {:?} does not match activation plan circuit {:?}",
                    artifact.circuit.id, activation_plan.circuit_id
                )));
            }

            for (param_id, parameter) in &artifact.params.refs {
                parameter_ref_count += 1;
                let tensor = parameter.tensor.clone().ok_or_else(|| {
                    CircuitPlanError(format!(
                        "{} parameter {:?} has no source tensor",
                        artifact.pedal.id, param_id
                    ))
                })?;
                parameters_by_tensor
                    .entry(tensor)
                    .or_default()
                    .push(PlannedParameterUse {
                        pedal_id: artifact.pedal.id.clone(),
                        circuit_id: artifact.circuit.id.clone(),
                        param_id: param_id.clone(),
                        role: parameter.role.clone(),
                        layout: artifact.params.layout.clone(),
                        storage: artifact.params.storage.clone(),
                    });
            }

            for state in &artifact.state.state_ports {
                state_allocations.push(PlannedStateResource {
                    pedal_id: artifact.pedal.id.clone(),
                    circuit_id: artifact.circuit.id.clone(),
                    state_id: state.id.clone(),
                    state_type: state.state_type.clone(),
                    shape: state.shape.clone(),
                    elements_per_activation: state.elements_per_activation(),
                    update: state.update.clone(),
                    growth: state.growth.clone(),
                    sharing: state.sharing.clone(),
                    owner: state.owner.clone(),
                    layout: state.layout.clone(),
                    source_layout: state.source_layout.clone(),
                });
            }

            unknown_temporary_shape_count += activation_plan
                .temporary_signals
                .iter()
                .filter(|signal_id| {
                    activation_plan
                        .signal(signal_id)
                        .and_then(|signal| signal.shape.as_ref())
                        .is_none()
                })
                .count();
            unknown_state_view_shape_count += activation_plan
                .state_view_signals
                .iter()
                .filter(|signal_id| {
                    activation_plan
                        .signal(signal_id)
                        .and_then(|signal| signal.shape.as_ref())
                        .is_none()
                })
                .count();

            let activation_frame = activation_plan.activation_frame_plan();
            activation_banks.push(PlannedActivationSlotBank {
                pedal_id: artifact.pedal.id.clone(),
                circuit_id: artifact.circuit.id.clone(),
                temporary_signal_count: activation_plan.temporary_signals.len(),
                slot_count: activation_frame.slot_count,
                assignments: activation_frame.assignments,
            });
        }

        let parameters = parameters_by_tensor
            .into_iter()
            .map(|(tensor, uses)| PlannedParameterResource { tensor, uses })
            .collect();

        Ok(Self {
            circuit_count: graph.circuits.len(),
            node_count: execution_plan.total_node_count(),
            parameter_ref_count,
            parameters,
            state_allocations,
            activation_banks,
            temporary_signal_count: execution_plan.temporary_signal_count(),
            state_view_signal_count: execution_plan.state_view_signal_count(),
            layer_local_activation_slot_count: execution_plan.layer_local_activation_slot_count(),
            unknown_temporary_shape_count,
            unknown_state_view_shape_count,
        })
    }

    pub fn unique_parameter_tensor_count(&self) -> usize {
        self.parameters.len()
    }

    pub fn stream_state_count(&self) -> usize {
        self.state_allocations.len()
    }

    pub fn intermediate_activation_shapes_known(&self) -> bool {
        self.unknown_temporary_shape_count == 0
    }
}

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
    pub update: Option<String>,
    pub growth: Option<String>,
    pub sharing: Option<String>,
    pub owner: Option<String>,
    pub layout: Option<String>,
    pub source_layout: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlannedActivationSlotBank {
    pub pedal_id: String,
    pub circuit_id: String,
    pub temporary_signal_count: usize,
    pub slot_count: usize,
    pub assignments: Vec<SignalSlotAssignment>,
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
}

impl PlannedStatePort {
    fn from_state_port(state: &StatePort) -> Self {
        Self {
            id: state.id.clone(),
            state_type: state.state_type.clone(),
            shape: state.shape.clone(),
            elements_per_activation: state.elements_per_activation(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlannedNode {
    pub index: usize,
    pub id: String,
    pub op: String,
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

fn infer_node_output_shapes(
    pedal_id: &str,
    node: &CircuitNode,
    signals: &BTreeMap<String, PlannedSignal>,
    params: &BTreeMap<String, ParameterRef>,
    tensor_index: Option<&TensorIndex>,
) -> Result<Vec<Option<Vec<usize>>>, CircuitPlanError> {
    let outputs = node.outputs.len();
    let unknown = || Ok(vec![None; outputs]);

    match node.op.as_str() {
        "rms_norm" | "rms_norm_per_head" | "silu" | "rotary_position_embedding" => {
            Ok(repeat_shape(first_input_shape(node, signals), outputs))
        }
        "multiply" | "residual_add" => Ok(repeat_shape(
            compatible_input_shape(pedal_id, node, signals)?,
            outputs,
        )),
        "linear" => infer_linear_output_shapes(pedal_id, node, signals, params, tensor_index),
        "split" => infer_split_output_shapes(pedal_id, node, signals),
        "rolling_state_update" => {
            let state_shape = node
                .inputs
                .get(1)
                .and_then(|input| signals.get(input))
                .and_then(|signal| signal.shape.clone());
            Ok(repeat_shape(state_shape, outputs))
        }
        "depthwise_conv1d" => {
            let output_shape = attr_usize(node, "groups")
                .map(|groups| vec![groups])
                .or_else(|| {
                    first_input_shape(node, signals)
                        .and_then(|shape| shape.last().copied().map(|last| vec![last]))
                });
            Ok(repeat_shape(output_shape, outputs))
        }
        "append_state_update" => unknown(),
        "scaled_dot_product_attention" => {
            Ok(repeat_shape(first_input_shape(node, signals), outputs))
        }
        _ => unknown(),
    }
}

fn infer_linear_output_shapes(
    pedal_id: &str,
    node: &CircuitNode,
    signals: &BTreeMap<String, PlannedSignal>,
    params: &BTreeMap<String, ParameterRef>,
    tensor_index: Option<&TensorIndex>,
) -> Result<Vec<Option<Vec<usize>>>, CircuitPlanError> {
    let Some(tensor_index) = tensor_index else {
        return Ok(vec![None; node.outputs.len()]);
    };
    let Some(param_id) = node.params.first() else {
        return Ok(vec![None; node.outputs.len()]);
    };
    let Some(parameter) = params.get(param_id) else {
        return Ok(vec![None; node.outputs.len()]);
    };
    let Some(tensor) = parameter.tensor.as_deref() else {
        return Ok(vec![None; node.outputs.len()]);
    };
    let Some(weight_shape) = tensor_index.tensor_shape(tensor) else {
        return Ok(vec![None; node.outputs.len()]);
    };
    if weight_shape.len() != 2 {
        return Ok(vec![None; node.outputs.len()]);
    }

    let output_width = weight_shape[0];
    let input_width = weight_shape[1];
    let output_shape = match first_input_shape(node, signals) {
        Some(mut input_shape) => {
            let Some(last_dim) = input_shape.last_mut() else {
                return Ok(vec![None; node.outputs.len()]);
            };
            if *last_dim != input_width {
                return Err(CircuitPlanError(format!(
                    "{} node {} linear input width {} does not match parameter {:?} width {}",
                    pedal_id, node.id, *last_dim, param_id, input_width
                )));
            }
            *last_dim = output_width;
            Some(input_shape)
        }
        None => Some(vec![output_width]),
    };

    Ok(repeat_shape(output_shape, node.outputs.len()))
}

fn infer_split_output_shapes(
    pedal_id: &str,
    node: &CircuitNode,
    signals: &BTreeMap<String, PlannedSignal>,
) -> Result<Vec<Option<Vec<usize>>>, CircuitPlanError> {
    let Some(mut input_shape) = first_input_shape(node, signals) else {
        return Ok(vec![None; node.outputs.len()]);
    };
    let Some(channel_dim) = input_shape.last_mut() else {
        return Ok(vec![None; node.outputs.len()]);
    };
    if node.outputs.is_empty() || *channel_dim % node.outputs.len() != 0 {
        return Err(CircuitPlanError(format!(
            "{} node {} cannot split shape {:?} across {} outputs",
            pedal_id,
            node.id,
            first_input_shape(node, signals),
            node.outputs.len()
        )));
    }
    *channel_dim /= node.outputs.len();
    Ok(repeat_shape(Some(input_shape), node.outputs.len()))
}

fn compatible_input_shape(
    pedal_id: &str,
    node: &CircuitNode,
    signals: &BTreeMap<String, PlannedSignal>,
) -> Result<Option<Vec<usize>>, CircuitPlanError> {
    let mut known_shape = None;
    for input in &node.inputs {
        let shape = signals.get(input).and_then(|signal| signal.shape.clone());
        if let Some(shape) = shape {
            if let Some(existing) = &known_shape {
                if existing != &shape {
                    return Err(CircuitPlanError(format!(
                        "{} node {} input {:?} shape {:?} does not match {:?}",
                        pedal_id, node.id, input, shape, existing
                    )));
                }
            } else {
                known_shape = Some(shape);
            }
        }
    }
    Ok(known_shape)
}

fn first_input_shape(
    node: &CircuitNode,
    signals: &BTreeMap<String, PlannedSignal>,
) -> Option<Vec<usize>> {
    node.inputs
        .first()
        .and_then(|input| signals.get(input))
        .and_then(|signal| signal.shape.clone())
}

fn repeat_shape(shape: Option<Vec<usize>>, count: usize) -> Vec<Option<Vec<usize>>> {
    (0..count).map(|_| shape.clone()).collect()
}

fn attr_usize(node: &CircuitNode, attr: &str) -> Option<usize> {
    node.attrs
        .get(attr)
        .and_then(|value| value.as_u64())
        .and_then(|value| usize::try_from(value).ok())
}

fn node_output_storage(node: &CircuitNode) -> SignalStorage {
    match node.op.as_str() {
        "append_state_update" | "rolling_state_update" => SignalStorage::StateView,
        _ => SignalStorage::Activation,
    }
}

fn validate_node_dependencies(
    pedal_id: &str,
    node: &CircuitNode,
    available: &BTreeSet<String>,
    state_ids: &BTreeSet<&String>,
    param_ids: &BTreeSet<&String>,
) -> Result<(), CircuitPlanError> {
    for input in &node.inputs {
        if !available.contains(input) {
            return Err(CircuitPlanError(format!(
                "{} node {} input {:?} is not available at its schedule position",
                pedal_id, node.id, input
            )));
        }
    }
    for param in &node.params {
        if !param_ids.contains(param) {
            return Err(CircuitPlanError(format!(
                "{} node {} parameter {:?} is not declared",
                pedal_id, node.id, param
            )));
        }
    }
    for state in node.state_reads.iter().chain(node.state_writes.iter()) {
        if !state_ids.contains(state) {
            return Err(CircuitPlanError(format!(
                "{} node {} state {:?} is not declared",
                pedal_id, node.id, state
            )));
        }
    }
    if node.outputs.is_empty() {
        return Err(CircuitPlanError(format!(
            "{} node {} has no outputs",
            pedal_id, node.id
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use super::*;
    use crate::stream_circuit::ResolvedLoweredPedalboard;

    fn lfm2_index_path() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("lowered")
            .join("lfm2_5_230m")
            .join("pedalboard.circuits.json")
    }

    fn lfm2_tensor_index_path() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("transpiled")
            .join("lfm2_5_230m")
            .join("tensors.json")
    }

    #[test]
    fn plans_lfm2_lowered_pedalboard_activation_schedule() {
        let graph = ResolvedLoweredPedalboard::from_index_file(lfm2_index_path()).unwrap();

        let plan = StreamCircuitExecutionPlan::from_graph(&graph).unwrap();

        assert_eq!(plan.wiring, "series");
        assert_eq!(plan.circuits.len(), 14);
        assert_eq!(plan.total_node_count(), 242);
        assert_eq!(plan.produced_signal_count(), 264);
        assert_eq!(plan.temporary_signal_count(), 230);
        assert_eq!(plan.state_view_signal_count(), 20);
        assert_eq!(plan.layer_local_activation_slot_count(), 56);
        assert_eq!(plan.operator_counts().get("linear"), Some(&82));
        assert_eq!(
            plan.state_type_counts().get("append_only_attention_memory"),
            Some(&6)
        );

        let layer_00 = &plan.circuits[0];
        let layer_00_frame = layer_00.activation_frame_plan();
        assert_eq!(layer_00.pedal_id, "layer_00");
        assert_eq!(layer_00.nodes.len(), 16);
        assert_eq!(layer_00.temporary_signals.len(), 16);
        assert_eq!(
            layer_00.state_view_signals,
            vec!["temporal_window".to_string()]
        );
        assert_eq!(layer_00_frame.slot_count, 4);
        assert_eq!(layer_00.input_ports[0].id, "input_frame");
        assert_eq!(layer_00.output_ports[0].id, "output_frame");
        assert_eq!(
            layer_00
                .nodes
                .iter()
                .find(|node| node.id == "temporal_memory_update")
                .unwrap()
                .state_writes,
            vec!["temporal_memory".to_string()]
        );

        let layer_02 = &plan.circuits[2];
        assert_eq!(layer_02.temporary_signals.len(), 17);
        assert_eq!(
            layer_02.state_view_signals,
            vec!["k_memory".to_string(), "v_memory".to_string()]
        );
        assert_eq!(layer_02.activation_frame_plan().slot_count, 4);
        assert!(
            layer_02
                .nodes
                .iter()
                .any(|node| node.op == "append_state_update")
        );
        assert!(
            layer_02
                .nodes
                .iter()
                .any(|node| node.op == "scaled_dot_product_attention")
        );
    }

    #[test]
    fn tensor_index_enables_lfm2_signal_shape_planning() {
        let graph = ResolvedLoweredPedalboard::from_index_file(lfm2_index_path()).unwrap();
        let tensor_index = TensorIndex::from_json_file(lfm2_tensor_index_path()).unwrap();

        let plan = StreamCircuitExecutionPlan::from_graph_with_tensor_index(&graph, &tensor_index)
            .unwrap();
        let resource_plan = StreamCircuitResourcePlan::from_graph_and_plan(&graph, &plan).unwrap();

        assert_eq!(tensor_index.schema, TENSOR_INDEX_SCHEMA);
        assert_eq!(resource_plan.temporary_signal_count, 230);
        assert_eq!(resource_plan.state_view_signal_count, 20);
        assert_eq!(resource_plan.unknown_temporary_shape_count, 0);
        assert_eq!(resource_plan.unknown_state_view_shape_count, 12);
        assert!(resource_plan.intermediate_activation_shapes_known());

        let layer_00 = &plan.circuits[0];
        assert_eq!(
            layer_00.signal("conv_projected").unwrap().shape,
            Some(vec![3072])
        );
        assert_eq!(layer_00.signal("gate_b").unwrap().shape, Some(vec![1024]));
        assert_eq!(
            layer_00.signal("temporal_window").unwrap().shape,
            Some(vec![3, 1024])
        );
        assert_eq!(
            layer_00.signal("ffn_hidden").unwrap().shape,
            Some(vec![2560])
        );

        let layer_02 = &plan.circuits[2];
        assert_eq!(
            layer_02.signal("q_projected").unwrap().shape,
            Some(vec![1024])
        );
        assert_eq!(
            layer_02.signal("k_projected").unwrap().shape,
            Some(vec![512])
        );
        assert_eq!(layer_02.signal("k_memory").unwrap().shape, None);
        assert_eq!(
            layer_02.signal("k_memory").unwrap().storage,
            SignalStorage::StateView
        );
        assert_eq!(layer_02.signal("v_memory").unwrap().shape, None);
        assert_eq!(
            layer_02.signal("v_memory").unwrap().storage,
            SignalStorage::StateView
        );
        assert_eq!(
            layer_02.signal("attention_out").unwrap().shape,
            Some(vec![1024])
        );
    }

    #[test]
    fn resource_plan_names_lfm2_mount_resources() {
        let graph = ResolvedLoweredPedalboard::from_index_file(lfm2_index_path()).unwrap();
        let execution_plan = StreamCircuitExecutionPlan::from_graph(&graph).unwrap();

        let resource_plan =
            StreamCircuitResourcePlan::from_graph_and_plan(&graph, &execution_plan).unwrap();

        assert_eq!(resource_plan.circuit_count, 14);
        assert_eq!(resource_plan.node_count, 242);
        assert_eq!(resource_plan.parameter_ref_count, 130);
        assert_eq!(resource_plan.unique_parameter_tensor_count(), 130);
        assert_eq!(resource_plan.stream_state_count(), 14);
        assert_eq!(resource_plan.temporary_signal_count, 230);
        assert_eq!(resource_plan.state_view_signal_count, 20);
        assert_eq!(resource_plan.layer_local_activation_slot_count, 56);
        assert_eq!(resource_plan.unknown_temporary_shape_count, 172);
        assert_eq!(resource_plan.unknown_state_view_shape_count, 12);
        assert!(!resource_plan.intermediate_activation_shapes_known());

        let conv_in = resource_plan
            .parameters
            .iter()
            .find(|parameter| parameter.tensor == "model.layers.0.conv.in_proj.weight")
            .unwrap();
        assert_eq!(conv_in.uses.len(), 1);
        assert_eq!(conv_in.uses[0].pedal_id, "layer_00");
        assert_eq!(conv_in.uses[0].param_id, "conv_in_projection");
        assert_eq!(
            conv_in.uses[0].role.as_deref(),
            Some("short_convolution_input_projection")
        );
        assert_eq!(conv_in.uses[0].storage, "source_tensor_refs");

        let rolling_states = resource_plan
            .state_allocations
            .iter()
            .filter(|state| state.state_type == "rolling_frame_memory")
            .count();
        let append_only_states = resource_plan
            .state_allocations
            .iter()
            .filter(|state| state.state_type == "append_only_attention_memory")
            .count();
        assert_eq!(rolling_states, 8);
        assert_eq!(append_only_states, 6);

        let layer_00_state = resource_plan
            .state_allocations
            .iter()
            .find(|state| state.pedal_id == "layer_00")
            .unwrap();
        assert_eq!(layer_00_state.state_id, "temporal_memory");
        assert_eq!(layer_00_state.shape, Some(vec![3, 1024]));
        assert_eq!(layer_00_state.elements_per_activation, None);
        assert_eq!(layer_00_state.layout.as_deref(), Some("time_hidden"));

        let layer_02_state = resource_plan
            .state_allocations
            .iter()
            .find(|state| state.pedal_id == "layer_02")
            .unwrap();
        assert_eq!(layer_02_state.state_id, "kv_memory");
        assert_eq!(layer_02_state.shape, None);
        assert_eq!(layer_02_state.elements_per_activation, Some(1024));
        assert_eq!(layer_02_state.layout.as_deref(), Some("append_only_kv"));

        let layer_00_bank = resource_plan
            .activation_banks
            .iter()
            .find(|bank| bank.pedal_id == "layer_00")
            .unwrap();
        assert_eq!(layer_00_bank.temporary_signal_count, 16);
        assert_eq!(layer_00_bank.slot_count, 4);
        assert_eq!(layer_00_bank.assignments.len(), 16);

        let layer_02_bank = resource_plan
            .activation_banks
            .iter()
            .find(|bank| bank.pedal_id == "layer_02")
            .unwrap();
        assert_eq!(layer_02_bank.temporary_signal_count, 17);
        assert_eq!(layer_02_bank.slot_count, 4);
    }

    #[test]
    fn resource_plan_rejects_mismatched_execution_plan() {
        let graph = ResolvedLoweredPedalboard::from_index_file(lfm2_index_path()).unwrap();
        let mut execution_plan = StreamCircuitExecutionPlan::from_graph(&graph).unwrap();
        execution_plan.circuits.pop();

        let error =
            StreamCircuitResourcePlan::from_graph_and_plan(&graph, &execution_plan).unwrap_err();

        assert!(error.to_string().contains("graph circuit count 14"));
    }

    #[test]
    fn activation_plan_tracks_signal_producers_and_consumers() {
        let graph = ResolvedLoweredPedalboard::from_index_file(lfm2_index_path()).unwrap();
        let plan = StreamCircuitExecutionPlan::from_graph(&graph).unwrap();
        let layer_00 = &plan.circuits[0];

        let input_frame = layer_00.signal("input_frame").unwrap();
        assert_eq!(input_frame.producer, SignalProducer::BoundaryInput);
        assert_eq!(
            input_frame.consumers,
            vec!["operator_norm".to_string(), "operator_residual".to_string()]
        );

        let norm_out = layer_00.signal("operator_norm_out").unwrap();
        assert_eq!(
            norm_out.producer,
            SignalProducer::Node {
                node_id: "operator_norm".to_string()
            }
        );
        assert_eq!(norm_out.consumers, vec!["conv_in_projection".to_string()]);

        let output_frame = layer_00.signal("output_frame").unwrap();
        assert!(output_frame.is_boundary_output);
        assert_eq!(
            output_frame.consumers,
            vec!["boundary.output:output_frame".to_string()]
        );

        let temporal_window = layer_00.signal("temporal_window").unwrap();
        assert_eq!(temporal_window.storage, SignalStorage::StateView);
        assert_eq!(
            temporal_window.consumers,
            vec!["depthwise_temporal_conv".to_string()]
        );
    }

    #[test]
    fn activation_frame_plan_reuses_temporary_signal_slots_by_liveness() {
        let graph = ResolvedLoweredPedalboard::from_index_file(lfm2_index_path()).unwrap();
        let plan = StreamCircuitExecutionPlan::from_graph(&graph).unwrap();
        let frame = plan.circuits[0].activation_frame_plan();

        assert_eq!(frame.liveness.len(), 16);
        assert_eq!(frame.slot_count, 4);
        assert_eq!(frame.slot_for("operator_norm_out"), Some(0));
        assert_eq!(frame.slot_for("gate_b"), Some(0));
        assert_eq!(frame.slot_for("gate_c"), Some(2));
        assert_eq!(frame.slot_for("projected_x"), Some(3));
        assert_eq!(frame.slot_for("operator_residual_out"), Some(1));
        assert_eq!(frame.slot_for("temporal_window"), None);

        let residual = frame.liveness_for("operator_residual_out").unwrap();
        assert_eq!(residual.produced_by, "operator_residual");
        assert_eq!(residual.produced_at, 8);
        assert_eq!(residual.last_consumed_at, 15);
        assert_eq!(
            residual.consumers,
            vec!["ffn_norm".to_string(), "ffn_residual".to_string()]
        );
    }

    #[test]
    fn activation_plan_rejects_unscheduled_signal_dependency() {
        let graph = ResolvedLoweredPedalboard::from_index_file(lfm2_index_path()).unwrap();
        let mut circuit = graph.circuits[0].circuit.clone();
        circuit.nodes[0].inputs = vec!["not_available_yet".to_string()];

        let error = CircuitActivationPlan::from_circuit("layer_00", &circuit).unwrap_err();

        assert!(
            error
                .to_string()
                .contains("input \"not_available_yet\" is not available")
        );
    }
}
