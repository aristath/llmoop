use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt::{Display, Formatter};

use crate::stream_circuit::{
    CircuitNode, CircuitPort, ResolvedCircuitArtifact, ResolvedLoweredPedalboard, StatePort,
    StreamCircuit,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CircuitPlanError(pub String);

impl Display for CircuitPlanError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for CircuitPlanError {}

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
}

impl CircuitActivationPlan {
    pub fn from_artifact(artifact: &ResolvedCircuitArtifact) -> Result<Self, CircuitPlanError> {
        Self::from_circuit(&artifact.pedal.id, &artifact.circuit)
    }

    pub fn from_circuit(
        pedal_id: impl Into<String>,
        circuit: &StreamCircuit,
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
                    is_boundary_output: false,
                },
            );
        }

        let mut planned_nodes = Vec::with_capacity(circuit.nodes.len());
        for (index, node) in circuit.nodes.iter().enumerate() {
            validate_node_dependencies(&pedal_id, node, &available, &state_ids, &param_ids)?;

            for input in &node.inputs {
                let signal = signals.get_mut(input).ok_or_else(|| {
                    CircuitPlanError(format!(
                        "{} node {} input {:?} is not in the planned signal table",
                        pedal_id, node.id, input
                    ))
                })?;
                signal.consumers.push(node.id.clone());
            }

            for output in &node.outputs {
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
                        shape: None,
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
                matches!(signal.producer, SignalProducer::Node { .. }) && !signal.is_boundary_output
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
    pub is_boundary_output: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SignalProducer {
    BoundaryInput,
    StatePort,
    Node { node_id: String },
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

    #[test]
    fn plans_lfm2_lowered_pedalboard_activation_schedule() {
        let graph = ResolvedLoweredPedalboard::from_index_file(lfm2_index_path()).unwrap();

        let plan = StreamCircuitExecutionPlan::from_graph(&graph).unwrap();

        assert_eq!(plan.wiring, "series");
        assert_eq!(plan.circuits.len(), 14);
        assert_eq!(plan.total_node_count(), 242);
        assert_eq!(plan.produced_signal_count(), 264);
        assert_eq!(plan.temporary_signal_count(), 250);
        assert_eq!(plan.layer_local_activation_slot_count(), 62);
        assert_eq!(plan.operator_counts().get("linear"), Some(&82));
        assert_eq!(
            plan.state_type_counts().get("append_only_attention_memory"),
            Some(&6)
        );

        let layer_00 = &plan.circuits[0];
        let layer_00_frame = layer_00.activation_frame_plan();
        assert_eq!(layer_00.pedal_id, "layer_00");
        assert_eq!(layer_00.nodes.len(), 16);
        assert_eq!(layer_00.temporary_signals.len(), 17);
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
        assert_eq!(layer_02.activation_frame_plan().slot_count, 5);
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
    }

    #[test]
    fn activation_frame_plan_reuses_temporary_signal_slots_by_liveness() {
        let graph = ResolvedLoweredPedalboard::from_index_file(lfm2_index_path()).unwrap();
        let plan = StreamCircuitExecutionPlan::from_graph(&graph).unwrap();
        let frame = plan.circuits[0].activation_frame_plan();

        assert_eq!(frame.liveness.len(), 17);
        assert_eq!(frame.slot_count, 4);
        assert_eq!(frame.slot_for("operator_norm_out"), Some(0));
        assert_eq!(frame.slot_for("gate_b"), Some(0));
        assert_eq!(frame.slot_for("gate_c"), Some(2));
        assert_eq!(frame.slot_for("projected_x"), Some(3));
        assert_eq!(frame.slot_for("operator_residual_out"), Some(0));

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
