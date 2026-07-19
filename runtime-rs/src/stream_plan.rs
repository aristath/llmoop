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
        let path = path.as_ref();
        let bytes = fs::read(path).map_err(|error| CircuitPlanError(error.to_string()))?;
        let mut index: Self =
            serde_json::from_slice(&bytes).map_err(|error| CircuitPlanError(error.to_string()))?;
        if index.schema != TENSOR_INDEX_SCHEMA {
            return Err(CircuitPlanError(format!(
                "unsupported tensor index schema {:?}",
                index.schema
            )));
        }
        let root = path.parent().unwrap_or_else(|| Path::new("."));
        for metadata in index.tensors.values_mut() {
            if let Some(source_file) = &metadata.source_file {
                let source_path = Path::new(source_file);
                if !source_path.is_absolute() {
                    metadata.source_file =
                        Some(root.join(source_path).to_string_lossy().into_owned());
                }
            }
        }
        Ok(index)
    }

    pub fn tensor_shape(&self, tensor: &str) -> Option<&[usize]> {
        self.tensors.get(tensor).map(|metadata| {
            metadata
                .logical_shape
                .as_deref()
                .unwrap_or(metadata.shape.as_slice())
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
pub struct TensorMetadata {
    pub dtype: String,
    pub shape: Vec<usize>,
    #[serde(default)]
    pub logical_shape: Option<Vec<usize>>,
    #[serde(default)]
    pub parameter_count: Option<usize>,
    #[serde(default)]
    pub byte_count: Option<usize>,
    #[serde(default)]
    pub data_offsets: Option<Vec<usize>>,
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
    pub transducer_parameter_ref_count: usize,
    pub transducer_parameters: Vec<PlannedParameterResource>,
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
        let mut transducer_parameter_ref_count = 0;
        let mut transducer_parameters_by_tensor: BTreeMap<String, Vec<PlannedParameterUse>> =
            BTreeMap::new();
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
                    element_bytes: state
                        .extra
                        .get("dtype")
                        .and_then(serde_json::Value::as_str)
                        .map(state_dtype_bytes)
                        .transpose()?,
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
                state_view_signal_count: activation_plan.state_view_signals.len(),
                slot_count: activation_frame.slot_count,
                slots: planned_activation_slots(activation_plan, &activation_frame),
                assignments: activation_frame.assignments,
            });
        }

        transducer_parameter_ref_count += collect_transducer_component_parameters(
            "input_transducer",
            &graph.index.graph.input_transducer,
            "input_transducer",
            &mut transducer_parameters_by_tensor,
        )?;
        transducer_parameter_ref_count += collect_output_transducer_parameters(
            &graph.index.graph.output_transducer,
            &mut transducer_parameters_by_tensor,
        )?;

        let parameters = parameters_by_tensor
            .into_iter()
            .map(|(tensor, uses)| PlannedParameterResource { tensor, uses })
            .collect();
        let transducer_parameters = transducer_parameters_by_tensor
            .into_iter()
            .map(|(tensor, uses)| PlannedParameterResource { tensor, uses })
            .collect();

        Ok(Self {
            circuit_count: graph.circuits.len(),
            node_count: execution_plan.total_node_count(),
            parameter_ref_count,
            parameters,
            transducer_parameter_ref_count,
            transducer_parameters,
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

    pub fn unique_transducer_parameter_tensor_count(&self) -> usize {
        self.transducer_parameters.len()
    }

    pub fn stream_state_count(&self) -> usize {
        self.state_allocations.len()
    }

    pub fn intermediate_activation_shapes_known(&self) -> bool {
        self.unknown_temporary_shape_count == 0
    }
}

fn collect_output_transducer_parameters(
    output_transducer: &serde_json::Value,
    parameters_by_tensor: &mut BTreeMap<String, Vec<PlannedParameterUse>>,
) -> Result<usize, CircuitPlanError> {
    let Some(components) = output_transducer
        .get("components")
        .and_then(serde_json::Value::as_array)
    else {
        return collect_transducer_component_parameters(
            "output_transducer",
            output_transducer,
            "output_transducer",
            parameters_by_tensor,
        );
    };

    let mut parameter_ref_count = 0usize;
    for (component_index, component) in components.iter().enumerate() {
        parameter_ref_count += collect_transducer_component_parameters(
            "output_transducer",
            component,
            &format!("component_{component_index}"),
            parameters_by_tensor,
        )?;
    }
    Ok(parameter_ref_count)
}

fn collect_transducer_component_parameters(
    transducer_id: &str,
    component: &serde_json::Value,
    fallback_component_id: &str,
    parameters_by_tensor: &mut BTreeMap<String, Vec<PlannedParameterUse>>,
) -> Result<usize, CircuitPlanError> {
    if component.is_null() {
        return Ok(0);
    }

    let component_id = component
        .get("id")
        .and_then(serde_json::Value::as_str)
        .unwrap_or(fallback_component_id);
    let component_type = component
        .get("type")
        .and_then(serde_json::Value::as_str)
        .map(str::to_string);
    let Some(params) = component
        .get("params")
        .and_then(serde_json::Value::as_object)
    else {
        return Ok(0);
    };

    let mut parameter_ref_count = 0usize;
    for (param_id, param_ref) in params {
        let tensor = param_ref
            .get("tensor")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| {
                CircuitPlanError(format!(
                    "{transducer_id}.{component_id} transducer parameter {param_id:?} has no source tensor"
                ))
            })?;
        parameter_ref_count += 1;
        parameters_by_tensor
            .entry(tensor.to_string())
            .or_default()
            .push(PlannedParameterUse {
                pedal_id: format!("{transducer_id}.{component_id}"),
                circuit_id: transducer_id.to_string(),
                param_id: param_id.clone(),
                role: component_type.clone(),
                layout: "transducer".to_string(),
                storage: "source_tensor_refs".to_string(),
            });
    }

    Ok(parameter_ref_count)
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
        "rms_norm"
        | "rms_norm_per_head"
        | "rms_norm_per_head_unscaled"
        | "silu"
        | "gelu_tanh"
        | "rotary_position_embedding"
        | "scalar_multiply" => Ok(repeat_shape(first_input_shape(node, signals), outputs)),
        "per_layer_embedding" => {
            let output_shape = attr_usize(node, "per_layer_width").map(|width| vec![width]);
            Ok(repeat_shape(output_shape, outputs))
        }
        "multiply"
        | "residual_add"
        | "scaled_residual_add"
        | "silu_multiply"
        | "sigmoid_multiply" => Ok(repeat_shape(
            compatible_input_shape(pedal_id, node, signals)?,
            outputs,
        )),
        "sigmoid_scalar_multiply" => Ok(repeat_shape(first_input_shape(node, signals), outputs)),
        "parallel_head_norm_rope_2way" => {
            if node.inputs.len() != 2 || node.outputs.len() != 2 || node.params.len() != 2 {
                return Err(CircuitPlanError(format!(
                    "{} node {} requires two head-norm/rope inputs, outputs, and parameters",
                    pedal_id, node.id
                )));
            }
            Ok(node
                .inputs
                .iter()
                .map(|input| signals.get(input).and_then(|signal| signal.shape.clone()))
                .collect())
        }
        "parallel_linear_2way" | "parallel_linear_3way" => {
            infer_parallel_linear_output_shapes(pedal_id, node, signals, params, tensor_index)
        }
        "linear" | "linear_residual" => {
            infer_linear_output_shapes(pedal_id, node, signals, params, tensor_index)
        }
        "linear_split_3way" => {
            infer_linear_split_output_shapes(pedal_id, node, signals, params, tensor_index)
        }
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
        "causal_conv1d_silu" => Ok(repeat_shape(first_input_shape(node, signals), outputs)),
        "gated_delta_step" => {
            let output_shape = attr_usize(node, "value_heads")
                .zip(attr_usize(node, "value_head_width"))
                .map(|(heads, width)| vec![heads * width]);
            Ok(repeat_shape(output_shape, outputs))
        }
        "rg_lru_step" => {
            let output_shape = attr_usize(node, "width").map(|width| vec![width]);
            Ok(repeat_shape(output_shape, outputs))
        }
        "moe_topk" => {
            let output_shape = attr_usize(node, "num_experts").map(|experts| vec![experts]);
            Ok(repeat_shape(output_shape, outputs))
        }
        "sparse_moe_experts" => {
            let output_shape = attr_usize(node, "num_experts")
                .zip(attr_usize(node, "hidden_size"))
                .map(|(experts, hidden)| vec![experts, hidden]);
            Ok(repeat_shape(output_shape, outputs))
        }
        "moe_reduce" => {
            let output_shape = attr_usize(node, "hidden_size").map(|hidden| vec![hidden]);
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

fn infer_parallel_linear_output_shapes(
    pedal_id: &str,
    node: &CircuitNode,
    signals: &BTreeMap<String, PlannedSignal>,
    params: &BTreeMap<String, ParameterRef>,
    tensor_index: Option<&TensorIndex>,
) -> Result<Vec<Option<Vec<usize>>>, CircuitPlanError> {
    let expected_branch_count = match node.op.as_str() {
        "parallel_linear_2way" => 2,
        "parallel_linear_3way" => 3,
        _ => unreachable!("parallel-linear shape inference called for {}", node.op),
    };
    let declared_branch_count = attr_usize(node, "branch_count");
    if node.params.len() != expected_branch_count
        || node.outputs.len() != expected_branch_count
        || declared_branch_count != Some(expected_branch_count)
    {
        return Err(CircuitPlanError(format!(
            "{} node {} declares {:?} parallel-linear branches for {} parameters and {} outputs; expected {}",
            pedal_id,
            node.id,
            declared_branch_count,
            node.params.len(),
            node.outputs.len(),
            expected_branch_count
        )));
    }
    let Some(tensor_index) = tensor_index else {
        return Ok(vec![None; node.outputs.len()]);
    };
    let input_shape = first_input_shape(node, signals);
    let input_width = input_shape.as_ref().and_then(|shape| shape.last()).copied();
    node.params
        .iter()
        .map(|param_id| {
            let parameter = params.get(param_id).ok_or_else(|| {
                CircuitPlanError(format!(
                    "{} node {} cannot resolve parallel-linear parameter {:?}",
                    pedal_id, node.id, param_id
                ))
            })?;
            let tensor = parameter.tensor.as_deref().ok_or_else(|| {
                CircuitPlanError(format!(
                    "{} node {} parallel-linear parameter {:?} has no tensor",
                    pedal_id, node.id, param_id
                ))
            })?;
            let weight_shape = tensor_index.tensor_shape(tensor).ok_or_else(|| {
                CircuitPlanError(format!(
                    "{} node {} parallel-linear tensor {:?} has no shape",
                    pedal_id, node.id, tensor
                ))
            })?;
            if weight_shape.len() != 2 {
                return Ok(None);
            }
            if input_width.is_some_and(|width| width != weight_shape[1]) {
                return Err(CircuitPlanError(format!(
                    "{} node {} parallel-linear input width {:?} does not match parameter {:?} width {}",
                    pedal_id, node.id, input_width, param_id, weight_shape[1]
                )));
            }
            let mut output_shape = input_shape
                .clone()
                .unwrap_or_else(|| vec![weight_shape[0]]);
            if let Some(last) = output_shape.last_mut() {
                *last = weight_shape[0];
            }
            Ok(Some(output_shape))
        })
        .collect()
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
    if let Some(part_widths) = node
        .attrs
        .get("part_widths")
        .and_then(|value| value.as_array())
    {
        let widths = part_widths
            .iter()
            .map(|value| value.as_u64().and_then(|width| usize::try_from(width).ok()))
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| {
                CircuitPlanError(format!(
                    "{} node {} has non-integer split part widths",
                    pedal_id, node.id
                ))
            })?;
        if widths.len() != node.outputs.len() || widths.iter().sum::<usize>() != *channel_dim {
            return Err(CircuitPlanError(format!(
                "{} node {} cannot split shape {:?} into widths {:?}",
                pedal_id,
                node.id,
                first_input_shape(node, signals),
                widths
            )));
        }
        return Ok(widths
            .into_iter()
            .map(|width| {
                let mut shape = input_shape.clone();
                *shape.last_mut().unwrap() = width;
                Some(shape)
            })
            .collect());
    }
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

fn infer_linear_split_output_shapes(
    pedal_id: &str,
    node: &CircuitNode,
    signals: &BTreeMap<String, PlannedSignal>,
    params: &BTreeMap<String, ParameterRef>,
    tensor_index: Option<&TensorIndex>,
) -> Result<Vec<Option<Vec<usize>>>, CircuitPlanError> {
    let combined_shapes =
        infer_linear_output_shapes(pedal_id, node, signals, params, tensor_index)?;
    let Some(Some(combined_shape)) = combined_shapes.first() else {
        return Ok(vec![None; node.outputs.len()]);
    };
    let Some(combined_width) = combined_shape.last().copied() else {
        return Ok(vec![None; node.outputs.len()]);
    };
    let part_widths = node
        .attrs
        .get("part_widths")
        .and_then(|value| value.as_array())
        .and_then(|widths| {
            widths
                .iter()
                .map(|value| value.as_u64().and_then(|width| usize::try_from(width).ok()))
                .collect::<Option<Vec<_>>>()
        })
        .ok_or_else(|| {
            CircuitPlanError(format!(
                "{} node {} requires integer linear-split part widths",
                pedal_id, node.id
            ))
        })?;
    if part_widths.len() != node.outputs.len()
        || part_widths.iter().sum::<usize>() != combined_width
    {
        return Err(CircuitPlanError(format!(
            "{} node {} cannot split linear output shape {:?} into widths {:?}",
            pedal_id, node.id, combined_shape, part_widths
        )));
    }
    Ok(part_widths
        .into_iter()
        .map(|width| {
            let mut shape = combined_shape.clone();
            *shape.last_mut().unwrap() = width;
            Some(shape)
        })
        .collect())
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

fn product(shape: &[usize]) -> Option<usize> {
    shape
        .iter()
        .try_fold(1usize, |total, value| total.checked_mul(*value))
}

fn node_output_storage(node: &CircuitNode) -> SignalStorage {
    match node.op.as_str() {
        "append_state_update" | "rolling_state_update" => SignalStorage::StateView,
        _ => SignalStorage::Activation,
    }
}

fn planned_activation_slots(
    activation_plan: &CircuitActivationPlan,
    frame: &ActivationFramePlan,
) -> Vec<PlannedActivationSlot> {
    let mut signals_by_slot: BTreeMap<usize, Vec<String>> = BTreeMap::new();
    for assignment in &frame.assignments {
        signals_by_slot
            .entry(assignment.slot)
            .or_default()
            .push(assignment.signal_id.clone());
    }

    signals_by_slot
        .into_iter()
        .map(|(slot, signal_ids)| {
            let mut max_elements = Some(0usize);
            for signal_id in &signal_ids {
                let elements = activation_plan
                    .signal(signal_id)
                    .and_then(|signal| signal.shape.as_ref())
                    .and_then(|shape| product(shape));
                match (max_elements, elements) {
                    (Some(max), Some(elements)) => max_elements = Some(max.max(elements)),
                    _ => {
                        max_elements = None;
                        break;
                    }
                }
            }
            PlannedActivationSlot {
                slot,
                signal_ids,
                max_elements,
            }
        })
        .collect()
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

    fn compiled_artifact_dir(env_var: &str, root_name: &str, marker_file: &str) -> PathBuf {
        if let Ok(path) = std::env::var(env_var) {
            return PathBuf::from(path);
        }
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join(root_name);
        let mut candidates = std::fs::read_dir(&root)
            .unwrap_or_else(|_| panic!("set {env_var} or compile a model into {}", root.display()))
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|path| path.is_dir() && path.join(marker_file).exists())
            .collect::<Vec<_>>();
        candidates.sort();
        candidates
            .pop()
            .unwrap_or_else(|| panic!("set {env_var} or compile a model into {}", root.display()))
    }

    fn fixture_model_index_path() -> PathBuf {
        compiled_artifact_dir(
            "LLMOOP_TEST_LOWERED_DIR",
            "lowered",
            "pedalboard.circuits.json",
        )
        .join("pedalboard.circuits.json")
    }

    fn fixture_model_tensor_index_path() -> PathBuf {
        compiled_artifact_dir("LLMOOP_TEST_TRANSPILED_DIR", "transpiled", "tensors.json")
            .join("tensors.json")
    }

    #[test]
    fn infers_unequal_fused_projection_split_shapes() {
        let node = crate::stream_circuit::CircuitNode {
            id: "qkv_split".to_string(),
            op: "split".to_string(),
            inputs: vec!["qkv".to_string()],
            outputs: vec!["q".to_string(), "k".to_string(), "v".to_string()],
            params: Vec::new(),
            state_reads: Vec::new(),
            state_writes: Vec::new(),
            attrs: serde_json::json!({"part_widths": [16, 8, 8]}),
        };
        let signals = BTreeMap::from([(
            "qkv".to_string(),
            PlannedSignal {
                id: "qkv".to_string(),
                producer: SignalProducer::BoundaryInput,
                consumers: vec!["qkv_split".to_string()],
                shape: Some(vec![32]),
                storage: SignalStorage::Boundary,
                is_boundary_output: false,
            },
        )]);

        assert_eq!(
            infer_split_output_shapes("layer_00", &node, &signals).unwrap(),
            vec![Some(vec![16]), Some(vec![8]), Some(vec![8])]
        );
    }

    #[test]
    fn rejects_parallel_linear_branch_metadata_mismatch_without_tensor_index() {
        let node = crate::stream_circuit::CircuitNode {
            id: "qkv".to_string(),
            op: "parallel_linear_3way".to_string(),
            inputs: vec!["hidden".to_string()],
            outputs: vec!["q".to_string(), "k".to_string(), "v".to_string()],
            params: vec![
                "q_weight".to_string(),
                "k_weight".to_string(),
                "v_weight".to_string(),
            ],
            state_reads: Vec::new(),
            state_writes: Vec::new(),
            attrs: serde_json::json!({"branch_count": 2}),
        };

        let error = infer_parallel_linear_output_shapes(
            "attention",
            &node,
            &BTreeMap::new(),
            &BTreeMap::new(),
            None,
        )
        .unwrap_err();

        assert!(error.0.contains("expected 3"), "{}", error.0);
    }

    #[test]
    fn plans_fixture_model_lowered_pedalboard_activation_schedule() {
        let graph = ResolvedLoweredPedalboard::from_index_file(fixture_model_index_path()).unwrap();

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
    fn tensor_index_enables_fixture_model_signal_shape_planning() {
        let graph = ResolvedLoweredPedalboard::from_index_file(fixture_model_index_path()).unwrap();
        let tensor_index = TensorIndex::from_json_file(fixture_model_tensor_index_path()).unwrap();

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

        let layer_00_bank = resource_plan
            .activation_banks
            .iter()
            .find(|bank| bank.pedal_id == "layer_00")
            .unwrap();
        assert_eq!(layer_00_bank.slot_count, 4);
        assert_eq!(
            layer_00_bank
                .slots
                .iter()
                .map(|slot| slot.max_elements)
                .collect::<Vec<_>>(),
            vec![Some(2560), Some(3072), Some(2560), Some(2560)]
        );

        let layer_02_bank = resource_plan
            .activation_banks
            .iter()
            .find(|bank| bank.pedal_id == "layer_02")
            .unwrap();
        assert_eq!(layer_02_bank.slot_count, 4);
        assert_eq!(
            layer_02_bank
                .slots
                .iter()
                .map(|slot| slot.max_elements)
                .collect::<Vec<_>>(),
            vec![Some(1024), Some(2560), Some(2560), Some(2560)]
        );
    }

    #[test]
    fn tensor_index_uses_logical_shape_without_changing_storage_shape() {
        let index: TensorIndex = serde_json::from_str(
            r#"{
              "schema": "llmoop.tensor_index.v1",
              "tensors": {
                "projection.qweight": {
                  "dtype": "I32",
                  "shape": [64, 768],
                  "logical_shape": [768, 512]
                }
              }
            }"#,
        )
        .unwrap();

        assert_eq!(
            index.tensor_shape("projection.qweight"),
            Some([768, 512].as_slice())
        );
        assert_eq!(index.tensors["projection.qweight"].shape, vec![64, 768]);
    }

    #[test]
    fn resource_plan_names_fixture_model_mount_resources() {
        let graph = ResolvedLoweredPedalboard::from_index_file(fixture_model_index_path()).unwrap();
        let execution_plan = StreamCircuitExecutionPlan::from_graph(&graph).unwrap();

        let resource_plan =
            StreamCircuitResourcePlan::from_graph_and_plan(&graph, &execution_plan).unwrap();

        assert_eq!(resource_plan.circuit_count, 14);
        assert_eq!(resource_plan.node_count, 242);
        assert_eq!(resource_plan.parameter_ref_count, 130);
        assert_eq!(resource_plan.unique_parameter_tensor_count(), 130);
        assert_eq!(resource_plan.transducer_parameter_ref_count, 3);
        assert_eq!(resource_plan.unique_transducer_parameter_tensor_count(), 2);
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

        let embed_tokens = resource_plan
            .transducer_parameters
            .iter()
            .find(|parameter| parameter.tensor == "model.embed_tokens.weight")
            .unwrap();
        assert_eq!(embed_tokens.uses.len(), 2);
        assert_eq!(
            embed_tokens
                .uses
                .iter()
                .map(|parameter_use| parameter_use.pedal_id.as_str())
                .collect::<Vec<_>>(),
            vec![
                "input_transducer.token_embedding",
                "output_transducer.output_projection",
            ]
        );
        assert_eq!(embed_tokens.uses[0].param_id, "weight");
        assert_eq!(
            embed_tokens.uses[0].role.as_deref(),
            Some("embedding_lookup")
        );
        assert_eq!(embed_tokens.uses[0].layout, "transducer");
        assert_eq!(
            embed_tokens.uses[1].role.as_deref(),
            Some("linear_projection")
        );

        let embedding_norm = resource_plan
            .transducer_parameters
            .iter()
            .find(|parameter| parameter.tensor == "model.embedding_norm.weight")
            .unwrap();
        assert_eq!(embedding_norm.uses.len(), 1);
        assert_eq!(
            embedding_norm.uses[0].pedal_id,
            "output_transducer.output_norm"
        );
        assert_eq!(embedding_norm.uses[0].role.as_deref(), Some("rms_norm"));

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
        let graph = ResolvedLoweredPedalboard::from_index_file(fixture_model_index_path()).unwrap();
        let mut execution_plan = StreamCircuitExecutionPlan::from_graph(&graph).unwrap();
        execution_plan.circuits.pop();

        let error =
            StreamCircuitResourcePlan::from_graph_and_plan(&graph, &execution_plan).unwrap_err();

        assert!(error.to_string().contains("graph circuit count 14"));
    }

    #[test]
    fn activation_plan_tracks_signal_producers_and_consumers() {
        let graph = ResolvedLoweredPedalboard::from_index_file(fixture_model_index_path()).unwrap();
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
        let graph = ResolvedLoweredPedalboard::from_index_file(fixture_model_index_path()).unwrap();
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
        let graph = ResolvedLoweredPedalboard::from_index_file(fixture_model_index_path()).unwrap();
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
