use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

pub const STREAM_CIRCUIT_SCHEMA: &str = "nerve.stream_circuit.v1";
pub const CIRCUIT_PARAMS_SCHEMA: &str = "nerve.circuit_params.v1";
pub const CIRCUIT_STATE_SCHEMA: &str = "nerve.circuit_state.v1";
pub const LOWERED_PEDALBOARD_SCHEMA: &str = "nerve.lowered_pedalboard.v1";
pub const STREAM_CIRCUIT_PLACEMENT_SCHEMA: &str = "nerve.stream_circuit_placement.v1";
pub const STREAM_CIRCUIT_RUNTIME_PATCH_SCHEMA: &str = "nerve.stream_circuit_runtime_patch.v1";
pub const RUNTIME_DEFAULT_LOGICAL_DEVICE_ID: &str = "runtime_default";
pub const RUNTIME_CABLE_ROUTES_SCHEMA: &str = "nerve.runtime_cable_routes.v1";
pub const RUNTIME_DEVICE_BINDINGS_SCHEMA: &str = "nerve.runtime_device_bindings.v1";
pub const RUNTIME_TOPOLOGY_SCHEMA: &str = "nerve.runtime_topology.v1";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CircuitArtifactError(pub String);

impl Display for CircuitArtifactError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for CircuitArtifactError {}

impl From<io::Error> for CircuitArtifactError {
    fn from(error: io::Error) -> Self {
        Self(error.to_string())
    }
}

impl From<serde_json::Error> for CircuitArtifactError {
    fn from(error: serde_json::Error) -> Self {
        Self(error.to_string())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CircuitPort {
    pub id: String,
    pub signal: String,
    pub shape: Vec<usize>,
    #[serde(default)]
    pub source: Option<String>,
    #[serde(default)]
    pub pedal_port: Option<String>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CircuitBoundary {
    #[serde(default)]
    pub inputs: Vec<CircuitPort>,
    #[serde(default)]
    pub outputs: Vec<CircuitPort>,
    #[serde(default)]
    pub controls: Vec<Value>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CircuitSource {
    pub pedal_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_layer_index: Option<usize>,
    pub source_operator_type: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StatePort {
    pub id: String,
    #[serde(rename = "type")]
    pub state_type: String,
    #[serde(default)]
    pub shape: Option<Vec<usize>>,
    #[serde(default)]
    pub update: Option<String>,
    #[serde(default)]
    pub key_shape_per_token: Option<Vec<usize>>,
    #[serde(default)]
    pub value_shape_per_token: Option<Vec<usize>>,
    #[serde(default)]
    pub growth: Option<String>,
    #[serde(default)]
    pub max_dynamic_activations: Option<usize>,
    #[serde(default)]
    pub sharing: Option<String>,
    #[serde(default)]
    pub owner: Option<String>,
    #[serde(default)]
    pub layout: Option<String>,
    #[serde(default)]
    pub source_layout: Option<String>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

impl StatePort {
    pub fn static_elements(&self) -> Option<usize> {
        self.shape.as_ref().and_then(|shape| product(shape))
    }

    pub fn elements_per_activation(&self) -> Option<usize> {
        match (&self.key_shape_per_token, &self.value_shape_per_token) {
            (Some(key), Some(value)) => Some(product(key)? + product(value)?),
            (Some(key), None) => product(key),
            (None, Some(value)) => product(value),
            (None, None) => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParameterRef {
    #[serde(default)]
    pub tensor: Option<String>,
    #[serde(default)]
    pub role: Option<String>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CircuitParameters {
    pub layout: String,
    pub storage: String,
    #[serde(default)]
    pub refs: BTreeMap<String, ParameterRef>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CircuitNode {
    pub id: String,
    pub op: String,
    #[serde(default)]
    pub inputs: Vec<String>,
    #[serde(default)]
    pub outputs: Vec<String>,
    #[serde(default)]
    pub params: Vec<String>,
    #[serde(default)]
    pub state_reads: Vec<String>,
    #[serde(default)]
    pub state_writes: Vec<String>,
    #[serde(default)]
    pub attrs: Value,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StreamCircuit {
    pub schema: String,
    pub id: String,
    pub source: CircuitSource,
    pub runtime_role: CircuitRuntimeRole,
    pub behavioral_role: String,
    pub implementation: String,
    pub boundary: CircuitBoundary,
    #[serde(default)]
    pub state_ports: Vec<StatePort>,
    pub parameters: CircuitParameters,
    #[serde(default)]
    pub nodes: Vec<CircuitNode>,
    #[serde(default)]
    pub behavioral_error_contract: Value,
    #[serde(default)]
    pub lowering_notes: Vec<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CircuitRuntimeRole {
    SignalProcessor,
    InputTransducer,
    OutputTransducer,
    Sampler,
    DraftProcessor,
    DraftInputAdapter,
    DraftOutputTransducer,
}

impl CircuitRuntimeRole {
    pub fn is_signal_processor(self) -> bool {
        matches!(self, Self::SignalProcessor)
    }
}

impl StreamCircuit {
    pub fn from_json_file(path: impl AsRef<Path>) -> Result<Self, CircuitArtifactError> {
        read_json(path)
    }

    pub fn validate_contract(&self) -> Result<(), CircuitArtifactError> {
        let mut issues = Vec::new();
        if self.schema != STREAM_CIRCUIT_SCHEMA {
            issues.push(format!(
                "unsupported stream circuit schema {:?}",
                self.schema
            ));
        }
        if self.id.is_empty() {
            issues.push("stream circuit id must not be empty".to_string());
        }
        if self.boundary.inputs.is_empty() {
            issues.push(format!("{} boundary.inputs must not be empty", self.id));
        }
        if self.boundary.outputs.is_empty() {
            issues.push(format!("{} boundary.outputs must not be empty", self.id));
        }
        if self.nodes.is_empty() {
            issues.push(format!("{} nodes must not be empty", self.id));
        }

        let mut produced = BTreeSet::new();
        let mut produced_by = BTreeMap::new();
        let mut input_port_ids = BTreeSet::new();
        for port in &self.boundary.inputs {
            validate_boundary_port(port, "input", &mut issues);
            if !input_port_ids.insert(port.id.clone()) {
                issues.push(format!("duplicate boundary input port id {:?}", port.id));
            }
            produced.insert(port.id.clone());
            produced_by.insert(port.id.clone(), "boundary.input".to_string());
        }

        let state_ids: BTreeSet<_> = self.state_ports.iter().map(|state| &state.id).collect();
        if state_ids.len() != self.state_ports.len() {
            issues.push(format!("{} has duplicate state port ids", self.id));
        }
        for state in &self.state_ports {
            if state.max_dynamic_activations == Some(0) {
                issues.push(format!(
                    "state port {:?} max_dynamic_activations must be positive",
                    state.id
                ));
            }
            if state.max_dynamic_activations.is_some() && state.elements_per_activation().is_none()
            {
                issues.push(format!(
                    "state port {:?} bounds dynamic activations but has no per-activation shape",
                    state.id
                ));
            }
        }
        let param_ids: BTreeSet<_> = self.parameters.refs.keys().collect();
        let mut node_ids = BTreeSet::new();

        for node in &self.nodes {
            if node.id.is_empty() {
                issues.push(format!("{} contains a node with an empty id", self.id));
            } else if !node_ids.insert(node.id.clone()) {
                issues.push(format!("{} has duplicate node id {:?}", self.id, node.id));
            }
            if node.op.is_empty() {
                issues.push(format!("node {:?} has an empty op", node.id));
            }

            for input in &node.inputs {
                if !produced.contains(input) && !state_ids.contains(input) {
                    issues.push(format!(
                        "node {} input {:?} is not produced or declared as state",
                        node.id, input
                    ));
                }
            }
            for param in &node.params {
                if !param_ids.contains(param) {
                    issues.push(format!(
                        "node {} parameter {:?} is not declared",
                        node.id, param
                    ));
                }
            }
            for state in node.state_reads.iter().chain(node.state_writes.iter()) {
                if !state_ids.contains(state) {
                    issues.push(format!(
                        "node {} state {:?} is not declared",
                        node.id, state
                    ));
                }
            }
            if node.outputs.is_empty() {
                issues.push(format!("node {} must declare output signals", node.id));
            }
            for output in &node.outputs {
                if let Some(previous) = produced_by.get(output) {
                    issues.push(format!(
                        "signal {:?} is produced twice, by {} and {}",
                        output, previous, node.id
                    ));
                    continue;
                }
                produced.insert(output.clone());
                produced_by.insert(output.clone(), node.id.clone());
            }
        }

        let mut output_port_ids = BTreeSet::new();
        for output in &self.boundary.outputs {
            validate_boundary_port(output, "output", &mut issues);
            if !output_port_ids.insert(output.id.clone()) {
                issues.push(format!("duplicate boundary output port id {:?}", output.id));
                continue;
            }
            let source = output.source.as_ref().unwrap_or(&output.id);
            if !produced.contains(source) {
                issues.push(format!(
                    "boundary output {} source {:?} is not produced",
                    output.id, source
                ));
            }
        }

        if issues.is_empty() {
            Ok(())
        } else {
            Err(CircuitArtifactError(format!(
                "stream circuit {} validation failed:\n- {}",
                self.id,
                issues.join("\n- ")
            )))
        }
    }
}

fn validate_boundary_port(port: &CircuitPort, direction: &str, issues: &mut Vec<String>) {
    if port.id.is_empty() {
        issues.push(format!("boundary {direction} port id must not be empty"));
    }
    if port.signal.is_empty() {
        issues.push(format!(
            "boundary {direction} port {:?} signal must not be empty",
            port.id
        ));
    }
    if port.shape.is_empty() || port.shape.contains(&0) {
        issues.push(format!(
            "boundary {direction} port {:?} shape must contain positive dimensions",
            port.id
        ));
    }
    if port.pedal_port.as_deref().is_none_or(str::is_empty) {
        issues.push(format!(
            "boundary {direction} port {:?} must map to a non-empty pedal_port",
            port.id
        ));
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LoweredPedalboardSource {
    pub format: String,
    pub artifact_root: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LoweredCircuitRef {
    pub id: String,
    pub operator_type: String,
    pub runtime_role: CircuitRuntimeRole,
    pub circuit: String,
    pub params: String,
    pub state: String,
    pub implementation: String,
    pub behavioral_role: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StreamCircuitCableEndpoint {
    pub pedal_id: String,
    pub port_id: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StreamCircuitGraphCable {
    pub id: String,
    pub source: StreamCircuitCableEndpoint,
    pub destination: StreamCircuitCableEndpoint,
    pub connection: StreamCircuitConnection,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StreamCircuitGraphBoundaryPort {
    pub id: String,
    pub endpoint: StreamCircuitCableEndpoint,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StreamCircuitGraphBoundary {
    pub external_inputs: Vec<StreamCircuitGraphBoundaryPort>,
    pub public_outputs: Vec<StreamCircuitGraphBoundaryPort>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum StreamCircuitConnection {
    #[default]
    Forward,
    TemporalFeedback {
        delay_activations: usize,
    },
}

impl StreamCircuitConnection {
    pub fn is_forward(&self) -> bool {
        matches!(self, Self::Forward)
    }

    pub fn validate(&self, cable_id: &str) -> Result<(), CircuitPlacementError> {
        if let Self::TemporalFeedback { delay_activations } = self
            && *delay_activations == 0
        {
            return Err(CircuitPlacementError(format!(
                "runtime patch temporal feedback cable {cable_id} must delay at least one activation"
            )));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LoweredPedalboardGraph {
    pub wiring: String,
    #[serde(default)]
    pub circuits: Vec<LoweredCircuitRef>,
    pub cables: Vec<StreamCircuitGraphCable>,
    pub boundary: StreamCircuitGraphBoundary,
    #[serde(default)]
    pub input_transducer: Value,
    #[serde(default)]
    pub output_transducer: Value,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LoweredPedalboardSummary {
    pub circuit_count: usize,
    #[serde(default)]
    pub operator_counts: BTreeMap<String, usize>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LoweredPedalboard {
    pub schema: String,
    pub source: LoweredPedalboardSource,
    #[serde(default)]
    pub architecture: Value,
    #[serde(default)]
    pub dimensions: Value,
    pub graph: LoweredPedalboardGraph,
    pub summary: LoweredPedalboardSummary,
    #[serde(default)]
    pub notes: Vec<String>,
}

impl LoweredPedalboard {
    pub fn from_json_file(path: impl AsRef<Path>) -> Result<Self, CircuitArtifactError> {
        read_json(path)
    }

    pub fn validate_index(&self) -> Result<(), CircuitArtifactError> {
        let mut issues = Vec::new();
        if self.schema != LOWERED_PEDALBOARD_SCHEMA {
            issues.push(format!(
                "unsupported lowered pedalboard schema {:?}",
                self.schema
            ));
        }
        if self.graph.circuits.is_empty() {
            issues.push("lowered pedalboard contains no circuits".to_string());
        }
        if self.summary.circuit_count != self.graph.circuits.len() {
            issues.push(format!(
                "summary circuit_count {} does not match graph circuit count {}",
                self.summary.circuit_count,
                self.graph.circuits.len()
            ));
        }

        let mut ids = BTreeSet::new();
        for circuit in &self.graph.circuits {
            if !ids.insert(circuit.id.clone()) {
                issues.push(format!("duplicate lowered circuit id {:?}", circuit.id));
            }
            if circuit.circuit.is_empty() || circuit.params.is_empty() || circuit.state.is_empty() {
                issues.push(format!(
                    "lowered circuit {} has missing artifact path",
                    circuit.id
                ));
            }
        }

        validate_index_boundary_ports(
            "external input",
            &self.graph.boundary.external_inputs,
            &ids,
            &mut issues,
        );
        validate_index_boundary_ports(
            "public output",
            &self.graph.boundary.public_outputs,
            &ids,
            &mut issues,
        );

        let mut cable_ids = BTreeSet::new();
        for cable in &self.graph.cables {
            if cable.id.is_empty() || !cable_ids.insert(cable.id.clone()) {
                issues.push(format!(
                    "invalid or duplicate graph cable id {:?}",
                    cable.id
                ));
            }
            if !ids.contains(&cable.source.pedal_id) {
                issues.push(format!(
                    "graph cable {} references unknown source pedal {:?}",
                    cable.id, cable.source.pedal_id
                ));
            }
            if !ids.contains(&cable.destination.pedal_id) {
                issues.push(format!(
                    "graph cable {} references unknown destination pedal {:?}",
                    cable.id, cable.destination.pedal_id
                ));
            }
            if matches!(
                cable.connection,
                StreamCircuitConnection::TemporalFeedback {
                    delay_activations: 0
                }
            ) {
                issues.push(format!(
                    "graph temporal feedback cable {} must delay at least one activation",
                    cable.id
                ));
            }
        }

        if issues.is_empty() {
            Ok(())
        } else {
            Err(CircuitArtifactError(format!(
                "lowered pedalboard validation failed:\n- {}",
                issues.join("\n- ")
            )))
        }
    }
}

fn validate_index_boundary_ports(
    kind: &str,
    ports: &[StreamCircuitGraphBoundaryPort],
    circuit_ids: &BTreeSet<String>,
    issues: &mut Vec<String>,
) {
    if ports.is_empty() {
        issues.push(format!("lowered pedalboard declares no {kind}s"));
        return;
    }
    let mut ids = BTreeSet::new();
    let mut endpoints = BTreeSet::new();
    for port in ports {
        if port.id.is_empty() || !ids.insert(port.id.as_str()) {
            issues.push(format!("invalid or duplicate {kind} id {:?}", port.id));
        }
        if !circuit_ids.contains(&port.endpoint.pedal_id) {
            issues.push(format!(
                "{kind} {} references unknown pedal {:?}",
                port.id, port.endpoint.pedal_id
            ));
        }
        if port.endpoint.port_id.is_empty()
            || !endpoints.insert((
                port.endpoint.pedal_id.as_str(),
                port.endpoint.port_id.as_str(),
            ))
        {
            issues.push(format!(
                "{kind} {} has an empty or duplicate endpoint {}.{}",
                port.id, port.endpoint.pedal_id, port.endpoint.port_id
            ));
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedLoweredPedalboard {
    pub artifact_root: PathBuf,
    pub index: LoweredPedalboard,
    pub circuits: Vec<ResolvedCircuitArtifact>,
}

impl ResolvedLoweredPedalboard {
    pub fn from_index_file(path: impl AsRef<Path>) -> Result<Self, CircuitArtifactError> {
        let path = path.as_ref();
        let artifact_root = path
            .parent()
            .ok_or_else(|| {
                CircuitArtifactError(format!(
                    "lowered pedalboard path {:?} does not have a parent directory",
                    path
                ))
            })?
            .to_path_buf();
        let index = LoweredPedalboard::from_json_file(path)?;
        index.validate_index()?;

        let mut circuits = Vec::with_capacity(index.graph.circuits.len());
        for pedal in &index.graph.circuits {
            let circuit = StreamCircuit::from_json_file(resolve_artifact_path(
                &artifact_root,
                &pedal.circuit,
            ))?;
            let params = CircuitParamsArtifact::from_json_file(resolve_artifact_path(
                &artifact_root,
                &pedal.params,
            ))?;
            let state = CircuitStateArtifact::from_json_file(resolve_artifact_path(
                &artifact_root,
                &pedal.state,
            ))?;
            let resolved = ResolvedCircuitArtifact {
                pedal: pedal.clone(),
                circuit,
                params,
                state,
            };
            resolved.validate()?;
            circuits.push(resolved);
        }

        Ok(Self {
            artifact_root,
            index,
            circuits,
        })
    }

    pub fn single_device_placement_plan(
        &self,
        device_id: impl Into<String>,
    ) -> Result<StreamCircuitPlacementPlan, CircuitPlacementError> {
        let spec = StreamCircuitPlacementSpec::new(device_id);
        self.placement_plan(&spec)
    }

    pub fn placement_plan(
        &self,
        spec: &StreamCircuitPlacementSpec,
    ) -> Result<StreamCircuitPlacementPlan, CircuitPlacementError> {
        StreamCircuitPlacementPlan::from_graph(self, spec)
    }

    pub fn default_runtime_patch(
        &self,
        default_device_id: impl Into<String>,
    ) -> Result<StreamCircuitRuntimePatch, CircuitPlacementError> {
        StreamCircuitRuntimePatch::from_source_series(self, default_device_id)
    }

    pub fn runtime_patch_from_placement(
        &self,
        spec: &StreamCircuitPlacementSpec,
    ) -> Result<StreamCircuitRuntimePatch, CircuitPlacementError> {
        StreamCircuitRuntimePatch::from_placement_spec(self, spec)
    }

    pub fn instantiate_runtime_patch(
        &self,
        patch: &StreamCircuitRuntimePatch,
    ) -> Result<ResolvedLoweredPedalboard, CircuitPlacementError> {
        patch.instantiate_graph(self)
    }
}
