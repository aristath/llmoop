use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::types::{
    DeviceMemoryPlan, HostPortsManifest, InstalledProcessorManifest, MemoryRegion,
    MemoryRegionKind, MemorySharing, PermanentCircuitManifest, StateAllocation, StreamTemplate,
};

pub const STREAM_CIRCUIT_SCHEMA: &str = "llmoop.stream_circuit.v1";
pub const CIRCUIT_PARAMS_SCHEMA: &str = "llmoop.circuit_params.v1";
pub const CIRCUIT_STATE_SCHEMA: &str = "llmoop.circuit_state.v1";
pub const LOWERED_PEDALBOARD_SCHEMA: &str = "llmoop.lowered_pedalboard.v1";

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
    pub pedal_file: String,
    pub source_layer_index: usize,
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
        for port in &self.boundary.inputs {
            if !produced.insert(port.id.clone()) {
                issues.push(format!("duplicate boundary input signal {:?}", port.id));
            }
            produced_by.insert(port.id.clone(), "boundary.input".to_string());
        }

        let state_ids: BTreeSet<_> = self.state_ports.iter().map(|state| &state.id).collect();
        if state_ids.len() != self.state_ports.len() {
            issues.push(format!("{} has duplicate state port ids", self.id));
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

        for output in &self.boundary.outputs {
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

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CircuitParamsArtifact {
    pub schema: String,
    pub circuit: String,
    pub layout: String,
    pub storage: String,
    #[serde(default)]
    pub refs: BTreeMap<String, ParameterRef>,
}

impl CircuitParamsArtifact {
    pub fn from_json_file(path: impl AsRef<Path>) -> Result<Self, CircuitArtifactError> {
        read_json(path)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CircuitStateArtifact {
    pub schema: String,
    pub circuit: String,
    #[serde(default)]
    pub state_ports: Vec<StatePort>,
}

impl CircuitStateArtifact {
    pub fn from_json_file(path: impl AsRef<Path>) -> Result<Self, CircuitArtifactError> {
        read_json(path)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LoweredPedalboardSource {
    pub pedalboard_dir: String,
    pub model_file: String,
    pub source_model_dir: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LoweredCircuitRef {
    pub id: String,
    pub operator_type: String,
    pub pedal_file: String,
    pub circuit: String,
    pub params: String,
    pub state: String,
    pub implementation: String,
    pub behavioral_role: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LoweredPedalboardGraph {
    pub wiring: String,
    #[serde(default)]
    pub circuits: Vec<LoweredCircuitRef>,
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
        if self.graph.wiring != "series" {
            issues.push(format!(
                "only series wiring is currently validated, got {:?}",
                self.graph.wiring
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

#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedCircuitArtifact {
    pub pedal: LoweredCircuitRef,
    pub circuit: StreamCircuit,
    pub params: CircuitParamsArtifact,
    pub state: CircuitStateArtifact,
}

impl ResolvedCircuitArtifact {
    pub fn validate(&self) -> Result<(), CircuitArtifactError> {
        self.circuit.validate_contract()?;
        if self.pedal.id != self.circuit.source.pedal_id {
            return Err(CircuitArtifactError(format!(
                "lowered circuit id {:?} does not match circuit source pedal {:?}",
                self.pedal.id, self.circuit.source.pedal_id
            )));
        }
        if self.pedal.operator_type != self.circuit.source.source_operator_type {
            return Err(CircuitArtifactError(format!(
                "lowered circuit {} operator {:?} does not match circuit source operator {:?}",
                self.pedal.id, self.pedal.operator_type, self.circuit.source.source_operator_type
            )));
        }
        if self.pedal.implementation != self.circuit.implementation {
            return Err(CircuitArtifactError(format!(
                "lowered circuit {} implementation {:?} does not match circuit {:?}",
                self.pedal.id, self.pedal.implementation, self.circuit.implementation
            )));
        }
        if self.params.schema != CIRCUIT_PARAMS_SCHEMA {
            return Err(CircuitArtifactError(format!(
                "{} params schema {:?} is unsupported",
                self.pedal.id, self.params.schema
            )));
        }
        if self.state.schema != CIRCUIT_STATE_SCHEMA {
            return Err(CircuitArtifactError(format!(
                "{} state schema {:?} is unsupported",
                self.pedal.id, self.state.schema
            )));
        }
        if self.params.circuit != self.circuit.id {
            return Err(CircuitArtifactError(format!(
                "{} params target {:?} does not match circuit {:?}",
                self.pedal.id, self.params.circuit, self.circuit.id
            )));
        }
        if self.state.circuit != self.circuit.id {
            return Err(CircuitArtifactError(format!(
                "{} state target {:?} does not match circuit {:?}",
                self.pedal.id, self.state.circuit, self.circuit.id
            )));
        }
        if self.params.refs.keys().collect::<BTreeSet<_>>()
            != self.circuit.parameters.refs.keys().collect::<BTreeSet<_>>()
        {
            return Err(CircuitArtifactError(format!(
                "{} params refs do not match circuit refs",
                self.pedal.id
            )));
        }
        if self
            .state
            .state_ports
            .iter()
            .map(|state| &state.id)
            .collect::<BTreeSet<_>>()
            != self
                .circuit
                .state_ports
                .iter()
                .map(|state| &state.id)
                .collect::<BTreeSet<_>>()
        {
            return Err(CircuitArtifactError(format!(
                "{} state ports do not match circuit state ports",
                self.pedal.id
            )));
        }
        Ok(())
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

    pub fn summary(&self) -> CircuitGraphSummary {
        let mut operator_counts = BTreeMap::new();
        let mut node_count = 0;
        let mut state_port_count = 0;
        let mut parameter_ref_count = 0;
        let mut static_state_elements = 0;
        let mut append_only_state_elements_per_activation = 0;

        for artifact in &self.circuits {
            *operator_counts
                .entry(artifact.pedal.operator_type.clone())
                .or_insert(0) += 1;
            node_count += artifact.circuit.nodes.len();
            state_port_count += artifact.circuit.state_ports.len();
            parameter_ref_count += artifact.circuit.parameters.refs.len();
            for state in &artifact.circuit.state_ports {
                static_state_elements += state.static_elements().unwrap_or(0);
                append_only_state_elements_per_activation +=
                    state.elements_per_activation().unwrap_or(0);
            }
        }

        CircuitGraphSummary {
            circuit_count: self.circuits.len(),
            operator_counts,
            node_count,
            state_port_count,
            parameter_ref_count,
            static_state_elements,
            append_only_state_elements_per_activation,
        }
    }

    pub fn node_operator_counts(&self) -> BTreeMap<String, usize> {
        let mut counts = BTreeMap::new();
        for artifact in &self.circuits {
            for node in &artifact.circuit.nodes {
                *counts.entry(node.op.clone()).or_insert(0) += 1;
            }
        }
        counts
    }

    pub fn state_type_counts(&self) -> BTreeMap<String, usize> {
        let mut counts = BTreeMap::new();
        for artifact in &self.circuits {
            for state in &artifact.circuit.state_ports {
                *counts.entry(state.state_type.clone()).or_insert(0) += 1;
            }
        }
        counts
    }

    pub fn capability_report(
        &self,
        capabilities: &StreamCircuitBackendCapabilities,
    ) -> CircuitCapabilityReport {
        let op_support = self
            .node_operator_counts()
            .into_iter()
            .map(|(op, count)| CircuitOpSupport {
                op: op.clone(),
                count,
                supported: capabilities.supported_ops.contains(&op),
            })
            .collect();
        let state_support = self
            .state_type_counts()
            .into_iter()
            .map(|(state_type, count)| CircuitStateSupport {
                state_type: state_type.clone(),
                count,
                supported: capabilities.supported_state_types.contains(&state_type),
            })
            .collect();

        CircuitCapabilityReport {
            backend_id: capabilities.backend_id.clone(),
            circuit_count: self.circuits.len(),
            wiring: self.index.graph.wiring.clone(),
            series_wiring_supported: self.index.graph.wiring == "series"
                && capabilities.supports_series_wiring,
            op_support,
            state_support,
        }
    }

    pub fn to_installed_processor_manifest(
        &self,
        install_id: impl Into<String>,
        backend: impl Into<String>,
    ) -> InstalledProcessorManifest {
        let first_input = self
            .circuits
            .first()
            .and_then(|artifact| artifact.circuit.boundary.inputs.first())
            .map(|port| port.signal.clone())
            .unwrap_or_else(|| "unknown".to_string());
        let last_output = self
            .circuits
            .last()
            .and_then(|artifact| artifact.circuit.boundary.outputs.first())
            .map(|port| port.signal.clone())
            .unwrap_or_else(|| "unknown".to_string());

        InstalledProcessorManifest {
            install_id: install_id.into(),
            backend: backend.into(),
            permanent_circuit: PermanentCircuitManifest {
                pedal_count: self.circuits.len(),
                input_signal: first_input,
                output_signal: last_output,
                source_model_dir: Some(self.index.source.source_model_dir.clone()),
            },
            host_ports: HostPortsManifest {
                inputs: vec![
                    "external_input".to_string(),
                    "control".to_string(),
                    "random_input".to_string(),
                ],
                outputs: vec!["public_output".to_string(), "events".to_string()],
                private_feedback: "device_owned_insert_loop".to_string(),
            },
            stream_template: StreamTemplate {
                id: "stream_circuit_template".to_string(),
                state_allocations: self
                    .circuits
                    .iter()
                    .flat_map(|artifact| {
                        artifact
                            .circuit
                            .state_ports
                            .iter()
                            .map(|state| StateAllocation {
                                pedal_id: artifact.pedal.id.clone(),
                                state_id: state.id.clone(),
                                state_type: state.state_type.clone(),
                                static_shape: state.shape.clone(),
                                elements_per_token: state.elements_per_activation(),
                            })
                    })
                    .collect(),
            },
            memory_plan: DeviceMemoryPlan {
                regions: vec![
                    MemoryRegion {
                        id: "source_tensor_refs".to_string(),
                        kind: MemoryRegionKind::PermanentParameters,
                        sharing: MemorySharing::SharedByAllStreams,
                        bytes: None,
                    },
                    MemoryRegion {
                        id: "stream_transient_state".to_string(),
                        kind: MemoryRegionKind::StreamTransientState,
                        sharing: MemorySharing::PerStream,
                        bytes: None,
                    },
                    MemoryRegion {
                        id: "input_queue".to_string(),
                        kind: MemoryRegionKind::InputQueue,
                        sharing: MemorySharing::HostVisibleQueue,
                        bytes: None,
                    },
                    MemoryRegion {
                        id: "output_queue".to_string(),
                        kind: MemoryRegionKind::OutputQueue,
                        sharing: MemorySharing::HostVisibleQueue,
                        bytes: None,
                    },
                ],
            },
        }
    }

    pub fn transient_allocations(&self) -> Vec<StreamCircuitStateAllocation> {
        self.circuits
            .iter()
            .flat_map(|artifact| {
                artifact
                    .circuit
                    .state_ports
                    .iter()
                    .map(|state| StreamCircuitStateAllocation {
                        pedal_id: artifact.pedal.id.clone(),
                        state_id: state.id.clone(),
                        state_type: state.state_type.clone(),
                        owner: state.owner.clone().unwrap_or_else(|| "stream".to_string()),
                        static_shape: state.shape.clone(),
                        elements_per_activation: state.elements_per_activation(),
                        layout: state.layout.clone(),
                    })
            })
            .collect()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CircuitGraphSummary {
    pub circuit_count: usize,
    pub operator_counts: BTreeMap<String, usize>,
    pub node_count: usize,
    pub state_port_count: usize,
    pub parameter_ref_count: usize,
    pub static_state_elements: usize,
    pub append_only_state_elements_per_activation: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StreamCircuitStateAllocation {
    pub pedal_id: String,
    pub state_id: String,
    pub state_type: String,
    pub owner: String,
    pub static_shape: Option<Vec<usize>>,
    pub elements_per_activation: Option<usize>,
    pub layout: Option<String>,
}

impl StreamCircuitStateAllocation {
    pub fn allocation_key(&self) -> String {
        format!("{}.{}", self.pedal_id, self.state_id)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StreamCircuitStreamTemplate {
    pub stream_id: String,
    pub allocations: Vec<StreamCircuitStateAllocation>,
}

impl StreamCircuitStreamTemplate {
    pub fn allocation_keys(&self) -> Vec<String> {
        self.allocations
            .iter()
            .map(StreamCircuitStateAllocation::allocation_key)
            .collect()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct InstalledStreamCircuit {
    pub graph: ResolvedLoweredPedalboard,
    pub manifest: InstalledProcessorManifest,
}

impl InstalledStreamCircuit {
    pub fn from_index_file(
        index_path: impl AsRef<Path>,
        install_id: impl Into<String>,
        backend: impl Into<String>,
    ) -> Result<Self, CircuitArtifactError> {
        let graph = ResolvedLoweredPedalboard::from_index_file(index_path)?;
        Ok(Self::new(graph, install_id, backend))
    }

    pub fn new(
        graph: ResolvedLoweredPedalboard,
        install_id: impl Into<String>,
        backend: impl Into<String>,
    ) -> Self {
        let manifest = graph.to_installed_processor_manifest(install_id, backend);
        Self { graph, manifest }
    }

    pub fn create_stream_template(
        &self,
        stream_id: impl Into<String>,
    ) -> StreamCircuitStreamTemplate {
        StreamCircuitStreamTemplate {
            stream_id: stream_id.into(),
            allocations: self.graph.transient_allocations(),
        }
    }

    pub fn capability_report(
        &self,
        capabilities: &StreamCircuitBackendCapabilities,
    ) -> CircuitCapabilityReport {
        self.graph.capability_report(capabilities)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StreamCircuitBackendCapabilities {
    pub backend_id: String,
    pub supports_series_wiring: bool,
    pub supported_ops: BTreeSet<String>,
    pub supported_state_types: BTreeSet<String>,
}

impl StreamCircuitBackendCapabilities {
    pub fn new(backend_id: impl Into<String>) -> Self {
        Self {
            backend_id: backend_id.into(),
            supports_series_wiring: true,
            supported_ops: BTreeSet::new(),
            supported_state_types: BTreeSet::new(),
        }
    }

    pub fn with_supported_ops<I, S>(mut self, ops: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.supported_ops = ops.into_iter().map(Into::into).collect();
        self
    }

    pub fn with_supported_state_types<I, S>(mut self, state_types: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.supported_state_types = state_types.into_iter().map(Into::into).collect();
        self
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CircuitOpSupport {
    pub op: String,
    pub count: usize,
    pub supported: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CircuitStateSupport {
    pub state_type: String,
    pub count: usize,
    pub supported: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CircuitCapabilityReport {
    pub backend_id: String,
    pub circuit_count: usize,
    pub wiring: String,
    pub series_wiring_supported: bool,
    pub op_support: Vec<CircuitOpSupport>,
    pub state_support: Vec<CircuitStateSupport>,
}

impl CircuitCapabilityReport {
    pub fn executable(&self) -> bool {
        self.series_wiring_supported
            && self.op_support.iter().all(|op| op.supported)
            && self.state_support.iter().all(|state| state.supported)
    }

    pub fn unsupported_ops(&self) -> Vec<String> {
        self.op_support
            .iter()
            .filter(|op| !op.supported)
            .map(|op| op.op.clone())
            .collect()
    }

    pub fn unsupported_state_types(&self) -> Vec<String> {
        self.state_support
            .iter()
            .filter(|state| !state.supported)
            .map(|state| state.state_type.clone())
            .collect()
    }
}

fn read_json<T: for<'de> Deserialize<'de>>(
    path: impl AsRef<Path>,
) -> Result<T, CircuitArtifactError> {
    let bytes = fs::read(path)?;
    Ok(serde_json::from_slice(&bytes)?)
}

fn resolve_artifact_path(artifact_root: &Path, path: &str) -> PathBuf {
    let path = Path::new(path);
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        artifact_root.join(path)
    }
}

fn product(shape: &[usize]) -> Option<usize> {
    shape
        .iter()
        .try_fold(1usize, |total, value| total.checked_mul(*value))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lfm2_index_path() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("lowered")
            .join("lfm2_5_230m")
            .join("pedalboard.circuits.json")
    }

    #[test]
    fn loads_lfm2_lowered_pedalboard_as_runtime_circuit_graph() {
        let resolved = ResolvedLoweredPedalboard::from_index_file(lfm2_index_path()).unwrap();
        let summary = resolved.summary();

        assert_eq!(resolved.index.schema, LOWERED_PEDALBOARD_SCHEMA);
        assert_eq!(resolved.index.graph.wiring, "series");
        assert_eq!(summary.circuit_count, 14);
        assert_eq!(summary.operator_counts.get("conv"), Some(&8));
        assert_eq!(summary.operator_counts.get("full_attention"), Some(&6));
        assert_eq!(summary.node_count, 242);
        assert_eq!(summary.state_port_count, 14);
        assert_eq!(summary.static_state_elements, 8 * 3 * 1024);
        assert_eq!(
            summary.append_only_state_elements_per_activation,
            6 * (8 * 64 + 8 * 64)
        );
        assert_eq!(
            resolved.circuits[0].circuit.implementation,
            "exact_lowering_lfm2_conv_layer_v1"
        );
        assert_eq!(
            resolved.circuits[2].circuit.state_ports[0].state_type,
            "append_only_attention_memory"
        );
    }

    #[test]
    fn lowered_pedalboard_can_describe_an_installed_processor_manifest() {
        let resolved = ResolvedLoweredPedalboard::from_index_file(lfm2_index_path()).unwrap();

        let manifest = resolved
            .to_installed_processor_manifest("lfm2_5_230m_stream_circuit", "stream_circuit_ir");

        assert_eq!(manifest.install_id, "lfm2_5_230m_stream_circuit");
        assert_eq!(manifest.backend, "stream_circuit_ir");
        assert_eq!(manifest.permanent_circuit.pedal_count, 14);
        assert_eq!(manifest.permanent_circuit.input_signal, "frame");
        assert_eq!(manifest.permanent_circuit.output_signal, "frame");
        assert_eq!(
            manifest.permanent_circuit.source_model_dir.as_deref(),
            Some("/home/aristath/models/lfm2.5/230m")
        );
        assert_eq!(manifest.stream_template.state_allocations.len(), 14);
        assert!(
            manifest
                .stream_template
                .state_allocations
                .iter()
                .any(|allocation| allocation.pedal_id == "layer_00"
                    && allocation.state_id == "temporal_memory"
                    && allocation.static_shape == Some(vec![3, 1024]))
        );
        assert!(
            manifest
                .stream_template
                .state_allocations
                .iter()
                .any(|allocation| allocation.pedal_id == "layer_02"
                    && allocation.state_id == "kv_memory"
                    && allocation.static_shape.is_none()
                    && allocation.elements_per_token == Some(1024))
        );
    }

    #[test]
    fn installed_stream_circuit_creates_stream_transient_template() {
        let installed = InstalledStreamCircuit::from_index_file(
            lfm2_index_path(),
            "lfm2_5_230m_stream_circuit",
            "stream_circuit_ir",
        )
        .unwrap();

        let stream = installed.create_stream_template("stream_a");
        let allocation_keys = stream.allocation_keys();

        assert_eq!(installed.manifest.permanent_circuit.pedal_count, 14);
        assert_eq!(stream.stream_id, "stream_a");
        assert_eq!(stream.allocations.len(), 14);
        assert_eq!(
            allocation_keys.first().map(String::as_str),
            Some("layer_00.temporal_memory")
        );
        assert_eq!(
            allocation_keys.get(2).map(String::as_str),
            Some("layer_02.kv_memory")
        );
        assert!(
            stream
                .allocations
                .iter()
                .all(|allocation| allocation.owner == "stream")
        );
        assert_eq!(
            stream
                .allocations
                .iter()
                .filter(|allocation| allocation.layout.as_deref() == Some("append_only_kv"))
                .count(),
            6
        );
    }

    #[test]
    fn stream_circuit_capability_report_names_missing_executors() {
        let installed = InstalledStreamCircuit::from_index_file(
            lfm2_index_path(),
            "lfm2_5_230m_stream_circuit",
            "stream_circuit_ir",
        )
        .unwrap();

        let report =
            installed.capability_report(&StreamCircuitBackendCapabilities::new("vulkan_spirv"));

        assert_eq!(report.backend_id, "vulkan_spirv");
        assert_eq!(report.circuit_count, 14);
        assert_eq!(report.wiring, "series");
        assert!(report.series_wiring_supported);
        assert!(!report.executable());
        assert_eq!(report.unsupported_ops().len(), 12);
        assert_eq!(
            report.unsupported_state_types(),
            vec![
                "append_only_attention_memory".to_string(),
                "rolling_frame_memory".to_string()
            ]
        );
        assert!(
            report
                .op_support
                .iter()
                .any(|op| op.op == "linear" && op.count == 82 && !op.supported)
        );
        assert!(
            report
                .op_support
                .iter()
                .any(|op| op.op == "scaled_dot_product_attention"
                    && op.count == 6
                    && !op.supported)
        );
    }

    #[test]
    fn stream_circuit_capability_report_can_mark_graph_executable() {
        let installed = InstalledStreamCircuit::from_index_file(
            lfm2_index_path(),
            "lfm2_5_230m_stream_circuit",
            "stream_circuit_ir",
        )
        .unwrap();
        let ops = installed.graph.node_operator_counts().into_keys();
        let state_types = installed.graph.state_type_counts().into_keys();
        let capabilities = StreamCircuitBackendCapabilities::new("future_backend")
            .with_supported_ops(ops)
            .with_supported_state_types(state_types);

        let report = installed.capability_report(&capabilities);

        assert!(report.executable());
        assert!(report.unsupported_ops().is_empty());
        assert!(report.unsupported_state_types().is_empty());
    }
}
