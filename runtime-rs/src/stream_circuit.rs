use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::backend::{BackendError, DeviceBackend};
use crate::stream_plan::{
    CircuitPlanError, StreamCircuitExecutionPlan, StreamCircuitResourcePlan, TensorIndex,
};
use crate::types::{
    ControlCommand, DeviceDispatchRun, DeviceMemoryPlan, DeviceOutputEvent, DispatchStatus,
    ForkPolicy, ForkRequest, HostPortsManifest, InputSignal, InstalledProcessorManifest,
    MemoryRegion, MemoryRegionKind, MemorySharing, PermanentCircuitManifest, PromptInjection,
    RandomPolicy, StateAllocation, StreamId, StreamTemplate,
};

pub const STREAM_CIRCUIT_SCHEMA: &str = "llmoop.stream_circuit.v1";
pub const CIRCUIT_PARAMS_SCHEMA: &str = "llmoop.circuit_params.v1";
pub const CIRCUIT_STATE_SCHEMA: &str = "llmoop.circuit_state.v1";
pub const LOWERED_PEDALBOARD_SCHEMA: &str = "llmoop.lowered_pedalboard.v1";
pub const STREAM_CIRCUIT_PLACEMENT_SCHEMA: &str = "llmoop.stream_circuit_placement.v1";
pub const STREAM_CIRCUIT_RUNTIME_PATCH_SCHEMA: &str = "llmoop.stream_circuit_runtime_patch.v1";
pub const RUNTIME_CABLE_ROUTES_SCHEMA: &str = "llmoop.runtime_cable_routes.v1";
pub const RUNTIME_DEVICE_BINDINGS_SCHEMA: &str = "llmoop.runtime_device_bindings.v1";

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
    pub format: String,
    pub artifact_root: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LoweredCircuitRef {
    pub id: String,
    pub operator_type: String,
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
pub struct CircuitPlacementError(pub String);

impl Display for CircuitPlacementError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for CircuitPlacementError {}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StreamCircuitPlacementSpec {
    pub schema: String,
    pub default_device_id: String,
    #[serde(default)]
    pub pedal_devices: BTreeMap<String, String>,
}

impl StreamCircuitPlacementSpec {
    pub fn new(default_device_id: impl Into<String>) -> Self {
        Self {
            schema: STREAM_CIRCUIT_PLACEMENT_SCHEMA.to_string(),
            default_device_id: default_device_id.into(),
            pedal_devices: BTreeMap::new(),
        }
    }

    pub fn with_pedal_device(
        mut self,
        pedal_id: impl Into<String>,
        device_id: impl Into<String>,
    ) -> Self {
        self.pedal_devices.insert(pedal_id.into(), device_id.into());
        self
    }

    pub fn device_for_pedal(&self, pedal_id: &str) -> &str {
        self.pedal_devices
            .get(pedal_id)
            .map(String::as_str)
            .unwrap_or(&self.default_device_id)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StreamCircuitRuntimePatch {
    pub schema: String,
    pub wiring: String,
    pub default_device_id: String,
    pub instances: Vec<StreamCircuitPedalInstance>,
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
        let patch = Self {
            schema: STREAM_CIRCUIT_RUNTIME_PATCH_SCHEMA.to_string(),
            wiring: graph.index.graph.wiring.clone(),
            default_device_id: default_device_id.into(),
            instances: chain
                .iter()
                .map(
                    |(instance_id, source_pedal_id)| StreamCircuitPedalInstance {
                        instance_id: instance_id.clone(),
                        source_pedal_id: source_pedal_id.clone(),
                        device_id: String::new(),
                        state_policy: StreamCircuitPedalInstanceStatePolicy::Fresh,
                    },
                )
                .collect(),
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
            wiring: graph.index.graph.wiring.clone(),
            default_device_id: spec.default_device_id.clone(),
            instances: graph
                .circuits
                .iter()
                .map(|artifact| StreamCircuitPedalInstance {
                    instance_id: artifact.pedal.id.clone(),
                    source_pedal_id: artifact.pedal.id.clone(),
                    device_id: spec.device_for_pedal(&artifact.pedal.id).to_string(),
                    state_policy: StreamCircuitPedalInstanceStatePolicy::Fresh,
                })
                .collect(),
        })
    }

    pub fn placement_spec(&self) -> StreamCircuitPlacementSpec {
        let mut spec = StreamCircuitPlacementSpec::new(self.default_device_id.clone());
        for instance in &self.instances {
            if instance.device_id != self.default_device_id {
                spec = spec.with_pedal_device(&instance.instance_id, &instance.device_id);
            }
        }
        spec
    }

    pub fn duplicate_after_instance(
        mut self,
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
            instance_id: new_instance_id,
            source_pedal_id: source.source_pedal_id,
            device_id: source.device_id,
            state_policy: StreamCircuitPedalInstanceStatePolicy::Fresh,
        };
        self.instances.insert(after_index + 1, duplicate);
        Ok(self)
    }

    pub fn with_source_chain(
        self,
        graph: &ResolvedLoweredPedalboard,
        chain: &[(String, String)],
    ) -> Result<Self, CircuitPlacementError> {
        let previous_devices = self
            .instances
            .iter()
            .map(|instance| (instance.instance_id.clone(), instance.device_id.clone()))
            .collect::<BTreeMap<_, _>>();
        let mut patch = Self::from_source_chain(graph, self.default_device_id, chain)?;
        for instance in &mut patch.instances {
            if let Some(device_id) = previous_devices.get(&instance.instance_id) {
                instance.device_id = device_id.clone();
            }
        }
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

        for instance in &self.instances {
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
        if self.wiring != "series" {
            return Err(CircuitPlacementError(format!(
                "only series runtime patches are currently planned, got {:?}",
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
        }

        for pair in self.instances.windows(2) {
            let source_instance = &pair[0];
            let destination_instance = &pair[1];
            let source = source_by_id[source_instance.source_pedal_id.as_str()];
            let destination = source_by_id[destination_instance.source_pedal_id.as_str()];
            validate_runtime_patch_cable(
                source_instance,
                source,
                destination_instance,
                destination,
            )?;
        }

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
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StreamCircuitPedalInstance {
    pub instance_id: String,
    pub source_pedal_id: String,
    pub device_id: String,
    pub state_policy: StreamCircuitPedalInstanceStatePolicy,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StreamCircuitPedalInstanceStatePolicy {
    Fresh,
    CloneFrom { instance_id: String },
    ShareWith { instance_id: String },
}

fn validate_runtime_patch_source_graph(
    graph: &ResolvedLoweredPedalboard,
) -> Result<(), CircuitPlacementError> {
    if graph.index.graph.wiring != "series" {
        return Err(CircuitPlacementError(format!(
            "only series runtime patches are currently planned, got {:?}",
            graph.index.graph.wiring
        )));
    }
    if graph.circuits.is_empty() {
        return Err(CircuitPlacementError(
            "cannot create runtime patch for an empty pedalboard".to_string(),
        ));
    }
    Ok(())
}

fn validate_placement_spec_against_graph(
    graph: &ResolvedLoweredPedalboard,
    spec: &StreamCircuitPlacementSpec,
) -> Result<(), CircuitPlacementError> {
    if spec.schema != STREAM_CIRCUIT_PLACEMENT_SCHEMA {
        return Err(CircuitPlacementError(format!(
            "unsupported stream-circuit placement schema {:?}",
            spec.schema
        )));
    }
    if spec.default_device_id.is_empty() {
        return Err(CircuitPlacementError(
            "placement default_device_id must not be empty".to_string(),
        ));
    }
    let pedal_ids = graph
        .circuits
        .iter()
        .map(|artifact| artifact.pedal.id.as_str())
        .collect::<BTreeSet<_>>();
    for pedal_id in spec.pedal_devices.keys() {
        if !pedal_ids.contains(pedal_id.as_str()) {
            return Err(CircuitPlacementError(format!(
                "placement references unknown pedal {pedal_id:?}"
            )));
        }
    }
    Ok(())
}

fn validate_runtime_patch_cable(
    source_instance: &StreamCircuitPedalInstance,
    source: &ResolvedCircuitArtifact,
    destination_instance: &StreamCircuitPedalInstance,
    destination: &ResolvedCircuitArtifact,
) -> Result<(), CircuitPlacementError> {
    let output = source.circuit.boundary.outputs.first().ok_or_else(|| {
        CircuitPlacementError(format!(
            "{} has no output port for runtime patch cable",
            source_instance.instance_id
        ))
    })?;
    let input = destination.circuit.boundary.inputs.first().ok_or_else(|| {
        CircuitPlacementError(format!(
            "{} has no input port for runtime patch cable",
            destination_instance.instance_id
        ))
    })?;
    if output.signal != input.signal || output.shape != input.shape {
        return Err(CircuitPlacementError(format!(
            "cannot patch cable {} -> {} without an adapter: output {:?}/{:?}, input {:?}/{:?}",
            source_instance.instance_id,
            destination_instance.instance_id,
            output.signal,
            output.shape,
            input.signal,
            input.shape
        )));
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StreamCircuitPlacementPlan {
    pub schema: String,
    pub wiring: String,
    pub pedals: Vec<PedalPlacement>,
    pub cables: Vec<PedalCablePlacement>,
    pub local_cable_count: usize,
    pub cross_device_cable_count: usize,
}

impl StreamCircuitPlacementPlan {
    pub fn from_graph(
        graph: &ResolvedLoweredPedalboard,
        spec: &StreamCircuitPlacementSpec,
    ) -> Result<Self, CircuitPlacementError> {
        if spec.schema != STREAM_CIRCUIT_PLACEMENT_SCHEMA {
            return Err(CircuitPlacementError(format!(
                "unsupported stream-circuit placement schema {:?}",
                spec.schema
            )));
        }
        if spec.default_device_id.is_empty() {
            return Err(CircuitPlacementError(
                "placement default_device_id must not be empty".to_string(),
            ));
        }
        if graph.index.graph.wiring != "series" {
            return Err(CircuitPlacementError(format!(
                "only series placement is currently planned, got {:?}",
                graph.index.graph.wiring
            )));
        }

        let pedal_ids: BTreeSet<_> = graph
            .circuits
            .iter()
            .map(|artifact| artifact.pedal.id.as_str())
            .collect();
        for pedal_id in spec.pedal_devices.keys() {
            if !pedal_ids.contains(pedal_id.as_str()) {
                return Err(CircuitPlacementError(format!(
                    "placement references unknown pedal {pedal_id:?}"
                )));
            }
        }

        let pedals = graph
            .circuits
            .iter()
            .enumerate()
            .map(|(pedal_index, artifact)| PedalPlacement {
                pedal_index,
                pedal_id: artifact.pedal.id.clone(),
                circuit_id: artifact.circuit.id.clone(),
                operator_type: artifact.pedal.operator_type.clone(),
                device_id: spec.device_for_pedal(&artifact.pedal.id).to_string(),
            })
            .collect::<Vec<_>>();

        let mut cables = Vec::with_capacity(graph.circuits.len().saturating_sub(1));
        let mut local_cable_count = 0usize;
        let mut cross_device_cable_count = 0usize;
        for (cable_index, pair) in graph.circuits.windows(2).enumerate() {
            let source = &pair[0];
            let destination = &pair[1];
            let output = source.circuit.boundary.outputs.first().ok_or_else(|| {
                CircuitPlacementError(format!(
                    "{} has no output port for placement cable",
                    source.pedal.id
                ))
            })?;
            let input = destination.circuit.boundary.inputs.first().ok_or_else(|| {
                CircuitPlacementError(format!(
                    "{} has no input port for placement cable",
                    destination.pedal.id
                ))
            })?;
            if output.signal != input.signal || output.shape != input.shape {
                return Err(CircuitPlacementError(format!(
                    "cannot place cable {} -> {} without an adapter: output {:?}/{:?}, input {:?}/{:?}",
                    source.pedal.id,
                    destination.pedal.id,
                    output.signal,
                    output.shape,
                    input.signal,
                    input.shape
                )));
            }

            let source_device_id = spec.device_for_pedal(&source.pedal.id).to_string();
            let destination_device_id = spec.device_for_pedal(&destination.pedal.id).to_string();
            let transport = if source_device_id == destination_device_id {
                local_cable_count += 1;
                CableTransport::LocalBuffer {
                    device_id: source_device_id.clone(),
                }
            } else {
                cross_device_cable_count += 1;
                CableTransport::CrossDevice {
                    from_device_id: source_device_id.clone(),
                    to_device_id: destination_device_id.clone(),
                }
            };

            cables.push(PedalCablePlacement {
                cable_index,
                signal: output.signal.clone(),
                shape: output.shape.clone(),
                source_pedal_id: source.pedal.id.clone(),
                source_device_id,
                source_port_id: output.id.clone(),
                source_pedal_port: output.pedal_port.clone(),
                destination_pedal_id: destination.pedal.id.clone(),
                destination_device_id,
                destination_port_id: input.id.clone(),
                destination_pedal_port: input.pedal_port.clone(),
                transport,
            });
        }

        Ok(Self {
            schema: STREAM_CIRCUIT_PLACEMENT_SCHEMA.to_string(),
            wiring: graph.index.graph.wiring.clone(),
            pedals,
            cables,
            local_cable_count,
            cross_device_cable_count,
        })
    }

    pub fn pedal(&self, pedal_id: &str) -> Option<&PedalPlacement> {
        self.pedals.iter().find(|pedal| pedal.pedal_id == pedal_id)
    }

    pub fn cross_device_cables(&self) -> Vec<&PedalCablePlacement> {
        self.cables
            .iter()
            .filter(|cable| cable.transport.is_cross_device())
            .collect()
    }

    pub fn runtime_cable_routes<F>(&self, target_for: F) -> RuntimeCableRoutes
    where
        F: FnMut(&str) -> RuntimeCableRouteTarget,
    {
        RuntimeCableRoutes::from_cables(&self.cables, target_for)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PedalPlacement {
    pub pedal_index: usize,
    pub pedal_id: String,
    pub circuit_id: String,
    pub operator_type: String,
    pub device_id: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PedalCablePlacement {
    pub cable_index: usize,
    pub signal: String,
    pub shape: Vec<usize>,
    pub source_pedal_id: String,
    pub source_device_id: String,
    pub source_port_id: String,
    pub source_pedal_port: Option<String>,
    pub destination_pedal_id: String,
    pub destination_device_id: String,
    pub destination_port_id: String,
    pub destination_pedal_port: Option<String>,
    pub transport: CableTransport,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CableTransport {
    LocalBuffer {
        device_id: String,
    },
    CrossDevice {
        from_device_id: String,
        to_device_id: String,
    },
}

impl CableTransport {
    pub fn is_cross_device(&self) -> bool {
        matches!(self, Self::CrossDevice { .. })
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeCableRouteTarget {
    pub target: Option<String>,
    pub physical_device_index: Option<usize>,
    pub binding_source: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeCableRouteKind {
    LogicalLocal,
    SamePhysicalTarget,
    CrossPhysicalTarget,
    UnresolvedRuntimeTarget,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeCableRoute {
    pub cable_index: usize,
    pub signal: String,
    pub shape: Vec<usize>,
    pub source_pedal_id: String,
    pub source_device_id: String,
    pub source_target: Option<String>,
    pub source_physical_device_index: Option<usize>,
    pub source_binding: String,
    pub destination_pedal_id: String,
    pub destination_device_id: String,
    pub destination_target: Option<String>,
    pub destination_physical_device_index: Option<usize>,
    pub destination_binding: String,
    pub route_kind: RuntimeCableRouteKind,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeCableRoutes {
    pub schema: String,
    pub cable_count: usize,
    pub logical_local_cable_count: usize,
    pub logical_cross_device_cable_count: usize,
    pub same_physical_target_cable_count: usize,
    pub cross_physical_target_cable_count: usize,
    pub unresolved_target_cable_count: usize,
    pub routes: Vec<RuntimeCableRoute>,
}

impl RuntimeCableRoutes {
    pub fn from_cables<F>(cables: &[PedalCablePlacement], mut target_for: F) -> Self
    where
        F: FnMut(&str) -> RuntimeCableRouteTarget,
    {
        let mut logical_local_cable_count = 0usize;
        let mut logical_cross_device_cable_count = 0usize;
        let mut same_physical_target_cable_count = 0usize;
        let mut cross_physical_target_cable_count = 0usize;
        let mut unresolved_target_cable_count = 0usize;

        let routes = cables
            .iter()
            .map(|cable| {
                let source_target = target_for(&cable.source_device_id);
                let destination_target = target_for(&cable.destination_device_id);
                let is_logical_local = cable.source_device_id == cable.destination_device_id;
                let route_kind = if is_logical_local {
                    logical_local_cable_count += 1;
                    RuntimeCableRouteKind::LogicalLocal
                } else {
                    logical_cross_device_cable_count += 1;
                    match (&source_target.target, &destination_target.target) {
                        (Some(source), Some(destination)) if source == destination => {
                            same_physical_target_cable_count += 1;
                            RuntimeCableRouteKind::SamePhysicalTarget
                        }
                        (Some(_), Some(_)) => {
                            cross_physical_target_cable_count += 1;
                            RuntimeCableRouteKind::CrossPhysicalTarget
                        }
                        _ => {
                            unresolved_target_cable_count += 1;
                            RuntimeCableRouteKind::UnresolvedRuntimeTarget
                        }
                    }
                };

                RuntimeCableRoute {
                    cable_index: cable.cable_index,
                    signal: cable.signal.clone(),
                    shape: cable.shape.clone(),
                    source_pedal_id: cable.source_pedal_id.clone(),
                    source_device_id: cable.source_device_id.clone(),
                    source_target: source_target.target,
                    source_physical_device_index: source_target.physical_device_index,
                    source_binding: source_target.binding_source,
                    destination_pedal_id: cable.destination_pedal_id.clone(),
                    destination_device_id: cable.destination_device_id.clone(),
                    destination_target: destination_target.target,
                    destination_physical_device_index: destination_target.physical_device_index,
                    destination_binding: destination_target.binding_source,
                    route_kind,
                }
            })
            .collect::<Vec<_>>();

        Self {
            schema: RUNTIME_CABLE_ROUTES_SCHEMA.to_string(),
            cable_count: cables.len(),
            logical_local_cable_count,
            logical_cross_device_cable_count,
            same_physical_target_cable_count,
            cross_physical_target_cable_count,
            unresolved_target_cable_count,
            routes,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeLogicalDeviceBinding {
    pub device_id: String,
    pub target: Option<String>,
    pub binding_source: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeDeviceBindings {
    pub schema: String,
    pub process_vulkan_device_index: Option<usize>,
    pub requested_vulkan_device_indices: Vec<usize>,
    pub default_vulkan_device_index: Option<usize>,
    pub explicit_bindings: BTreeMap<String, String>,
    pub logical_devices: Vec<RuntimeLogicalDeviceBinding>,
    pub can_mount_in_process: bool,
    pub mounting_model: String,
    pub unsupported_targets: Vec<String>,
    pub notes: Vec<String>,
}

impl RuntimeDeviceBindings {
    pub fn from_vulkan_targets<F>(
        logical_device_ids: &[String],
        explicit_bindings: &BTreeMap<String, String>,
        default_vulkan_device_index: Option<usize>,
        mut vulkan_physical_device_index_for_target: F,
    ) -> Self
    where
        F: FnMut(&str) -> Result<Option<usize>, String>,
    {
        let mut logical_ids = logical_device_ids.to_vec();
        for logical_device_id in explicit_bindings.keys() {
            if !logical_ids.contains(logical_device_id) {
                logical_ids.push(logical_device_id.clone());
            }
        }
        logical_ids.sort();
        logical_ids.dedup();

        let mut vulkan_indices = Vec::new();
        let mut unsupported_targets = Vec::new();
        if let Some(index) = default_vulkan_device_index {
            vulkan_indices.push(index);
        }
        for (logical_device_id, target) in explicit_bindings {
            match vulkan_physical_device_index_for_target(target) {
                Ok(Some(index)) => vulkan_indices.push(index),
                Ok(None) => unsupported_targets.push(format!("{logical_device_id}={target}")),
                Err(error) => {
                    unsupported_targets.push(format!("{logical_device_id}={target} ({error})"))
                }
            }
        }
        for logical_device_id in &logical_ids {
            if explicit_bindings.contains_key(logical_device_id) {
                continue;
            }
            match vulkan_physical_device_index_for_target(logical_device_id) {
                Ok(Some(index)) => vulkan_indices.push(index),
                Ok(None) if logical_device_id.contains(':') => {
                    unsupported_targets.push(logical_device_id.clone())
                }
                Err(error) if logical_device_id.contains(':') => {
                    unsupported_targets.push(format!("{logical_device_id} ({error})"))
                }
                Ok(None) | Err(_) => {}
            }
        }
        vulkan_indices.sort_unstable();
        vulkan_indices.dedup();
        unsupported_targets.sort();
        unsupported_targets.dedup();

        let logical_devices = logical_ids
            .iter()
            .map(|logical_device_id| {
                let explicit_target = explicit_bindings.get(logical_device_id);
                let direct_target = if explicit_target.is_none() {
                    match vulkan_physical_device_index_for_target(logical_device_id) {
                        Ok(Some(_)) => Some(logical_device_id.clone()),
                        Ok(None) | Err(_) if logical_device_id.contains(':') => {
                            Some(logical_device_id.clone())
                        }
                        Ok(None) | Err(_) => None,
                    }
                } else {
                    None
                };
                let target = explicit_target
                    .cloned()
                    .or(direct_target)
                    .or_else(|| default_vulkan_device_index.map(|index| format!("vulkan:{index}")));
                let binding_source = if explicit_target.is_some() {
                    "explicit"
                } else if target.as_deref() == Some(logical_device_id.as_str())
                    && logical_device_id.contains(':')
                {
                    "device_id"
                } else if default_vulkan_device_index.is_some() {
                    "process_default"
                } else {
                    "runtime_default"
                };
                RuntimeLogicalDeviceBinding {
                    device_id: logical_device_id.clone(),
                    target,
                    binding_source: binding_source.to_string(),
                }
            })
            .collect::<Vec<_>>();

        let can_mount_in_process = unsupported_targets.is_empty();

        Self {
            schema: RUNTIME_DEVICE_BINDINGS_SCHEMA.to_string(),
            process_vulkan_device_index: default_vulkan_device_index,
            requested_vulkan_device_indices: vulkan_indices,
            default_vulkan_device_index,
            explicit_bindings: explicit_bindings.clone(),
            logical_devices,
            can_mount_in_process,
            mounting_model: if can_mount_in_process {
                "local_vulkan_device_pool".to_string()
            } else {
                "unsupported_targets".to_string()
            },
            unsupported_targets,
            notes: if can_mount_in_process {
                vec![
                    "mounted logical device slices can use distinct local Vulkan physical devices in this runtime process"
                        .to_string(),
                ]
            } else {
                vec![
                    "only local vulkan:N targets are mountable by this runtime process".to_string(),
                ]
            },
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeBoundDevice {
    pub device_id: String,
    pub target: Option<String>,
    pub physical_device_index: Option<usize>,
    pub device_name: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimePedalPortSummary {
    pub id: String,
    pub signal: String,
    pub shape: Vec<usize>,
    pub source: Option<String>,
    pub pedal_port: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeSourcePedal {
    pub pedal_index: usize,
    pub pedal_id: String,
    pub operator_type: String,
    pub implementation: String,
    pub behavioral_role: String,
    pub source_layer_index: usize,
    pub circuit_id: String,
    pub input_ports: Vec<RuntimePedalPortSummary>,
    pub output_ports: Vec<RuntimePedalPortSummary>,
    pub state_port_count: usize,
    pub parameter_ref_count: usize,
    pub node_count: usize,
    pub kernel_count: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeCapacityProfileSummary {
    pub min_dynamic_state_capacity_activations: usize,
    pub max_dynamic_state_capacity_activations: usize,
    pub shader_override_count: usize,
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

    pub fn execution_plan(&self) -> Result<StreamCircuitExecutionPlan, CircuitPlanError> {
        StreamCircuitExecutionPlan::from_graph(&self.graph)
    }

    pub fn resource_plan(&self) -> Result<StreamCircuitResourcePlan, CircuitPlanError> {
        let execution_plan = self.execution_plan()?;
        StreamCircuitResourcePlan::from_graph_and_plan(&self.graph, &execution_plan)
    }

    pub fn execution_plan_with_tensor_index(
        &self,
        tensor_index: &TensorIndex,
    ) -> Result<StreamCircuitExecutionPlan, CircuitPlanError> {
        StreamCircuitExecutionPlan::from_graph_with_tensor_index(&self.graph, tensor_index)
    }

    pub fn resource_plan_with_tensor_index(
        &self,
        tensor_index: &TensorIndex,
    ) -> Result<StreamCircuitResourcePlan, CircuitPlanError> {
        let execution_plan = self.execution_plan_with_tensor_index(tensor_index)?;
        StreamCircuitResourcePlan::from_graph_and_plan(&self.graph, &execution_plan)
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StreamCircuitRuntimeError {
    Backend(BackendError),
    UnsupportedExecutionPlan(CircuitCapabilityReport),
    ExecutionUnavailable(String),
}

impl Display for StreamCircuitRuntimeError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Backend(error) => Display::fmt(error, f),
            Self::UnsupportedExecutionPlan(report) => write!(
                f,
                "stream circuit backend {:?} cannot execute graph yet; unsupported ops: {:?}; unsupported state types: {:?}",
                report.backend_id,
                report.unsupported_ops(),
                report.unsupported_state_types()
            ),
            Self::ExecutionUnavailable(message) => f.write_str(message),
        }
    }
}

impl Error for StreamCircuitRuntimeError {}

impl From<BackendError> for StreamCircuitRuntimeError {
    fn from(error: BackendError) -> Self {
        Self::Backend(error)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MountedStreamCircuit {
    pub template: StreamCircuitStreamTemplate,
    pending_external: VecDeque<InputSignal>,
    remaining_public_outputs: u32,
    input_counter: u64,
}

impl MountedStreamCircuit {
    fn new(installed: &InstalledStreamCircuit, stream_id: impl Into<String>) -> Self {
        Self {
            template: installed.create_stream_template(stream_id),
            pending_external: VecDeque::new(),
            remaining_public_outputs: 0,
            input_counter: 0,
        }
    }

    fn has_work(&self) -> bool {
        !self.pending_external.is_empty()
    }

    fn reset(&mut self) {
        self.pending_external.clear();
        self.remaining_public_outputs = 0;
        self.input_counter = 0;
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct StreamCircuitDeviceBackend {
    device_id: String,
    installed: InstalledStreamCircuit,
    capabilities: StreamCircuitBackendCapabilities,
    capability_report: CircuitCapabilityReport,
    streams: BTreeMap<StreamId, MountedStreamCircuit>,
    active_queue: VecDeque<StreamId>,
    output_queue: Vec<DeviceOutputEvent>,
}

impl StreamCircuitDeviceBackend {
    pub const BACKEND_ID: &'static str = "stream_circuit_ir";

    pub fn new(
        device_id: impl Into<String>,
        installed: InstalledStreamCircuit,
        capabilities: StreamCircuitBackendCapabilities,
    ) -> Self {
        let capability_report = installed.capability_report(&capabilities);
        Self {
            device_id: device_id.into(),
            installed,
            capabilities,
            capability_report,
            streams: BTreeMap::new(),
            active_queue: VecDeque::new(),
            output_queue: Vec::new(),
        }
    }

    pub fn from_index_file(
        device_id: impl Into<String>,
        index_path: impl AsRef<Path>,
        capabilities: StreamCircuitBackendCapabilities,
    ) -> Result<Self, CircuitArtifactError> {
        let installed = InstalledStreamCircuit::from_index_file(
            index_path,
            "stream_circuit_installed_processor",
            capabilities.backend_id.clone(),
        )?;
        Ok(Self::new(device_id, installed, capabilities))
    }

    pub fn capability_report(&self) -> &CircuitCapabilityReport {
        &self.capability_report
    }

    pub fn stream_template(&self, stream_id: &str) -> Option<&StreamCircuitStreamTemplate> {
        self.streams.get(stream_id).map(|stream| &stream.template)
    }

    pub fn execution_plan(&self) -> Result<StreamCircuitExecutionPlan, CircuitPlanError> {
        self.installed.execution_plan()
    }

    pub fn resource_plan(&self) -> Result<StreamCircuitResourcePlan, CircuitPlanError> {
        self.installed.resource_plan()
    }

    pub fn execution_plan_with_tensor_index(
        &self,
        tensor_index: &TensorIndex,
    ) -> Result<StreamCircuitExecutionPlan, CircuitPlanError> {
        self.installed
            .execution_plan_with_tensor_index(tensor_index)
    }

    pub fn resource_plan_with_tensor_index(
        &self,
        tensor_index: &TensorIndex,
    ) -> Result<StreamCircuitResourcePlan, CircuitPlanError> {
        self.installed.resource_plan_with_tensor_index(tensor_index)
    }

    pub fn capabilities(&self) -> &StreamCircuitBackendCapabilities {
        &self.capabilities
    }

    fn stream_mut(&mut self, stream_id: &str) -> Result<&mut MountedStreamCircuit, BackendError> {
        self.streams
            .get_mut(stream_id)
            .ok_or_else(|| BackendError::UnknownStream(stream_id.to_string()))
    }

    fn schedule(&mut self, stream_id: &str) {
        if !self.active_queue.iter().any(|active| active == stream_id) {
            self.active_queue.push_back(stream_id.to_string());
        }
    }

    fn deschedule_if_idle(&mut self, stream_id: &str) {
        if self
            .streams
            .get(stream_id)
            .is_some_and(MountedStreamCircuit::has_work)
        {
            return;
        }
        self.active_queue.retain(|active| active != stream_id);
    }
}

impl DeviceBackend for StreamCircuitDeviceBackend {
    type Error = StreamCircuitRuntimeError;

    fn backend_id(&self) -> &str {
        Self::BACKEND_ID
    }

    fn device_id(&self) -> &str {
        &self.device_id
    }

    fn has_stream(&self, stream_id: &str) -> bool {
        self.streams.contains_key(stream_id)
    }

    fn create_stream(&mut self, stream_id: &str) -> Result<(), Self::Error> {
        if self.has_stream(stream_id) {
            return Err(BackendError::DuplicateStream(stream_id.to_string()).into());
        }
        self.streams.insert(
            stream_id.to_string(),
            MountedStreamCircuit::new(&self.installed, stream_id),
        );
        Ok(())
    }

    fn inject_prompt(&mut self, injection: PromptInjection) -> Result<(), Self::Error> {
        let stream_id = injection.stream_id.clone();
        let stream = self.stream_mut(&stream_id)?;
        for token_id in injection.prompt_ids {
            let signal = InputSignal::external(
                format!("input_{}", stream.input_counter),
                token_id,
                injection.origin.clone(),
            );
            stream.input_counter += 1;
            stream.pending_external.push_back(signal);
        }
        stream.remaining_public_outputs = stream
            .remaining_public_outputs
            .saturating_add(injection.max_new_tokens);
        self.schedule(&stream_id);
        Ok(())
    }

    fn inject_token(&mut self, stream_id: &str, signal: InputSignal) -> Result<(), Self::Error> {
        self.stream_mut(stream_id)?
            .pending_external
            .push_back(signal);
        self.schedule(stream_id);
        Ok(())
    }

    fn control(&mut self, stream_id: &str, command: ControlCommand) -> Result<(), Self::Error> {
        let stream = self.stream_mut(stream_id)?;
        match command {
            ControlCommand::Continue {
                additional_public_outputs,
                ..
            } => {
                stream.remaining_public_outputs = stream
                    .remaining_public_outputs
                    .saturating_add(additional_public_outputs);
            }
            ControlCommand::Interrupt { .. } | ControlCommand::StopAfterCurrent { .. } => {
                stream.remaining_public_outputs = 0;
            }
            ControlCommand::ResetState { .. } => {
                stream.reset();
            }
            ControlCommand::ReseedRandom { .. } => {}
        }
        if stream.has_work() {
            self.schedule(stream_id);
        } else {
            self.deschedule_if_idle(stream_id);
        }
        Ok(())
    }

    fn fork_stream(&mut self, request: ForkRequest) -> Result<(), Self::Error> {
        if self.has_stream(&request.child_stream_id) {
            return Err(BackendError::DuplicateStream(request.child_stream_id).into());
        }
        let mut child = match request.state_policy {
            ForkPolicy::Clone => self
                .streams
                .get(&request.parent_stream_id)
                .ok_or_else(|| BackendError::UnknownStream(request.parent_stream_id.clone()))?
                .clone(),
            ForkPolicy::Fresh => {
                MountedStreamCircuit::new(&self.installed, &request.child_stream_id)
            }
        };
        child.template.stream_id = request.child_stream_id.clone();
        let child_has_work = child.has_work();
        self.streams.insert(request.child_stream_id.clone(), child);
        if request.random_policy == RandomPolicy::Fresh {
            // Random queues are part of the circuit contract; this shell has no sampler executor yet.
        }
        if child_has_work {
            self.schedule(&request.child_stream_id);
        }
        Ok(())
    }

    fn dispatch(&mut self, max_ticks: u32) -> Result<DeviceDispatchRun, Self::Error> {
        if self.active_queue.is_empty() {
            return Ok(DeviceDispatchRun {
                device_id: self.device_id.clone(),
                ticks: Vec::new(),
                outputs: Vec::new(),
                status: DispatchStatus::Idle,
                active_streams: Vec::new(),
            });
        }
        if max_ticks == 0 {
            return Ok(DeviceDispatchRun {
                device_id: self.device_id.clone(),
                ticks: Vec::new(),
                outputs: Vec::new(),
                status: DispatchStatus::BudgetExhausted,
                active_streams: self.active_queue.iter().cloned().collect(),
            });
        }
        if !self.capability_report.executable() {
            return Err(StreamCircuitRuntimeError::UnsupportedExecutionPlan(
                self.capability_report.clone(),
            ));
        }
        Err(StreamCircuitRuntimeError::ExecutionUnavailable(
            "stream circuit executor registry is not installed".to_string(),
        ))
    }

    fn drain_outputs(&mut self) -> Result<Vec<DeviceOutputEvent>, Self::Error> {
        Ok(std::mem::take(&mut self.output_queue))
    }

    fn describe(&self) -> InstalledProcessorManifest {
        self.installed.manifest.clone()
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
    fn loads_fixture_model_lowered_pedalboard_as_runtime_circuit_graph() {
        let resolved =
            ResolvedLoweredPedalboard::from_index_file(fixture_model_index_path()).unwrap();
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
            "reference_shortconv_layer_circuit_v1"
        );
        assert_eq!(
            resolved.circuits[2].circuit.state_ports[0].state_type,
            "append_only_attention_memory"
        );
    }

    #[test]
    fn lowered_pedalboard_can_describe_an_installed_processor_manifest() {
        let resolved =
            ResolvedLoweredPedalboard::from_index_file(fixture_model_index_path()).unwrap();

        let manifest = resolved
            .to_installed_processor_manifest("compiled_model_stream_circuit", "stream_circuit_ir");

        assert_eq!(manifest.install_id, "compiled_model_stream_circuit");
        assert_eq!(manifest.backend, "stream_circuit_ir");
        assert_eq!(manifest.permanent_circuit.pedal_count, 14);
        assert_eq!(manifest.permanent_circuit.input_signal, "frame");
        assert_eq!(manifest.permanent_circuit.output_signal, "frame");
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
            fixture_model_index_path(),
            "compiled_model_stream_circuit",
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
    fn placement_plan_keeps_layer_pedals_as_deployable_units() {
        let resolved =
            ResolvedLoweredPedalboard::from_index_file(fixture_model_index_path()).unwrap();

        let placement = resolved.single_device_placement_plan("gpu0").unwrap();

        assert_eq!(placement.schema, STREAM_CIRCUIT_PLACEMENT_SCHEMA);
        assert_eq!(placement.wiring, "series");
        assert_eq!(placement.pedals.len(), 14);
        assert_eq!(placement.cables.len(), 13);
        assert_eq!(placement.local_cable_count, 13);
        assert_eq!(placement.cross_device_cable_count, 0);
        assert_eq!(
            placement.pedal("layer_00").unwrap(),
            &PedalPlacement {
                pedal_index: 0,
                pedal_id: "layer_00".to_string(),
                circuit_id: "layer_00_shortconv_circuit_v1".to_string(),
                operator_type: "conv".to_string(),
                device_id: "gpu0".to_string(),
            }
        );

        let first_cable = &placement.cables[0];
        assert_eq!(first_cable.source_pedal_id, "layer_00");
        assert_eq!(first_cable.destination_pedal_id, "layer_01");
        assert_eq!(first_cable.signal, "frame");
        assert_eq!(first_cable.shape, vec![1024]);
        assert_eq!(first_cable.source_port_id, "output_frame");
        assert_eq!(first_cable.destination_port_id, "input_frame");
        assert_eq!(first_cable.source_pedal_port.as_deref(), Some("output"));
        assert_eq!(first_cable.destination_pedal_port.as_deref(), Some("input"));
        assert_eq!(
            first_cable.transport,
            CableTransport::LocalBuffer {
                device_id: "gpu0".to_string(),
            }
        );
    }

    #[test]
    fn placement_plan_changes_cables_not_pedalboard_when_devices_differ() {
        let resolved =
            ResolvedLoweredPedalboard::from_index_file(fixture_model_index_path()).unwrap();
        let spec = StreamCircuitPlacementSpec::new("gpu0")
            .with_pedal_device("layer_01", "cpu0")
            .with_pedal_device("layer_02", "gpu1")
            .with_pedal_device("layer_03", "lan:worker-a");

        let placement = resolved.placement_plan(&spec).unwrap();

        assert_eq!(placement.pedals.len(), 14);
        assert_eq!(placement.cables.len(), 13);
        assert_eq!(placement.local_cable_count, 9);
        assert_eq!(placement.cross_device_cable_count, 4);
        assert_eq!(
            placement
                .pedal("layer_01")
                .map(|pedal| pedal.device_id.as_str()),
            Some("cpu0")
        );
        assert_eq!(
            placement
                .pedal("layer_02")
                .map(|pedal| pedal.device_id.as_str()),
            Some("gpu1")
        );
        assert_eq!(
            placement
                .pedal("layer_03")
                .map(|pedal| pedal.device_id.as_str()),
            Some("lan:worker-a")
        );
        assert_eq!(
            placement
                .pedal("layer_04")
                .map(|pedal| pedal.device_id.as_str()),
            Some("gpu0")
        );

        let cross = placement.cross_device_cables();
        assert_eq!(cross.len(), 4);
        assert_eq!(
            cross
                .iter()
                .map(|cable| (
                    cable.source_pedal_id.as_str(),
                    cable.source_device_id.as_str(),
                    cable.destination_pedal_id.as_str(),
                    cable.destination_device_id.as_str()
                ))
                .collect::<Vec<_>>(),
            vec![
                ("layer_00", "gpu0", "layer_01", "cpu0"),
                ("layer_01", "cpu0", "layer_02", "gpu1"),
                ("layer_02", "gpu1", "layer_03", "lan:worker-a"),
                ("layer_03", "lan:worker-a", "layer_04", "gpu0"),
            ]
        );
        assert_eq!(
            cross[2].transport,
            CableTransport::CrossDevice {
                from_device_id: "gpu1".to_string(),
                to_device_id: "lan:worker-a".to_string(),
            }
        );
    }

    #[test]
    fn runtime_cable_routes_classify_logical_and_physical_routes() {
        let resolved =
            ResolvedLoweredPedalboard::from_index_file(fixture_model_index_path()).unwrap();
        let spec = StreamCircuitPlacementSpec::new("gpu0")
            .with_pedal_device("layer_01", "gpu1")
            .with_pedal_device("layer_02", "gpu2");
        let placement = resolved.placement_plan(&spec).unwrap();

        let routes = placement.runtime_cable_routes(|device_id| {
            let (target, physical_device_index) = match device_id {
                "gpu0" | "gpu1" => (Some("vulkan:0".to_string()), Some(0)),
                "gpu2" => (Some("vulkan:1".to_string()), Some(1)),
                _ => (None, None),
            };
            RuntimeCableRouteTarget {
                target,
                physical_device_index,
                binding_source: "test".to_string(),
            }
        });

        assert_eq!(routes.schema, RUNTIME_CABLE_ROUTES_SCHEMA);
        assert_eq!(routes.cable_count, 13);
        assert_eq!(routes.logical_local_cable_count, 10);
        assert_eq!(routes.logical_cross_device_cable_count, 3);
        assert_eq!(routes.same_physical_target_cable_count, 1);
        assert_eq!(routes.cross_physical_target_cable_count, 2);
        assert_eq!(routes.unresolved_target_cable_count, 0);
        assert_eq!(
            routes.routes[0].route_kind,
            RuntimeCableRouteKind::SamePhysicalTarget
        );
        assert_eq!(
            routes.routes[1].route_kind,
            RuntimeCableRouteKind::CrossPhysicalTarget
        );
        assert_eq!(
            routes.routes[3].route_kind,
            RuntimeCableRouteKind::LogicalLocal
        );

        let payload = serde_json::to_value(&routes).unwrap();
        assert_eq!(payload["routes"][0]["route_kind"], "same_physical_target");
        assert_eq!(payload["routes"][1]["route_kind"], "cross_physical_target");
        assert_eq!(payload["routes"][3]["route_kind"], "logical_local");
    }

    #[test]
    fn runtime_device_bindings_capture_runtime_target_contract() {
        let logical_device_ids = vec![
            "gpu0".to_string(),
            "gpu1".to_string(),
            "vulkan:7".to_string(),
        ];
        let mut explicit_bindings = BTreeMap::new();
        explicit_bindings.insert("gpu1".to_string(), "vulkan:5".to_string());
        explicit_bindings.insert("remote0".to_string(), "lan:worker-a".to_string());

        let bindings = RuntimeDeviceBindings::from_vulkan_targets(
            &logical_device_ids,
            &explicit_bindings,
            Some(0),
            |target| {
                if let Some(index) = target.strip_prefix("vulkan:") {
                    return index.parse::<usize>().map(Some).map_err(|error| {
                        format!("invalid Vulkan physical device reference {target:?}: {error}")
                    });
                }
                Ok(None)
            },
        );

        assert_eq!(bindings.schema, RUNTIME_DEVICE_BINDINGS_SCHEMA);
        assert_eq!(bindings.process_vulkan_device_index, Some(0));
        assert_eq!(bindings.default_vulkan_device_index, Some(0));
        assert_eq!(bindings.requested_vulkan_device_indices, vec![0, 5, 7]);
        assert!(!bindings.can_mount_in_process);
        assert_eq!(bindings.mounting_model, "unsupported_targets");
        assert_eq!(bindings.unsupported_targets, vec!["remote0=lan:worker-a"]);
        assert_eq!(
            bindings
                .logical_devices
                .iter()
                .map(|device| (
                    device.device_id.as_str(),
                    device.target.as_deref(),
                    device.binding_source.as_str()
                ))
                .collect::<Vec<_>>(),
            vec![
                ("gpu0", Some("vulkan:0"), "process_default"),
                ("gpu1", Some("vulkan:5"), "explicit"),
                ("remote0", Some("lan:worker-a"), "explicit"),
                ("vulkan:7", Some("vulkan:7"), "device_id"),
            ]
        );

        let payload = serde_json::to_value(&bindings).unwrap();
        assert_eq!(payload["schema"], RUNTIME_DEVICE_BINDINGS_SCHEMA);
        assert_eq!(payload["logical_devices"][0]["device_id"], "gpu0");
        assert_eq!(
            payload["logical_devices"][0]["binding_source"],
            "process_default"
        );
        assert_eq!(payload["unsupported_targets"][0], "remote0=lan:worker-a");
    }

    #[test]
    fn runtime_bound_device_serializes_logical_target_report() {
        let bound = RuntimeBoundDevice {
            device_id: "gpu1".to_string(),
            target: Some("vulkan:5".to_string()),
            physical_device_index: Some(5),
            device_name: "Radeon Test Device".to_string(),
        };

        let payload = serde_json::to_value(&bound).unwrap();

        assert_eq!(payload["device_id"], "gpu1");
        assert_eq!(payload["target"], "vulkan:5");
        assert_eq!(payload["physical_device_index"], 5);
        assert_eq!(payload["device_name"], "Radeon Test Device");
    }

    #[test]
    fn runtime_source_pedal_serializes_compiled_pedal_summary() {
        let pedal = RuntimeSourcePedal {
            pedal_index: 5,
            pedal_id: "layer_05".to_string(),
            operator_type: "attention".to_string(),
            implementation: "vulkan_resident".to_string(),
            behavioral_role: "transformer_layer".to_string(),
            source_layer_index: 5,
            circuit_id: "layer_05_circuit_v1".to_string(),
            input_ports: vec![RuntimePedalPortSummary {
                id: "input_frame".to_string(),
                signal: "frame".to_string(),
                shape: vec![1024],
                source: Some("hidden_states".to_string()),
                pedal_port: Some("input".to_string()),
            }],
            output_ports: vec![RuntimePedalPortSummary {
                id: "output_frame".to_string(),
                signal: "frame".to_string(),
                shape: vec![1024],
                source: Some("hidden_states".to_string()),
                pedal_port: Some("output".to_string()),
            }],
            state_port_count: 1,
            parameter_ref_count: 12,
            node_count: 7,
            kernel_count: 7,
        };

        let payload = serde_json::to_value(&pedal).unwrap();

        assert_eq!(payload["pedal_index"], 5);
        assert_eq!(payload["pedal_id"], "layer_05");
        assert_eq!(payload["operator_type"], "attention");
        assert_eq!(payload["input_ports"][0]["id"], "input_frame");
        assert_eq!(payload["input_ports"][0]["pedal_port"], "input");
        assert_eq!(payload["output_ports"][0]["pedal_port"], "output");
        assert_eq!(payload["kernel_count"], 7);
    }

    #[test]
    fn runtime_capacity_profile_serializes_capacity_summary() {
        let profile = RuntimeCapacityProfileSummary {
            min_dynamic_state_capacity_activations: 4,
            max_dynamic_state_capacity_activations: 64,
            shader_override_count: 14,
        };

        let payload = serde_json::to_value(&profile).unwrap();

        assert_eq!(payload["min_dynamic_state_capacity_activations"], 4);
        assert_eq!(payload["max_dynamic_state_capacity_activations"], 64);
        assert_eq!(payload["shader_override_count"], 14);
    }

    #[test]
    fn placement_plan_rejects_unknown_pedal_overrides() {
        let resolved =
            ResolvedLoweredPedalboard::from_index_file(fixture_model_index_path()).unwrap();
        let spec = StreamCircuitPlacementSpec::new("gpu0").with_pedal_device("layer_99", "gpu1");

        let error = resolved.placement_plan(&spec).unwrap_err();

        assert!(error.0.contains("unknown pedal"));
        assert!(error.0.contains("layer_99"));
    }

    #[test]
    fn runtime_patch_defaults_to_source_series_with_device_overrides() {
        let resolved =
            ResolvedLoweredPedalboard::from_index_file(fixture_model_index_path()).unwrap();
        let spec = StreamCircuitPlacementSpec::new("gpu0")
            .with_pedal_device("layer_02", "gpu1")
            .with_pedal_device("layer_07", "lan:worker-a");

        let patch = resolved.runtime_patch_from_placement(&spec).unwrap();

        assert_eq!(patch.schema, STREAM_CIRCUIT_RUNTIME_PATCH_SCHEMA);
        assert_eq!(patch.wiring, "series");
        assert_eq!(patch.instances.len(), resolved.circuits.len());
        assert_eq!(patch.instances[0].instance_id, "layer_00");
        assert_eq!(patch.instances[0].source_pedal_id, "layer_00");
        assert_eq!(patch.instances[0].device_id, "gpu0");
        assert_eq!(
            patch
                .instances
                .iter()
                .find(|instance| instance.instance_id == "layer_02")
                .unwrap()
                .device_id,
            "gpu1"
        );
        assert_eq!(
            patch
                .instances
                .iter()
                .find(|instance| instance.instance_id == "layer_07")
                .unwrap()
                .device_id,
            "lan:worker-a"
        );
        assert_eq!(patch.placement_spec(), spec);
    }

    #[test]
    fn runtime_patch_can_duplicate_a_layer_as_a_new_pedal_instance() {
        let resolved =
            ResolvedLoweredPedalboard::from_index_file(fixture_model_index_path()).unwrap();
        let patch = resolved
            .default_runtime_patch("gpu0")
            .unwrap()
            .duplicate_after_instance("layer_05", "layer_05_repeat")
            .unwrap()
            .with_instance_device("layer_05_repeat", "gpu1")
            .unwrap();

        let instantiated = resolved.instantiate_runtime_patch(&patch).unwrap();
        let placement = instantiated
            .placement_plan(&patch.placement_spec())
            .unwrap();

        assert_eq!(instantiated.circuits.len(), resolved.circuits.len() + 1);
        assert_eq!(
            instantiated.index.summary.circuit_count,
            resolved.circuits.len() + 1
        );
        let original_index = instantiated
            .circuits
            .iter()
            .position(|artifact| artifact.pedal.id == "layer_05")
            .unwrap();
        let duplicate_index = original_index + 1;
        let duplicate = &instantiated.circuits[duplicate_index];

        assert_eq!(duplicate.pedal.id, "layer_05_repeat");
        assert_eq!(duplicate.circuit.source.pedal_id, "layer_05");
        assert_eq!(
            duplicate.params.refs.keys().collect::<Vec<_>>(),
            resolved.circuits[original_index]
                .params
                .refs
                .keys()
                .collect::<Vec<_>>()
        );
        assert!(
            instantiated
                .transient_allocations()
                .iter()
                .any(|allocation| allocation.pedal_id == "layer_05_repeat")
        );
        assert_eq!(
            placement.pedal("layer_05_repeat").unwrap().device_id,
            "gpu1"
        );
        assert_eq!(
            placement.cables[duplicate_index - 1].source_pedal_id,
            "layer_05"
        );
        assert_eq!(
            placement.cables[duplicate_index - 1].destination_pedal_id,
            "layer_05_repeat"
        );
        assert_eq!(
            placement.cables[duplicate_index].source_pedal_id,
            "layer_05_repeat"
        );
        assert_eq!(
            placement.cables[duplicate_index].destination_pedal_id,
            "layer_06"
        );
        assert_eq!(placement.cross_device_cable_count, 2);
    }

    #[test]
    fn runtime_patch_can_use_an_explicit_source_chain() {
        let resolved =
            ResolvedLoweredPedalboard::from_index_file(fixture_model_index_path()).unwrap();
        let chain = vec![
            ("layer_00".to_string(), "layer_00".to_string()),
            ("layer_01".to_string(), "layer_01".to_string()),
            ("layer_05".to_string(), "layer_05".to_string()),
            ("layer_05_repeat".to_string(), "layer_05".to_string()),
            ("layer_06".to_string(), "layer_06".to_string()),
            ("layer_13".to_string(), "layer_13".to_string()),
        ];
        let patch = StreamCircuitRuntimePatch::from_source_chain(&resolved, "gpu0", &chain)
            .unwrap()
            .with_instance_device("layer_05_repeat", "gpu1")
            .unwrap();

        assert_eq!(
            patch
                .instances
                .iter()
                .map(|instance| (
                    instance.instance_id.as_str(),
                    instance.source_pedal_id.as_str()
                ))
                .collect::<Vec<_>>(),
            vec![
                ("layer_00", "layer_00"),
                ("layer_01", "layer_01"),
                ("layer_05", "layer_05"),
                ("layer_05_repeat", "layer_05"),
                ("layer_06", "layer_06"),
                ("layer_13", "layer_13"),
            ]
        );

        let instantiated = resolved.instantiate_runtime_patch(&patch).unwrap();
        let placement = instantiated
            .placement_plan(&patch.placement_spec())
            .unwrap();

        assert_eq!(instantiated.circuits.len(), chain.len());
        assert!(
            instantiated
                .circuits
                .iter()
                .all(|artifact| artifact.pedal.id != "layer_02")
        );
        assert_eq!(
            placement
                .pedals
                .iter()
                .map(|pedal| pedal.pedal_id.as_str())
                .collect::<Vec<_>>(),
            vec![
                "layer_00",
                "layer_01",
                "layer_05",
                "layer_05_repeat",
                "layer_06",
                "layer_13",
            ]
        );
        assert_eq!(
            placement.pedal("layer_05_repeat").unwrap().device_id,
            "gpu1"
        );
        assert_eq!(placement.cross_device_cable_count, 2);
    }

    #[test]
    fn installed_stream_circuit_exposes_mount_plans() {
        let installed = InstalledStreamCircuit::from_index_file(
            fixture_model_index_path(),
            "compiled_model_stream_circuit",
            "stream_circuit_ir",
        )
        .unwrap();
        let tensor_index = TensorIndex::from_json_file(fixture_model_tensor_index_path()).unwrap();

        let execution_plan = installed.execution_plan().unwrap();
        let resource_plan = installed.resource_plan().unwrap();
        let shaped_resource_plan = installed
            .resource_plan_with_tensor_index(&tensor_index)
            .unwrap();

        assert_eq!(execution_plan.circuits.len(), 14);
        assert_eq!(execution_plan.total_node_count(), 242);
        assert_eq!(resource_plan.circuit_count, 14);
        assert_eq!(resource_plan.parameter_ref_count, 130);
        assert_eq!(resource_plan.stream_state_count(), 14);
        assert_eq!(resource_plan.state_view_signal_count, 20);
        assert_eq!(resource_plan.layer_local_activation_slot_count, 56);
        assert_eq!(resource_plan.unknown_temporary_shape_count, 172);
        assert_eq!(resource_plan.unknown_state_view_shape_count, 12);
        assert_eq!(shaped_resource_plan.unknown_temporary_shape_count, 0);
        assert_eq!(shaped_resource_plan.unknown_state_view_shape_count, 12);
        assert_eq!(
            resource_plan
                .activation_banks
                .iter()
                .find(|bank| bank.pedal_id == "layer_02")
                .map(|bank| bank.slot_count),
            Some(4)
        );
    }

    #[test]
    fn stream_circuit_capability_report_names_missing_executors() {
        let installed = InstalledStreamCircuit::from_index_file(
            fixture_model_index_path(),
            "compiled_model_stream_circuit",
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
            fixture_model_index_path(),
            "compiled_model_stream_circuit",
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

    #[test]
    fn stream_circuit_device_mounts_streams_without_claiming_execution() {
        let capabilities = StreamCircuitBackendCapabilities::new("stream_circuit_ir");
        let mut backend = StreamCircuitDeviceBackend::from_index_file(
            "device_0",
            fixture_model_index_path(),
            capabilities,
        )
        .unwrap();

        backend.create_stream("s0").unwrap();
        backend
            .inject_prompt(PromptInjection::new("s0", vec![1, 2], 1))
            .unwrap();

        let manifest = backend.describe();
        let template = backend.stream_template("s0").unwrap();
        let resource_plan = backend.resource_plan().unwrap();
        let tensor_index = TensorIndex::from_json_file(fixture_model_tensor_index_path()).unwrap();
        let shaped_resource_plan = backend
            .resource_plan_with_tensor_index(&tensor_index)
            .unwrap();

        assert_eq!(backend.backend_id(), StreamCircuitDeviceBackend::BACKEND_ID);
        assert_eq!(manifest.permanent_circuit.pedal_count, 14);
        assert_eq!(template.stream_id, "s0");
        assert_eq!(template.allocations.len(), 14);
        assert_eq!(resource_plan.parameter_ref_count, 130);
        assert_eq!(resource_plan.activation_banks.len(), 14);
        assert_eq!(resource_plan.state_view_signal_count, 20);
        assert_eq!(shaped_resource_plan.unknown_temporary_shape_count, 0);
        assert_eq!(shaped_resource_plan.unknown_state_view_shape_count, 12);
        assert_eq!(
            template.allocation_keys().get(2).map(String::as_str),
            Some("layer_02.kv_memory")
        );
        assert_eq!(
            backend.dispatch(0).unwrap().status,
            DispatchStatus::BudgetExhausted
        );

        match backend.dispatch(1) {
            Err(StreamCircuitRuntimeError::UnsupportedExecutionPlan(report)) => {
                assert_eq!(report.unsupported_ops().len(), 12);
                assert!(report.unsupported_ops().contains(&"linear".to_string()));
            }
            other => panic!("expected unsupported execution plan, got {other:?}"),
        }
    }

    #[test]
    fn stream_circuit_device_fork_preserves_allocations_with_child_identity() {
        let capabilities = StreamCircuitBackendCapabilities::new("stream_circuit_ir");
        let mut backend = StreamCircuitDeviceBackend::from_index_file(
            "device_0",
            fixture_model_index_path(),
            capabilities,
        )
        .unwrap();
        backend.create_stream("parent").unwrap();

        backend
            .fork_stream(ForkRequest {
                parent_stream_id: "parent".to_string(),
                child_stream_id: "child".to_string(),
                state_policy: ForkPolicy::Clone,
                random_policy: RandomPolicy::Clone,
                random_seed: None,
            })
            .unwrap();

        let parent = backend.stream_template("parent").unwrap();
        let child = backend.stream_template("child").unwrap();

        assert_eq!(parent.stream_id, "parent");
        assert_eq!(child.stream_id, "child");
        assert_eq!(parent.allocation_keys(), child.allocation_keys());
    }
}
