use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

pub const STREAM_CIRCUIT_SCHEMA: &str = "llmoop.stream_circuit.v1";
pub const CIRCUIT_PARAMS_SCHEMA: &str = "llmoop.circuit_params.v1";
pub const CIRCUIT_STATE_SCHEMA: &str = "llmoop.circuit_state.v1";
pub const LOWERED_PEDALBOARD_SCHEMA: &str = "llmoop.lowered_pedalboard.v1";
pub const STREAM_CIRCUIT_PLACEMENT_SCHEMA: &str = "llmoop.stream_circuit_placement.v1";
pub const STREAM_CIRCUIT_RUNTIME_PATCH_SCHEMA: &str = "llmoop.stream_circuit_runtime_patch.v1";
pub const RUNTIME_CABLE_ROUTES_SCHEMA: &str = "llmoop.runtime_cable_routes.v1";
pub const RUNTIME_DEVICE_BINDINGS_SCHEMA: &str = "llmoop.runtime_device_bindings.v1";
pub const RUNTIME_TOPOLOGY_SCHEMA: &str = "llmoop.runtime_topology.v1";

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
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LoweredPedalboardGraph {
    pub wiring: String,
    #[serde(default)]
    pub circuits: Vec<LoweredCircuitRef>,
    pub cables: Vec<StreamCircuitGraphCable>,
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
    pub cables: Vec<StreamCircuitGraphCable>,
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
        let patch = Self {
            schema: STREAM_CIRCUIT_RUNTIME_PATCH_SCHEMA.to_string(),
            wiring: "explicit_graph".to_string(),
            default_device_id: default_device_id.into(),
            instances,
            cables,
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
        let source_output = source_artifact
            .circuit
            .boundary
            .outputs
            .first()
            .ok_or_else(|| {
                CircuitPlacementError(format!(
                    "runtime patch instance {} has no output port",
                    source.instance_id
                ))
            })?;
        let duplicate_input = source_artifact
            .circuit
            .boundary
            .inputs
            .first()
            .ok_or_else(|| {
                CircuitPlacementError(format!(
                    "runtime patch duplicate {} has no input port",
                    new_instance_id
                ))
            })?;
        let outgoing = self
            .cables
            .iter()
            .enumerate()
            .filter(|(_, cable)| cable.source.pedal_id == after_instance_id)
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        if outgoing.len() > 1 {
            return Err(CircuitPlacementError(format!(
                "cannot insert duplicate after branching pedal {after_instance_id:?}; wire the explicit graph instead"
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
        };
        if let Some(outgoing_index) = outgoing.first().copied() {
            self.cables[outgoing_index].source = StreamCircuitCableEndpoint {
                pedal_id: new_instance_id.clone(),
                port_id: source_output.id.clone(),
            };
            self.cables.insert(outgoing_index, inserted_cable);
        } else {
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

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StreamCircuitPedalInstance {
    pub instance_id: String,
    pub source_pedal_id: String,
    pub device_id: String,
    #[serde(default = "default_enabled")]
    pub enabled: bool,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub control_values: BTreeMap<String, serde_json::Value>,
    pub state_policy: StreamCircuitPedalInstanceStatePolicy,
}

fn default_enabled() -> bool {
    true
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StreamCircuitPedalInstanceStatePolicy {
    Fresh,
    CloneFrom { instance_id: String },
    ShareWith { instance_id: String },
}

fn series_cables_for_instances(
    graph: &ResolvedLoweredPedalboard,
    instances: &[StreamCircuitPedalInstance],
) -> Result<Vec<StreamCircuitGraphCable>, CircuitPlacementError> {
    let source_by_id = graph
        .circuits
        .iter()
        .map(|artifact| (artifact.pedal.id.as_str(), artifact))
        .collect::<BTreeMap<_, _>>();
    instances
        .windows(2)
        .enumerate()
        .map(|(index, pair)| {
            let source = source_by_id
                .get(pair[0].source_pedal_id.as_str())
                .ok_or_else(|| {
                    CircuitPlacementError(format!(
                        "runtime patch instance {} references unknown source pedal {}",
                        pair[0].instance_id, pair[0].source_pedal_id
                    ))
                })?;
            let destination = source_by_id
                .get(pair[1].source_pedal_id.as_str())
                .ok_or_else(|| {
                    CircuitPlacementError(format!(
                        "runtime patch instance {} references unknown source pedal {}",
                        pair[1].instance_id, pair[1].source_pedal_id
                    ))
                })?;
            let output = source.circuit.boundary.outputs.first().ok_or_else(|| {
                CircuitPlacementError(format!(
                    "runtime patch instance {} has no output port",
                    pair[0].instance_id
                ))
            })?;
            let input = destination.circuit.boundary.inputs.first().ok_or_else(|| {
                CircuitPlacementError(format!(
                    "runtime patch instance {} has no input port",
                    pair[1].instance_id
                ))
            })?;
            Ok(StreamCircuitGraphCable {
                id: format!("cable_{index:04}"),
                source: StreamCircuitCableEndpoint {
                    pedal_id: pair[0].instance_id.clone(),
                    port_id: output.id.clone(),
                },
                destination: StreamCircuitCableEndpoint {
                    pedal_id: pair[1].instance_id.clone(),
                    port_id: input.id.clone(),
                },
            })
        })
        .collect()
}

fn allocate_cable_id(
    cables: &[StreamCircuitGraphCable],
    source_id: &str,
    destination_id: &str,
) -> String {
    let base = format!("{source_id}_to_{destination_id}");
    if !cables.iter().any(|cable| cable.id == base) {
        return base;
    }
    (2..)
        .map(|suffix| format!("{base}_{suffix}"))
        .find(|candidate| !cables.iter().any(|cable| cable.id == *candidate))
        .expect("unbounded cable id suffix space")
}

fn validate_instance_state_policy(
    instance: &StreamCircuitPedalInstance,
    instances: &[StreamCircuitPedalInstance],
    source_by_id: &BTreeMap<&str, &ResolvedCircuitArtifact>,
) -> Result<(), CircuitPlacementError> {
    let target_id = match &instance.state_policy {
        StreamCircuitPedalInstanceStatePolicy::Fresh => return Ok(()),
        StreamCircuitPedalInstanceStatePolicy::CloneFrom { instance_id }
        | StreamCircuitPedalInstanceStatePolicy::ShareWith { instance_id } => instance_id,
    };
    if target_id == &instance.instance_id {
        return Err(CircuitPlacementError(format!(
            "runtime patch instance {} cannot source state from itself",
            instance.instance_id
        )));
    }
    let target = instances
        .iter()
        .find(|candidate| candidate.instance_id == *target_id)
        .ok_or_else(|| {
            CircuitPlacementError(format!(
                "runtime patch instance {} sources state from unknown instance {}",
                instance.instance_id, target_id
            ))
        })?;
    if !target.enabled || !instance.enabled {
        return Err(CircuitPlacementError(format!(
            "state-linked instances {} and {} must both be enabled",
            instance.instance_id, target.instance_id
        )));
    }
    let source_state = &source_by_id[instance.source_pedal_id.as_str()]
        .state
        .state_ports;
    let target_state = &source_by_id[target.source_pedal_id.as_str()]
        .state
        .state_ports;
    if source_state != target_state {
        return Err(CircuitPlacementError(format!(
            "runtime patch instances {} and {} have incompatible state contracts",
            instance.instance_id, target.instance_id
        )));
    }
    if matches!(
        instance.state_policy,
        StreamCircuitPedalInstanceStatePolicy::ShareWith { .. }
    ) && instance.device_id != target.device_id
    {
        return Err(CircuitPlacementError(format!(
            "runtime patch instances {} and {} cannot share state across devices {} and {}",
            instance.instance_id, target.instance_id, instance.device_id, target.device_id
        )));
    }
    Ok(())
}

fn validate_state_policy_dependencies(
    instances: &[StreamCircuitPedalInstance],
) -> Result<(), CircuitPlacementError> {
    let dependencies = instances
        .iter()
        .filter_map(|instance| {
            let source = match &instance.state_policy {
                StreamCircuitPedalInstanceStatePolicy::Fresh => return None,
                StreamCircuitPedalInstanceStatePolicy::CloneFrom { instance_id }
                | StreamCircuitPedalInstanceStatePolicy::ShareWith { instance_id } => instance_id,
            };
            Some((instance.instance_id.as_str(), source.as_str()))
        })
        .collect::<BTreeMap<_, _>>();
    for start in dependencies.keys() {
        let mut visited = BTreeSet::new();
        let mut current = *start;
        while let Some(next) = dependencies.get(current) {
            if !visited.insert(current) {
                return Err(CircuitPlacementError(format!(
                    "runtime patch state policies contain a dependency cycle at {current}"
                )));
            }
            current = next;
        }
    }
    Ok(())
}

fn validate_explicit_cables(
    patch: &StreamCircuitRuntimePatch,
    source_by_id: &BTreeMap<&str, &ResolvedCircuitArtifact>,
) -> Result<(), CircuitPlacementError> {
    let instances = patch
        .instances
        .iter()
        .map(|instance| (instance.instance_id.as_str(), instance))
        .collect::<BTreeMap<_, _>>();
    let mut ids = BTreeSet::new();
    let mut source_ports = BTreeSet::new();
    let mut destination_ports = BTreeSet::new();
    let mut incoming_count = BTreeMap::<&str, usize>::new();
    let mut outgoing_count = BTreeMap::<&str, usize>::new();
    for cable in &patch.cables {
        if cable.id.is_empty() || !ids.insert(cable.id.as_str()) {
            return Err(CircuitPlacementError(format!(
                "runtime patch contains an empty or duplicate cable id {:?}",
                cable.id
            )));
        }
        let source_instance = instances
            .get(cable.source.pedal_id.as_str())
            .ok_or_else(|| {
                CircuitPlacementError(format!(
                    "runtime patch cable {} references unknown source instance {}",
                    cable.id, cable.source.pedal_id
                ))
            })?;
        let destination_instance = instances
            .get(cable.destination.pedal_id.as_str())
            .ok_or_else(|| {
                CircuitPlacementError(format!(
                    "runtime patch cable {} references unknown destination instance {}",
                    cable.id, cable.destination.pedal_id
                ))
            })?;
        if source_instance.instance_id == destination_instance.instance_id {
            return Err(CircuitPlacementError(format!(
                "runtime patch cable {} creates an un-delayed self-loop on {}",
                cable.id, source_instance.instance_id
            )));
        }
        if !source_ports.insert((
            source_instance.instance_id.as_str(),
            cable.source.port_id.as_str(),
        )) {
            return Err(CircuitPlacementError(format!(
                "runtime patch output {}.{} has more than one cable; use an explicit splitter pedal",
                source_instance.instance_id, cable.source.port_id
            )));
        }
        if !destination_ports.insert((
            destination_instance.instance_id.as_str(),
            cable.destination.port_id.as_str(),
        )) {
            return Err(CircuitPlacementError(format!(
                "runtime patch input {}.{} has more than one cable",
                destination_instance.instance_id, cable.destination.port_id
            )));
        }
        *incoming_count
            .entry(destination_instance.instance_id.as_str())
            .or_default() += 1;
        *outgoing_count
            .entry(source_instance.instance_id.as_str())
            .or_default() += 1;
        validate_graph_cable_contract(
            cable,
            source_instance,
            source_by_id[source_instance.source_pedal_id.as_str()],
            destination_instance,
            source_by_id[destination_instance.source_pedal_id.as_str()],
        )?;
    }
    for instance in patch.instances.iter().filter(|instance| !instance.enabled) {
        if incoming_count
            .get(instance.instance_id.as_str())
            .copied()
            .unwrap_or(0)
            > 1
            || outgoing_count
                .get(instance.instance_id.as_str())
                .copied()
                .unwrap_or(0)
                > 1
        {
            return Err(CircuitPlacementError(format!(
                "disabled branching or joining pedal {} cannot be bypassed automatically",
                instance.instance_id
            )));
        }
    }
    let effective = patch.effective_cables()?;
    let enabled = patch
        .instances
        .iter()
        .filter(|instance| instance.enabled)
        .map(|instance| (instance.instance_id.as_str(), instance))
        .collect::<BTreeMap<_, _>>();
    for cable in &effective {
        let source_instance = enabled[cable.source.pedal_id.as_str()];
        let destination_instance = enabled[cable.destination.pedal_id.as_str()];
        validate_graph_cable_contract(
            cable,
            source_instance,
            source_by_id[source_instance.source_pedal_id.as_str()],
            destination_instance,
            source_by_id[destination_instance.source_pedal_id.as_str()],
        )?;
    }
    let connected_outputs = effective
        .iter()
        .map(|cable| {
            (
                cable.source.pedal_id.as_str(),
                cable.source.port_id.as_str(),
            )
        })
        .collect::<BTreeSet<_>>();
    let connected_inputs = effective
        .iter()
        .map(|cable| {
            (
                cable.destination.pedal_id.as_str(),
                cable.destination.port_id.as_str(),
            )
        })
        .collect::<BTreeSet<_>>();
    let mut open_inputs = Vec::new();
    let mut open_outputs = Vec::new();
    for instance in enabled.values() {
        let artifact = source_by_id[instance.source_pedal_id.as_str()];
        for port in &artifact.circuit.boundary.inputs {
            if !connected_inputs.contains(&(instance.instance_id.as_str(), port.id.as_str())) {
                open_inputs.push(format!("{}.{}", instance.instance_id, port.id));
            }
        }
        for port in &artifact.circuit.boundary.outputs {
            if !connected_outputs.contains(&(instance.instance_id.as_str(), port.id.as_str())) {
                open_outputs.push(format!("{}.{}", instance.instance_id, port.id));
            }
        }
    }
    if open_inputs.len() != 1 || open_outputs.len() != 1 {
        return Err(CircuitPlacementError(format!(
            "runtime patch must expose exactly one model input and one model output; open inputs={open_inputs:?}, open outputs={open_outputs:?}"
        )));
    }
    Ok(())
}

fn validate_graph_cable_contract(
    cable: &StreamCircuitGraphCable,
    source_instance: &StreamCircuitPedalInstance,
    source: &ResolvedCircuitArtifact,
    destination_instance: &StreamCircuitPedalInstance,
    destination: &ResolvedCircuitArtifact,
) -> Result<(), CircuitPlacementError> {
    let output = source
        .circuit
        .boundary
        .outputs
        .iter()
        .find(|port| port.id == cable.source.port_id)
        .ok_or_else(|| {
            CircuitPlacementError(format!(
                "runtime patch cable {} references unknown output {}.{}",
                cable.id, source_instance.instance_id, cable.source.port_id
            ))
        })?;
    let input = destination
        .circuit
        .boundary
        .inputs
        .iter()
        .find(|port| port.id == cable.destination.port_id)
        .ok_or_else(|| {
            CircuitPlacementError(format!(
                "runtime patch cable {} references unknown input {}.{}",
                cable.id, destination_instance.instance_id, cable.destination.port_id
            ))
        })?;
    if output.signal != input.signal || output.shape != input.shape {
        return Err(CircuitPlacementError(format!(
            "cannot patch cable {} ({}.{} -> {}.{}) without an adapter: output {:?}/{:?}, input {:?}/{:?}",
            cable.id,
            source_instance.instance_id,
            output.id,
            destination_instance.instance_id,
            input.id,
            output.signal,
            output.shape,
            input.signal,
            input.shape
        )));
    }
    Ok(())
}

fn effective_runtime_patch_cables(
    instances: &[StreamCircuitPedalInstance],
    cables: &[StreamCircuitGraphCable],
) -> Result<Vec<StreamCircuitGraphCable>, CircuitPlacementError> {
    let enabled = instances
        .iter()
        .map(|instance| (instance.instance_id.as_str(), instance.enabled))
        .collect::<BTreeMap<_, _>>();
    let outgoing = cables.iter().fold(
        BTreeMap::<&str, Vec<&StreamCircuitGraphCable>>::new(),
        |mut map, cable| {
            map.entry(cable.source.pedal_id.as_str())
                .or_default()
                .push(cable);
            map
        },
    );
    let mut effective = Vec::new();
    for cable in cables {
        if !enabled
            .get(cable.source.pedal_id.as_str())
            .copied()
            .unwrap_or(false)
        {
            continue;
        }
        let mut destination = cable.destination.clone();
        let mut visited = BTreeSet::new();
        while !enabled
            .get(destination.pedal_id.as_str())
            .copied()
            .unwrap_or(false)
        {
            if !visited.insert(destination.pedal_id.clone()) {
                return Err(CircuitPlacementError(format!(
                    "runtime patch bypass path contains a cycle at {}",
                    destination.pedal_id
                )));
            }
            let next = outgoing
                .get(destination.pedal_id.as_str())
                .and_then(|candidates| candidates.first())
                .copied();
            let Some(next) = next else {
                break;
            };
            destination = next.destination.clone();
        }
        if enabled
            .get(destination.pedal_id.as_str())
            .copied()
            .unwrap_or(false)
        {
            effective.push(StreamCircuitGraphCable {
                id: cable.id.clone(),
                source: cable.source.clone(),
                destination,
            });
        }
    }
    Ok(effective)
}

fn topological_runtime_patch_order(
    instances: &[StreamCircuitPedalInstance],
    cables: &[StreamCircuitGraphCable],
) -> Result<Vec<String>, CircuitPlacementError> {
    let enabled_ids = instances
        .iter()
        .filter(|instance| instance.enabled)
        .map(|instance| instance.instance_id.as_str())
        .collect::<BTreeSet<_>>();
    let mut indegree = enabled_ids
        .iter()
        .map(|id| (*id, 0usize))
        .collect::<BTreeMap<_, _>>();
    let mut outgoing = BTreeMap::<&str, Vec<&str>>::new();
    for cable in cables {
        *indegree
            .get_mut(cable.destination.pedal_id.as_str())
            .ok_or_else(|| {
                CircuitPlacementError(format!(
                    "effective cable {} has a disabled destination",
                    cable.id
                ))
            })? += 1;
        outgoing
            .entry(cable.source.pedal_id.as_str())
            .or_default()
            .push(cable.destination.pedal_id.as_str());
    }
    let mut remaining = enabled_ids;
    let mut ordered = Vec::with_capacity(remaining.len());
    while !remaining.is_empty() {
        let ready = instances
            .iter()
            .filter(|instance| instance.enabled)
            .map(|instance| instance.instance_id.as_str())
            .find(|id| remaining.contains(id) && indegree[id] == 0)
            .ok_or_else(|| {
                CircuitPlacementError(
                    "runtime patch graph contains a cycle; feedback requires an explicit stateful delay pedal"
                        .to_string(),
                )
            })?;
        remaining.remove(ready);
        ordered.push(ready.to_string());
        for destination in outgoing.get(ready).into_iter().flatten() {
            let value = indegree
                .get_mut(destination)
                .expect("validated cable destination must have indegree");
            *value -= 1;
        }
    }
    Ok(ordered)
}

fn validate_runtime_patch_source_graph(
    graph: &ResolvedLoweredPedalboard,
) -> Result<(), CircuitPlacementError> {
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

        let circuit_by_id = graph
            .circuits
            .iter()
            .map(|artifact| (artifact.pedal.id.as_str(), artifact))
            .collect::<BTreeMap<_, _>>();
        let mut cables = Vec::with_capacity(graph.index.graph.cables.len());
        let mut local_cable_count = 0usize;
        let mut cross_device_cable_count = 0usize;
        for (cable_index, cable) in graph.index.graph.cables.iter().enumerate() {
            let source = circuit_by_id
                .get(cable.source.pedal_id.as_str())
                .ok_or_else(|| {
                    CircuitPlacementError(format!(
                        "placement cable {} references unknown source pedal {}",
                        cable.id, cable.source.pedal_id
                    ))
                })?;
            let destination = circuit_by_id
                .get(cable.destination.pedal_id.as_str())
                .ok_or_else(|| {
                    CircuitPlacementError(format!(
                        "placement cable {} references unknown destination pedal {}",
                        cable.id, cable.destination.pedal_id
                    ))
                })?;
            let output = source
                .circuit
                .boundary
                .outputs
                .iter()
                .find(|port| port.id == cable.source.port_id)
                .ok_or_else(|| {
                    CircuitPlacementError(format!(
                        "{} has no output port {} for placement cable {}",
                        source.pedal.id, cable.source.port_id, cable.id
                    ))
                })?;
            let input = destination
                .circuit
                .boundary
                .inputs
                .iter()
                .find(|port| port.id == cable.destination.port_id)
                .ok_or_else(|| {
                    CircuitPlacementError(format!(
                        "{} has no input port {} for placement cable {}",
                        destination.pedal.id, cable.destination.port_id, cable.id
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
                Err(error) => unsupported_targets.push(format!("{logical_device_id} ({error})")),
                Ok(None) => {}
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
                        Ok(Some(index)) => Some(format!("vulkan:{index}")),
                        Ok(None) | Err(_) if logical_device_id.contains(':') => {
                            Some(logical_device_id.clone())
                        }
                        Ok(None) | Err(_) => None,
                    }
                } else {
                    None
                };
                let has_direct_target = direct_target.is_some();
                let target = explicit_target
                    .cloned()
                    .or(direct_target)
                    .or_else(|| default_vulkan_device_index.map(|index| format!("vulkan:{index}")));
                let binding_source = if explicit_target.is_some() {
                    "explicit"
                } else if has_direct_target {
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
pub struct RuntimeAvailableMemoryHeap {
    pub heap_index: u32,
    pub size_bytes: u64,
    pub device_local: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeAvailableDevice {
    pub device_id: String,
    pub backend: String,
    pub available: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runtime_device_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub physical_device_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub physical_device_index: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vendor_id: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw_device_id: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_version: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub driver_version: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compute_queue_family_indices: Option<Vec<u32>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory_heaps: Option<Vec<RuntimeAvailableMemoryHeap>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selected_by_default: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selected_by_runtime: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runtime_binding: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub can_host_runtime_pedals_on_physical_device: Option<bool>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub notes: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
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
pub struct RuntimePatchSourceChainEntry {
    pub instance_id: String,
    pub source_pedal_id: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimePatchDuplicateAfterControl {
    pub after_instance_id: String,
    pub new_instance_id: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimePatchControls {
    pub default_device_id: Option<String>,
    pub pedal_devices: BTreeMap<String, String>,
    pub source_chain: Option<Vec<RuntimePatchSourceChainEntry>>,
    pub duplicate_after: Vec<RuntimePatchDuplicateAfterControl>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RuntimeCompiledPedalboardSummary {
    pub wiring: String,
    pub default_device_id: String,
    pub pedal_devices: BTreeMap<String, String>,
    pub source_pedal_count: usize,
    pub source_pedals: Vec<RuntimeSourcePedal>,
    pub max_context_activations: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeEffectivePedalboardTopology {
    pub wiring: String,
    pub pedal_count: usize,
    pub cable_count: usize,
    pub local_cable_count: usize,
    pub cross_device_cable_count: usize,
    pub device_count: usize,
    pub device_ids: Vec<String>,
    pub device_bindings: RuntimeDeviceBindings,
    pub cable_routes: RuntimeCableRoutes,
    pub pedals: Vec<PedalPlacement>,
    pub cables: Vec<PedalCablePlacement>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RuntimeTopologyReport {
    pub ok: bool,
    pub schema: String,
    pub package_manifest: PathBuf,
    pub package_root: PathBuf,
    pub package_id: String,
    pub compiled_schema: String,
    pub config_path: String,
    pub tokenizer: Value,
    pub available_devices: Vec<RuntimeAvailableDevice>,
    pub compiled: RuntimeCompiledPedalboardSummary,
    pub runtime_patch_controls: RuntimePatchControls,
    pub runtime_patch: StreamCircuitRuntimePatch,
    pub effective: RuntimeEffectivePedalboardTopology,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RuntimePackageInspectionReport {
    pub ok: bool,
    pub package_manifest: PathBuf,
    pub package_root: PathBuf,
    pub schema: String,
    pub package_id: String,
    pub config_path: String,
    pub tokenizer: Value,
    pub compiled_wiring: String,
    pub compiled_default_device_id: String,
    pub compiled_pedal_devices: BTreeMap<String, String>,
    pub runtime_patch: RuntimePatchControls,
    pub device_bindings: RuntimeDeviceBindings,
    pub max_context_activations: usize,
    pub source_pedal_count: usize,
    pub source_pedals: Vec<RuntimeSourcePedal>,
    pub available_devices: Vec<RuntimeAvailableDevice>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimePatchPlacementReport {
    pub schema: String,
    pub wiring: String,
    pub local_cable_count: usize,
    pub cross_device_cable_count: usize,
    pub runtime_routes: RuntimeCableRoutes,
    pub pedals: Vec<PedalPlacement>,
    pub cables: Vec<PedalCablePlacement>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimePatchInspectionReport {
    pub ok: bool,
    pub package_manifest: PathBuf,
    pub package_root: PathBuf,
    pub package_id: String,
    pub compiled_source_pedal_count: usize,
    pub runtime_patch_controls: RuntimePatchControls,
    pub runtime_patch: StreamCircuitRuntimePatch,
    pub device_bindings: RuntimeDeviceBindings,
    pub effective_pedal_count: usize,
    pub effective_cable_count: usize,
    pub placement: RuntimePatchPlacementReport,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeLocalCableBufferReport {
    pub cable_index: usize,
    pub signal: String,
    pub source_pedal_id: String,
    pub destination_pedal_id: String,
    pub device_id: String,
    pub byte_capacity: Option<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeRemoteCableBufferReport {
    pub cable_index: usize,
    pub signal: String,
    pub source_device_id: String,
    pub source_pedal_id: String,
    pub destination_device_id: String,
    pub destination_pedal_id: String,
    pub byte_capacity: Option<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeDeviceTickPlanReport {
    pub stage_count: usize,
    pub receive_stage_count: usize,
    pub dispatch_stage_count: usize,
    pub publish_stage_count: usize,
    pub local_cable_read_count: usize,
    pub local_cable_write_count: usize,
    pub incoming_cable_read_count: usize,
    pub outgoing_cable_write_count: usize,
    pub model_input_read_count: usize,
    pub model_output_write_count: usize,
    pub can_execute: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeDeviceSliceReport {
    pub ok: bool,
    pub package_manifest: PathBuf,
    pub device_name: String,
    pub device_id: String,
    pub context_window_activations: usize,
    pub hosted_pedals: Vec<String>,
    pub local_cables: Vec<RuntimeLocalCableBufferReport>,
    pub incoming_cables: Vec<RuntimeRemoteCableBufferReport>,
    pub outgoing_cables: Vec<RuntimeRemoteCableBufferReport>,
    pub hosted_pedal_count: usize,
    pub incoming_cable_count: usize,
    pub outgoing_cable_count: usize,
    pub permanent_parameter_count: usize,
    pub permanent_parameter_bytes: usize,
    pub reusable_kernel_word_count: usize,
    pub loaded_kernel_artifact_count: usize,
    pub dispatch_count: usize,
    pub descriptor_count: usize,
    pub model_boundary_descriptor_count: usize,
    pub incoming_cable_descriptor_count: usize,
    pub outgoing_cable_descriptor_count: usize,
    pub tick_plan: RuntimeDeviceTickPlanReport,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimePlacementReport {
    pub ok: bool,
    pub package_manifest: PathBuf,
    pub context_window_activations: usize,
    pub runtime_patch: RuntimePatchControls,
    pub device_bindings: RuntimeDeviceBindings,
    pub bound_devices: Vec<RuntimeBoundDevice>,
    pub cable_routes: RuntimeCableRoutes,
    pub device_count: usize,
    pub device_ids: Vec<String>,
    pub devices: Vec<RuntimeDeviceSliceReport>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeTokenizerOptionsReport {
    pub add_special_tokens: bool,
    pub skip_special_tokens: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimePlacedTransportStatsReport {
    pub pending_packet_count: usize,
    pub pending_byte_count: usize,
    pub pending_direct_cable_count: usize,
    pub pending_direct_byte_count: usize,
    pub published_packet_count: usize,
    pub published_byte_count: usize,
    pub received_packet_count: usize,
    pub received_byte_count: usize,
    pub direct_copy_count: usize,
    pub direct_copy_byte_count: usize,
    pub direct_receive_count: usize,
    pub direct_receive_byte_count: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimePlacedTransportReport {
    pub published_packet_count: usize,
    pub published_byte_count: usize,
    pub received_packet_count: usize,
    pub received_byte_count: usize,
    pub direct_copy_count: usize,
    pub direct_copy_byte_count: usize,
    pub direct_receive_count: usize,
    pub direct_receive_byte_count: usize,
    pub by_tick: Vec<RuntimePlacedTransportStatsReport>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimePromptTimingReport {
    pub setup_time_ns: u64,
    pub run_time_ns: u64,
    pub total_time_ns: u64,
    pub generated_token_count: usize,
    pub tick_count: usize,
    pub scheduler_turn_count: usize,
    pub average_generated_token_time_ns: Option<u64>,
    pub average_tick_time_ns: Option<u64>,
    pub average_scheduler_turn_time_ns: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimePlacedPedalDispatchTimingReport {
    pub dispatch_index: usize,
    pub kernel_id: String,
    pub node_id: String,
    pub op: String,
    pub reusable_family_id: String,
    pub run_time_ns: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimePlacedPedalTimingReport {
    pub stream_tick: u64,
    pub device_id: String,
    pub pedal_id: String,
    pub dispatch_count: usize,
    pub run_time_ns: u64,
    pub average_dispatch_time_ns: Option<u64>,
    pub dispatches: Vec<RuntimePlacedPedalDispatchTimingReport>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimePlacedPedalTimingSummaryReport {
    pub device_id: String,
    pub pedal_id: String,
    pub tick_count: usize,
    pub dispatch_count: usize,
    pub total_run_time_ns: u64,
    pub average_tick_time_ns: Option<u64>,
    pub average_dispatch_time_ns: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RuntimePromptBenchmarkU64MetricReport {
    pub total: u64,
    pub min: u64,
    pub max: u64,
    pub average: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RuntimePromptBenchmarkUsizeMetricReport {
    pub total: usize,
    pub min: usize,
    pub max: usize,
    pub average: f64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimePromptBenchmarkTransportTotalsReport {
    pub published_packet_count: usize,
    pub published_byte_count: usize,
    pub received_packet_count: usize,
    pub received_byte_count: usize,
    pub direct_copy_count: usize,
    pub direct_copy_byte_count: usize,
    pub direct_receive_count: usize,
    pub direct_receive_byte_count: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RuntimePromptBenchmarkRunReport {
    pub run_index: usize,
    pub execution_mode: String,
    pub stop_reason: String,
    pub generated_token_count: usize,
    pub tick_count: usize,
    pub scheduler_turn_count: usize,
    pub setup_time_ns: u64,
    pub run_time_ns: u64,
    pub total_time_ns: u64,
    pub generated_tokens_per_second: Option<f64>,
    pub transport: Option<RuntimePromptBenchmarkTransportTotalsReport>,
    pub pedal_timing_summaries: Vec<RuntimePlacedPedalTimingSummaryReport>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RuntimePromptBenchmarkReport {
    pub ok: bool,
    pub execution_mode: String,
    pub package_manifest: PathBuf,
    pub tokenizer_dir: PathBuf,
    pub runtime_patch: RuntimePatchControls,
    pub device_bindings: RuntimeDeviceBindings,
    pub device_count: usize,
    pub device_ids: Vec<String>,
    pub profile_runs: usize,
    pub prompt_text: String,
    pub prompt_ids: Vec<u32>,
    pub max_new_tokens: usize,
    pub setup_time_ns: RuntimePromptBenchmarkU64MetricReport,
    pub run_time_ns: RuntimePromptBenchmarkU64MetricReport,
    pub total_time_ns: RuntimePromptBenchmarkU64MetricReport,
    pub generated_token_count: RuntimePromptBenchmarkUsizeMetricReport,
    pub tick_count: RuntimePromptBenchmarkUsizeMetricReport,
    pub scheduler_turn_count: RuntimePromptBenchmarkUsizeMetricReport,
    pub generated_tokens_per_second: Option<f64>,
    pub stop_reasons: BTreeMap<String, usize>,
    pub transport_totals: Option<RuntimePromptBenchmarkTransportTotalsReport>,
    pub pedal_timing_summaries: Vec<RuntimePlacedPedalTimingSummaryReport>,
    pub runs: Vec<RuntimePromptBenchmarkRunReport>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeSingleDevicePromptRunReport {
    pub ok: bool,
    pub execution_mode: String,
    pub package_manifest: PathBuf,
    pub tokenizer_dir: PathBuf,
    pub device_name: String,
    pub device_id: String,
    pub runtime_patch: RuntimePatchControls,
    pub device_bindings: RuntimeDeviceBindings,
    pub pedal_count: usize,
    pub dispatches_per_tick: usize,
    pub descriptors_per_tick: usize,
    pub push_constant_bytes_per_tick: u32,
    pub context_window_activations: usize,
    pub scheduled_token_activations: usize,
    pub tokenizer: RuntimeTokenizerOptionsReport,
    pub prompt_text: String,
    pub prompt_ids: Vec<u32>,
    pub generated_ids: Vec<u32>,
    pub generated_text: String,
    pub output_text: String,
    pub stop_reason: String,
    pub scheduler_turns: usize,
    pub runtime_cycles: usize,
    pub timing: RuntimePromptTimingReport,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimePlacedPromptRunReport {
    pub ok: bool,
    pub execution_mode: String,
    pub package_manifest: PathBuf,
    pub tokenizer_dir: PathBuf,
    pub input_device_id: String,
    pub output_device_id: String,
    pub device_count: usize,
    pub device_ids: Vec<String>,
    pub bound_devices: Vec<RuntimeBoundDevice>,
    pub cable_routes: RuntimeCableRoutes,
    pub runtime_patch: RuntimePatchControls,
    pub device_bindings: RuntimeDeviceBindings,
    pub hosted_pedal_count: usize,
    pub context_window_activations: usize,
    pub scheduled_token_activations: usize,
    pub tokenizer: RuntimeTokenizerOptionsReport,
    pub prompt_text: String,
    pub prompt_ids: Vec<u32>,
    pub generated_ids: Vec<u32>,
    pub generated_text: String,
    pub output_text: String,
    pub stop_reason: String,
    pub tick_count: usize,
    pub scheduler_turns: usize,
    pub max_scheduler_turns_per_tick: usize,
    pub completed_stage_deltas: Vec<usize>,
    pub transport: RuntimePlacedTransportReport,
    pub timing: RuntimePromptTimingReport,
    pub pedal_timings: Vec<RuntimePlacedPedalTimingReport>,
    pub pedal_timing_summaries: Vec<RuntimePlacedPedalTimingSummaryReport>,
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

    #[test]
    fn placement_plan_keeps_layer_pedals_as_deployable_units() {
        let resolved =
            ResolvedLoweredPedalboard::from_index_file(fixture_model_index_path()).unwrap();

        let placement = resolved.single_device_placement_plan("gpu0").unwrap();

        assert_eq!(placement.schema, STREAM_CIRCUIT_PLACEMENT_SCHEMA);
        assert_eq!(placement.wiring, "explicit_graph");
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
    fn bypassed_instance_is_retained_in_draft_and_removed_from_execution_graph() {
        let source =
            ResolvedLoweredPedalboard::from_index_file(fixture_model_index_path()).unwrap();
        let patch = StreamCircuitRuntimePatch::from_source_series(&source, "gpu0")
            .unwrap()
            .with_instance_enabled("layer_01", false)
            .unwrap();

        let effective = source.instantiate_runtime_patch(&patch).unwrap();
        let placement = effective.placement_plan(&patch.placement_spec()).unwrap();

        assert_eq!(patch.instances.len(), 14);
        assert!(!patch.instances[1].enabled);
        assert_eq!(effective.circuits.len(), 13);
        assert!(
            effective
                .circuits
                .iter()
                .all(|circuit| circuit.pedal.id != "layer_01")
        );
        assert_eq!(placement.cables[0].source_pedal_id, "layer_00");
        assert_eq!(placement.cables[0].destination_pedal_id, "layer_02");
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
    fn runtime_device_bindings_treat_cpu_targets_as_direct_runtime_devices() {
        let logical_device_ids = vec!["cpu0".to_string(), "gpu0".to_string()];
        let bindings = RuntimeDeviceBindings::from_vulkan_targets(
            &logical_device_ids,
            &BTreeMap::new(),
            Some(0),
            |target| match target {
                "cpu0" => Ok(Some(6)),
                raw if raw.starts_with("vulkan:") => raw
                    .strip_prefix("vulkan:")
                    .unwrap()
                    .parse::<usize>()
                    .map(Some)
                    .map_err(|error| {
                        format!("invalid Vulkan physical device reference {target:?}: {error}")
                    }),
                _ => Ok(None),
            },
        );

        assert_eq!(bindings.requested_vulkan_device_indices, vec![0, 6]);
        assert!(bindings.can_mount_in_process);
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
                ("cpu0", Some("vulkan:6"), "device_id"),
                ("gpu0", Some("vulkan:0"), "process_default"),
            ]
        );
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
    fn runtime_available_device_serializes_inventory_entries() {
        let available = RuntimeAvailableDevice {
            device_id: "vulkan:5".to_string(),
            backend: "vulkan_compute".to_string(),
            available: true,
            runtime_device_id: None,
            physical_device_id: Some("vulkan:5".to_string()),
            physical_device_index: Some(5),
            device_name: Some("Radeon Test Device".to_string()),
            device_type: Some("discrete_gpu".to_string()),
            vendor_id: Some(4098),
            raw_device_id: Some(29_567),
            api_version: Some(4_203_000),
            driver_version: Some(1_024),
            compute_queue_family_indices: Some(vec![0, 2]),
            memory_heaps: Some(vec![RuntimeAvailableMemoryHeap {
                heap_index: 0,
                size_bytes: 8 * 1024 * 1024 * 1024,
                device_local: true,
            }]),
            selected_by_default: Some(false),
            selected_by_runtime: Some(false),
            runtime_binding: Some("inventory_only".to_string()),
            can_host_runtime_pedals_on_physical_device: Some(true),
            notes: vec![
                "auto-detected by Vulkan inventory; can be selected with --bind-device LOGICAL=vulkan:N"
                    .to_string(),
            ],
            error: None,
        };
        let unavailable = RuntimeAvailableDevice {
            device_id: "runtime_default".to_string(),
            backend: "vulkan_compute".to_string(),
            available: false,
            runtime_device_id: None,
            physical_device_id: None,
            physical_device_index: None,
            device_name: None,
            device_type: None,
            vendor_id: None,
            raw_device_id: None,
            api_version: None,
            driver_version: None,
            compute_queue_family_indices: None,
            memory_heaps: None,
            selected_by_default: None,
            selected_by_runtime: None,
            runtime_binding: None,
            can_host_runtime_pedals_on_physical_device: None,
            notes: vec!["no compute-capable Vulkan physical devices were found".to_string()],
            error: None,
        };

        let available_payload = serde_json::to_value(&available).unwrap();
        assert_eq!(available_payload["device_id"], "vulkan:5");
        assert_eq!(available_payload["physical_device_index"], 5);
        assert_eq!(available_payload["memory_heaps"][0]["device_local"], true);
        assert_eq!(
            available_payload["can_host_runtime_pedals_on_physical_device"],
            true
        );
        assert!(available_payload.get("runtime_device_id").is_none());

        let unavailable_payload = serde_json::to_value(&unavailable).unwrap();
        assert_eq!(unavailable_payload["device_id"], "runtime_default");
        assert_eq!(unavailable_payload["available"], false);
        assert!(unavailable_payload.get("physical_device_id").is_none());
        assert!(unavailable_payload.get("error").is_none());
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
    fn runtime_topology_report_serializes_ui_facing_contract() {
        let logical_device_ids = vec!["gpu0".to_string()];
        let bindings = RuntimeDeviceBindings::from_vulkan_targets(
            &logical_device_ids,
            &BTreeMap::new(),
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
        let source_pedal = RuntimeSourcePedal {
            pedal_index: 0,
            pedal_id: "layer_00".to_string(),
            operator_type: "layer".to_string(),
            implementation: "vulkan_resident".to_string(),
            behavioral_role: "transformer_layer".to_string(),
            source_layer_index: 0,
            circuit_id: "layer_00_circuit_v1".to_string(),
            input_ports: Vec::new(),
            output_ports: Vec::new(),
            state_port_count: 0,
            parameter_ref_count: 0,
            node_count: 0,
            kernel_count: 0,
        };
        let report = RuntimeTopologyReport {
            ok: true,
            schema: RUNTIME_TOPOLOGY_SCHEMA.to_string(),
            package_manifest: PathBuf::from("package.json"),
            package_root: PathBuf::from("."),
            package_id: "model-test".to_string(),
            compiled_schema: "llmoop.vulkan_resident_model_package.v1".to_string(),
            config_path: "config.json".to_string(),
            tokenizer: serde_json::json!({"path": "tokenizer"}),
            available_devices: vec![RuntimeAvailableDevice {
                device_id: "gpu0".to_string(),
                backend: "vulkan_compute".to_string(),
                available: true,
                runtime_device_id: Some("gpu0".to_string()),
                physical_device_id: Some("vulkan:0".to_string()),
                physical_device_index: Some(0),
                device_name: Some("Radeon Test Device".to_string()),
                device_type: Some("discrete_gpu".to_string()),
                vendor_id: Some(4098),
                raw_device_id: Some(29_567),
                api_version: Some(4_203_000),
                driver_version: Some(1_024),
                compute_queue_family_indices: Some(vec![0]),
                memory_heaps: Some(Vec::new()),
                selected_by_default: Some(true),
                selected_by_runtime: Some(true),
                runtime_binding: Some("default_local_vulkan_target".to_string()),
                can_host_runtime_pedals_on_physical_device: Some(true),
                notes: Vec::new(),
                error: None,
            }],
            compiled: RuntimeCompiledPedalboardSummary {
                wiring: "series".to_string(),
                default_device_id: "runtime_default".to_string(),
                pedal_devices: BTreeMap::new(),
                source_pedal_count: 1,
                source_pedals: vec![source_pedal],
                max_context_activations: 16,
            },
            runtime_patch_controls: RuntimePatchControls {
                default_device_id: Some("gpu0".to_string()),
                pedal_devices: BTreeMap::new(),
                source_chain: None,
                duplicate_after: vec![RuntimePatchDuplicateAfterControl {
                    after_instance_id: "layer_00".to_string(),
                    new_instance_id: "layer_00_repeat".to_string(),
                }],
            },
            runtime_patch: StreamCircuitRuntimePatch {
                schema: STREAM_CIRCUIT_RUNTIME_PATCH_SCHEMA.to_string(),
                wiring: "explicit_graph".to_string(),
                default_device_id: "gpu0".to_string(),
                instances: vec![StreamCircuitPedalInstance {
                    instance_id: "layer_00".to_string(),
                    source_pedal_id: "layer_00".to_string(),
                    device_id: "gpu0".to_string(),
                    enabled: true,
                    control_values: BTreeMap::new(),
                    state_policy: StreamCircuitPedalInstanceStatePolicy::Fresh,
                }],
                cables: Vec::new(),
            },
            effective: RuntimeEffectivePedalboardTopology {
                wiring: "series".to_string(),
                pedal_count: 1,
                cable_count: 0,
                local_cable_count: 0,
                cross_device_cable_count: 0,
                device_count: 1,
                device_ids: vec!["gpu0".to_string()],
                device_bindings: bindings,
                cable_routes: RuntimeCableRoutes {
                    schema: RUNTIME_CABLE_ROUTES_SCHEMA.to_string(),
                    cable_count: 0,
                    logical_local_cable_count: 0,
                    logical_cross_device_cable_count: 0,
                    same_physical_target_cable_count: 0,
                    cross_physical_target_cable_count: 0,
                    unresolved_target_cable_count: 0,
                    routes: Vec::new(),
                },
                pedals: vec![PedalPlacement {
                    pedal_index: 0,
                    pedal_id: "layer_00".to_string(),
                    circuit_id: "layer_00_circuit_v1".to_string(),
                    operator_type: "layer".to_string(),
                    device_id: "gpu0".to_string(),
                }],
                cables: Vec::new(),
            },
        };

        let payload = serde_json::to_value(&report).unwrap();

        assert_eq!(payload["schema"], RUNTIME_TOPOLOGY_SCHEMA);
        assert_eq!(payload["compiled"]["default_device_id"], "runtime_default");
        assert_eq!(
            payload["available_devices"][0]["physical_device_id"],
            "vulkan:0"
        );
        assert_eq!(
            payload["runtime_patch_controls"]["duplicate_after"][0]["new_instance_id"],
            "layer_00_repeat"
        );
        assert_eq!(
            payload["effective"]["device_bindings"]["can_mount_in_process"],
            true
        );
        assert_eq!(payload["effective"]["pedals"][0]["pedal_id"], "layer_00");
    }

    #[test]
    fn runtime_package_inspection_report_serializes_box_of_parts_contract() {
        let report = RuntimePackageInspectionReport {
            ok: true,
            package_manifest: PathBuf::from("package.json"),
            package_root: PathBuf::from("."),
            schema: "llmoop.vulkan_resident_model_package.v1".to_string(),
            package_id: "model-test".to_string(),
            config_path: "config.json".to_string(),
            tokenizer: serde_json::json!({"path": "tokenizer"}),
            compiled_wiring: "series".to_string(),
            compiled_default_device_id: "runtime_default".to_string(),
            compiled_pedal_devices: BTreeMap::new(),
            runtime_patch: RuntimePatchControls {
                default_device_id: None,
                pedal_devices: BTreeMap::new(),
                source_chain: None,
                duplicate_after: Vec::new(),
            },
            device_bindings: RuntimeDeviceBindings::from_vulkan_targets(
                &Vec::<String>::new(),
                &BTreeMap::new(),
                Some(0),
                |target| {
                    if let Some(index) = target.strip_prefix("vulkan:") {
                        return index.parse::<usize>().map(Some).map_err(|error| {
                            format!("invalid Vulkan physical device reference {target:?}: {error}")
                        });
                    }
                    Ok(None)
                },
            ),
            max_context_activations: 16,
            source_pedal_count: 0,
            source_pedals: Vec::new(),
            available_devices: Vec::new(),
        };

        let payload = serde_json::to_value(&report).unwrap();

        assert_eq!(payload["package_id"], "model-test");
        assert_eq!(payload["compiled_default_device_id"], "runtime_default");
        assert_eq!(
            payload["runtime_patch"]["default_device_id"],
            serde_json::Value::Null
        );
        assert_eq!(payload["source_pedal_count"], 0);
    }

    #[test]
    fn runtime_patch_inspection_report_serializes_patch_preview_contract() {
        let report = RuntimePatchInspectionReport {
            ok: true,
            package_manifest: PathBuf::from("package.json"),
            package_root: PathBuf::from("."),
            package_id: "model-test".to_string(),
            compiled_source_pedal_count: 14,
            runtime_patch_controls: RuntimePatchControls {
                default_device_id: Some("gpu0".to_string()),
                pedal_devices: BTreeMap::new(),
                source_chain: Some(vec![RuntimePatchSourceChainEntry {
                    instance_id: "layer_05_repeat".to_string(),
                    source_pedal_id: "layer_05".to_string(),
                }]),
                duplicate_after: Vec::new(),
            },
            runtime_patch: StreamCircuitRuntimePatch {
                schema: STREAM_CIRCUIT_RUNTIME_PATCH_SCHEMA.to_string(),
                wiring: "explicit_graph".to_string(),
                default_device_id: "gpu0".to_string(),
                instances: vec![StreamCircuitPedalInstance {
                    instance_id: "layer_05_repeat".to_string(),
                    source_pedal_id: "layer_05".to_string(),
                    device_id: "vulkan:5".to_string(),
                    enabled: true,
                    control_values: BTreeMap::new(),
                    state_policy: StreamCircuitPedalInstanceStatePolicy::Fresh,
                }],
                cables: Vec::new(),
            },
            device_bindings: RuntimeDeviceBindings::from_vulkan_targets(
                &["vulkan:5".to_string()],
                &BTreeMap::new(),
                Some(0),
                |target| {
                    if let Some(index) = target.strip_prefix("vulkan:") {
                        return index.parse::<usize>().map(Some).map_err(|error| {
                            format!("invalid Vulkan physical device reference {target:?}: {error}")
                        });
                    }
                    Ok(None)
                },
            ),
            effective_pedal_count: 1,
            effective_cable_count: 0,
            placement: RuntimePatchPlacementReport {
                schema: STREAM_CIRCUIT_PLACEMENT_SCHEMA.to_string(),
                wiring: "series".to_string(),
                local_cable_count: 0,
                cross_device_cable_count: 0,
                runtime_routes: RuntimeCableRoutes {
                    schema: RUNTIME_CABLE_ROUTES_SCHEMA.to_string(),
                    cable_count: 0,
                    logical_local_cable_count: 0,
                    logical_cross_device_cable_count: 0,
                    same_physical_target_cable_count: 0,
                    cross_physical_target_cable_count: 0,
                    unresolved_target_cable_count: 0,
                    routes: Vec::new(),
                },
                pedals: vec![PedalPlacement {
                    pedal_index: 0,
                    pedal_id: "layer_05_repeat".to_string(),
                    circuit_id: "layer_05_circuit_v1".to_string(),
                    operator_type: "layer".to_string(),
                    device_id: "vulkan:5".to_string(),
                }],
                cables: Vec::new(),
            },
        };

        let payload = serde_json::to_value(&report).unwrap();

        assert_eq!(payload["compiled_source_pedal_count"], 14);
        assert_eq!(
            payload["runtime_patch"]["instances"][0]["device_id"],
            "vulkan:5"
        );
        assert_eq!(
            payload["runtime_patch_controls"]["source_chain"][0]["instance_id"],
            "layer_05_repeat"
        );
        assert_eq!(payload["placement"]["pedals"][0]["device_id"], "vulkan:5");
    }

    #[test]
    fn runtime_device_slice_report_serializes_mounted_device_contract() {
        let report = RuntimeDeviceSliceReport {
            ok: true,
            package_manifest: PathBuf::from("package.json"),
            device_name: "Radeon Test Device".to_string(),
            device_id: "gpu1".to_string(),
            context_window_activations: 16,
            hosted_pedals: vec!["layer_05".to_string(), "layer_06".to_string()],
            local_cables: vec![RuntimeLocalCableBufferReport {
                cable_index: 5,
                signal: "hidden_state".to_string(),
                source_pedal_id: "layer_05".to_string(),
                destination_pedal_id: "layer_06".to_string(),
                device_id: "gpu1".to_string(),
                byte_capacity: Some(4096),
            }],
            incoming_cables: vec![RuntimeRemoteCableBufferReport {
                cable_index: 4,
                signal: "hidden_state".to_string(),
                source_device_id: "gpu0".to_string(),
                source_pedal_id: "layer_04".to_string(),
                destination_device_id: "gpu1".to_string(),
                destination_pedal_id: "layer_05".to_string(),
                byte_capacity: Some(4096),
            }],
            outgoing_cables: vec![RuntimeRemoteCableBufferReport {
                cable_index: 6,
                signal: "hidden_state".to_string(),
                source_device_id: "gpu1".to_string(),
                source_pedal_id: "layer_06".to_string(),
                destination_device_id: "gpu2".to_string(),
                destination_pedal_id: "layer_07".to_string(),
                byte_capacity: Some(4096),
            }],
            hosted_pedal_count: 2,
            incoming_cable_count: 1,
            outgoing_cable_count: 1,
            permanent_parameter_count: 12,
            permanent_parameter_bytes: 2048,
            reusable_kernel_word_count: 128,
            loaded_kernel_artifact_count: 4,
            dispatch_count: 8,
            descriptor_count: 24,
            model_boundary_descriptor_count: 2,
            incoming_cable_descriptor_count: 1,
            outgoing_cable_descriptor_count: 1,
            tick_plan: RuntimeDeviceTickPlanReport {
                stage_count: 4,
                receive_stage_count: 1,
                dispatch_stage_count: 2,
                publish_stage_count: 1,
                local_cable_read_count: 1,
                local_cable_write_count: 1,
                incoming_cable_read_count: 1,
                outgoing_cable_write_count: 1,
                model_input_read_count: 0,
                model_output_write_count: 0,
                can_execute: true,
            },
        };

        let payload = serde_json::to_value(&report).unwrap();

        assert_eq!(payload["device_id"], "gpu1");
        assert_eq!(payload["hosted_pedals"][0], "layer_05");
        assert_eq!(payload["local_cables"][0]["byte_capacity"], 4096);
        assert_eq!(payload["incoming_cables"][0]["source_device_id"], "gpu0");
        assert_eq!(
            payload["outgoing_cables"][0]["destination_device_id"],
            "gpu2"
        );
        assert_eq!(payload["tick_plan"]["can_execute"], true);
    }

    #[test]
    fn runtime_placement_report_serializes_device_slice_collection() {
        let report = RuntimePlacementReport {
            ok: true,
            package_manifest: PathBuf::from("package.json"),
            context_window_activations: 16,
            runtime_patch: RuntimePatchControls {
                default_device_id: Some("gpu0".to_string()),
                pedal_devices: BTreeMap::new(),
                source_chain: None,
                duplicate_after: Vec::new(),
            },
            device_bindings: RuntimeDeviceBindings::from_vulkan_targets(
                &["gpu0".to_string()],
                &BTreeMap::new(),
                Some(0),
                |target| {
                    if let Some(index) = target.strip_prefix("vulkan:") {
                        return index.parse::<usize>().map(Some).map_err(|error| {
                            format!("invalid Vulkan physical device reference {target:?}: {error}")
                        });
                    }
                    Ok(None)
                },
            ),
            bound_devices: vec![RuntimeBoundDevice {
                device_id: "gpu0".to_string(),
                target: Some("vulkan:0".to_string()),
                physical_device_index: Some(0),
                device_name: "Radeon Test Device".to_string(),
            }],
            cable_routes: RuntimeCableRoutes {
                schema: RUNTIME_CABLE_ROUTES_SCHEMA.to_string(),
                cable_count: 0,
                logical_local_cable_count: 0,
                logical_cross_device_cable_count: 0,
                same_physical_target_cable_count: 0,
                cross_physical_target_cable_count: 0,
                unresolved_target_cable_count: 0,
                routes: Vec::new(),
            },
            device_count: 1,
            device_ids: vec!["gpu0".to_string()],
            devices: Vec::new(),
        };

        let payload = serde_json::to_value(&report).unwrap();

        assert_eq!(payload["device_count"], 1);
        assert_eq!(payload["device_ids"][0], "gpu0");
        assert_eq!(payload["bound_devices"][0]["target"], "vulkan:0");
        assert_eq!(
            payload["device_bindings"]["logical_devices"][0]["binding_source"],
            "process_default"
        );
    }

    #[test]
    fn runtime_prompt_run_reports_serialize_execution_contracts() {
        let bindings = RuntimeDeviceBindings::from_vulkan_targets(
            &["gpu0".to_string()],
            &BTreeMap::new(),
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
        let tokenizer = RuntimeTokenizerOptionsReport {
            add_special_tokens: true,
            skip_special_tokens: true,
        };
        let timing = RuntimePromptTimingReport {
            setup_time_ns: 10,
            run_time_ns: 90,
            total_time_ns: 100,
            generated_token_count: 1,
            tick_count: 1,
            scheduler_turn_count: 1,
            average_generated_token_time_ns: Some(90),
            average_tick_time_ns: Some(90),
            average_scheduler_turn_time_ns: Some(90),
        };
        let single = RuntimeSingleDevicePromptRunReport {
            ok: true,
            execution_mode: "single_device_resident".to_string(),
            package_manifest: PathBuf::from("package.json"),
            tokenizer_dir: PathBuf::from("tokenizer"),
            device_name: "Radeon Test Device".to_string(),
            device_id: "gpu0".to_string(),
            runtime_patch: RuntimePatchControls {
                default_device_id: Some("gpu0".to_string()),
                pedal_devices: BTreeMap::new(),
                source_chain: None,
                duplicate_after: Vec::new(),
            },
            device_bindings: bindings.clone(),
            pedal_count: 14,
            dispatches_per_tick: 42,
            descriptors_per_tick: 64,
            push_constant_bytes_per_tick: 128,
            context_window_activations: 16,
            scheduled_token_activations: 2,
            tokenizer: tokenizer.clone(),
            prompt_text: "Hello".to_string(),
            prompt_ids: vec![1],
            generated_ids: vec![2],
            generated_text: " world".to_string(),
            output_text: "Hello world".to_string(),
            stop_reason: "max_new_tokens".to_string(),
            scheduler_turns: 1,
            runtime_cycles: 1,
            timing: timing.clone(),
        };
        let placed = RuntimePlacedPromptRunReport {
            ok: true,
            execution_mode: "placed_in_process".to_string(),
            package_manifest: PathBuf::from("package.json"),
            tokenizer_dir: PathBuf::from("tokenizer"),
            input_device_id: "gpu0".to_string(),
            output_device_id: "gpu1".to_string(),
            device_count: 1,
            device_ids: vec!["gpu0".to_string()],
            bound_devices: Vec::new(),
            cable_routes: RuntimeCableRoutes {
                schema: RUNTIME_CABLE_ROUTES_SCHEMA.to_string(),
                cable_count: 0,
                logical_local_cable_count: 0,
                logical_cross_device_cable_count: 0,
                same_physical_target_cable_count: 0,
                cross_physical_target_cable_count: 0,
                unresolved_target_cable_count: 0,
                routes: Vec::new(),
            },
            runtime_patch: RuntimePatchControls {
                default_device_id: Some("gpu0".to_string()),
                pedal_devices: BTreeMap::new(),
                source_chain: None,
                duplicate_after: Vec::new(),
            },
            device_bindings: bindings,
            hosted_pedal_count: 14,
            context_window_activations: 16,
            scheduled_token_activations: 2,
            tokenizer,
            prompt_text: "Hello".to_string(),
            prompt_ids: vec![1],
            generated_ids: vec![2],
            generated_text: " world".to_string(),
            output_text: "Hello world".to_string(),
            stop_reason: "max_new_tokens".to_string(),
            tick_count: 1,
            scheduler_turns: 1,
            max_scheduler_turns_per_tick: 1024,
            completed_stage_deltas: vec![42],
            transport: RuntimePlacedTransportReport {
                published_packet_count: 0,
                published_byte_count: 0,
                received_packet_count: 0,
                received_byte_count: 0,
                direct_copy_count: 2,
                direct_copy_byte_count: 4096,
                direct_receive_count: 2,
                direct_receive_byte_count: 4096,
                by_tick: vec![RuntimePlacedTransportStatsReport {
                    pending_packet_count: 0,
                    pending_byte_count: 0,
                    pending_direct_cable_count: 0,
                    pending_direct_byte_count: 0,
                    published_packet_count: 0,
                    published_byte_count: 0,
                    received_packet_count: 0,
                    received_byte_count: 0,
                    direct_copy_count: 2,
                    direct_copy_byte_count: 4096,
                    direct_receive_count: 2,
                    direct_receive_byte_count: 4096,
                }],
            },
            timing,
            pedal_timings: vec![RuntimePlacedPedalTimingReport {
                stream_tick: 0,
                device_id: "gpu0".to_string(),
                pedal_id: "layer_00".to_string(),
                dispatch_count: 1,
                run_time_ns: 90,
                average_dispatch_time_ns: Some(90),
                dispatches: vec![RuntimePlacedPedalDispatchTimingReport {
                    dispatch_index: 0,
                    kernel_id: "matmul".to_string(),
                    node_id: "layer_00.matmul".to_string(),
                    op: "linear".to_string(),
                    reusable_family_id: "linear".to_string(),
                    run_time_ns: 90,
                }],
            }],
            pedal_timing_summaries: vec![RuntimePlacedPedalTimingSummaryReport {
                device_id: "gpu0".to_string(),
                pedal_id: "layer_00".to_string(),
                tick_count: 1,
                dispatch_count: 1,
                total_run_time_ns: 90,
                average_tick_time_ns: Some(90),
                average_dispatch_time_ns: Some(90),
            }],
        };
        let benchmark_transport = RuntimePromptBenchmarkTransportTotalsReport {
            published_packet_count: 0,
            published_byte_count: 0,
            received_packet_count: 0,
            received_byte_count: 0,
            direct_copy_count: 2,
            direct_copy_byte_count: 4096,
            direct_receive_count: 2,
            direct_receive_byte_count: 4096,
        };
        let benchmark = RuntimePromptBenchmarkReport {
            ok: true,
            execution_mode: "placed_in_process".to_string(),
            package_manifest: PathBuf::from("package.json"),
            tokenizer_dir: PathBuf::from("tokenizer"),
            runtime_patch: placed.runtime_patch.clone(),
            device_bindings: placed.device_bindings.clone(),
            device_count: 1,
            device_ids: vec!["gpu0".to_string()],
            profile_runs: 1,
            prompt_text: "Hello".to_string(),
            prompt_ids: vec![1],
            max_new_tokens: 1,
            setup_time_ns: RuntimePromptBenchmarkU64MetricReport {
                total: 10,
                min: 10,
                max: 10,
                average: 10.0,
            },
            run_time_ns: RuntimePromptBenchmarkU64MetricReport {
                total: 90,
                min: 90,
                max: 90,
                average: 90.0,
            },
            total_time_ns: RuntimePromptBenchmarkU64MetricReport {
                total: 100,
                min: 100,
                max: 100,
                average: 100.0,
            },
            generated_token_count: RuntimePromptBenchmarkUsizeMetricReport {
                total: 1,
                min: 1,
                max: 1,
                average: 1.0,
            },
            tick_count: RuntimePromptBenchmarkUsizeMetricReport {
                total: 1,
                min: 1,
                max: 1,
                average: 1.0,
            },
            scheduler_turn_count: RuntimePromptBenchmarkUsizeMetricReport {
                total: 1,
                min: 1,
                max: 1,
                average: 1.0,
            },
            generated_tokens_per_second: Some(11_111_111.111),
            stop_reasons: BTreeMap::from([("max_new_tokens".to_string(), 1)]),
            transport_totals: Some(benchmark_transport.clone()),
            pedal_timing_summaries: placed.pedal_timing_summaries.clone(),
            runs: vec![RuntimePromptBenchmarkRunReport {
                run_index: 0,
                execution_mode: "placed_in_process".to_string(),
                stop_reason: "max_new_tokens".to_string(),
                generated_token_count: 1,
                tick_count: 1,
                scheduler_turn_count: 1,
                setup_time_ns: 10,
                run_time_ns: 90,
                total_time_ns: 100,
                generated_tokens_per_second: Some(11_111_111.111),
                transport: Some(benchmark_transport),
                pedal_timing_summaries: placed.pedal_timing_summaries.clone(),
            }],
        };

        let single_payload = serde_json::to_value(&single).unwrap();
        let placed_payload = serde_json::to_value(&placed).unwrap();
        let benchmark_payload = serde_json::to_value(&benchmark).unwrap();

        assert_eq!(single_payload["execution_mode"], "single_device_resident");
        assert_eq!(single_payload["generated_ids"][0], 2);
        assert_eq!(placed_payload["execution_mode"], "placed_in_process");
        assert_eq!(placed_payload["transport"]["direct_copy_count"], 2);
        assert_eq!(placed_payload["completed_stage_deltas"][0], 42);
        assert_eq!(single_payload["timing"]["total_time_ns"], 100);
        assert_eq!(
            placed_payload["timing"]["average_generated_token_time_ns"],
            90
        );
        assert_eq!(placed_payload["pedal_timings"][0]["pedal_id"], "layer_00");
        assert_eq!(placed_payload["pedal_timings"][0]["run_time_ns"], 90);
        assert_eq!(
            placed_payload["pedal_timings"][0]["dispatches"][0]["node_id"],
            "layer_00.matmul"
        );
        assert_eq!(
            placed_payload["pedal_timing_summaries"][0]["total_run_time_ns"],
            90
        );
        assert_eq!(benchmark_payload["profile_runs"], 1);
        assert_eq!(benchmark_payload["run_time_ns"]["average"], 90.0);
        assert_eq!(
            benchmark_payload["transport_totals"]["direct_copy_byte_count"],
            4096
        );
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
        assert_eq!(patch.wiring, "explicit_graph");
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
            .duplicate_after_instance(&resolved, "layer_05", "layer_05_repeat")
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
        assert_eq!(
            duplicate.circuit.state_ports,
            instantiated.circuits[original_index].circuit.state_ports
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
    fn runtime_patch_execution_order_comes_from_cables_not_instance_storage_order() {
        let resolved =
            ResolvedLoweredPedalboard::from_index_file(fixture_model_index_path()).unwrap();
        let chain = vec![
            ("first".to_string(), "layer_00".to_string()),
            ("second".to_string(), "layer_01".to_string()),
            ("third".to_string(), "layer_02".to_string()),
        ];
        let mut patch =
            StreamCircuitRuntimePatch::from_source_chain(&resolved, "gpu0", &chain).unwrap();
        patch.instances.reverse();

        let instantiated = patch.instantiate_graph(&resolved).unwrap();

        assert_eq!(
            instantiated
                .circuits
                .iter()
                .map(|artifact| artifact.pedal.id.as_str())
                .collect::<Vec<_>>(),
            vec!["first", "second", "third"]
        );
    }

    #[test]
    fn runtime_patch_validates_state_policy_targets_compatibility_and_cycles() {
        let resolved =
            ResolvedLoweredPedalboard::from_index_file(fixture_model_index_path()).unwrap();
        let mut patch = resolved
            .default_runtime_patch("gpu0")
            .unwrap()
            .duplicate_after_instance(&resolved, "layer_05", "layer_05_repeat")
            .unwrap();
        patch
            .instances
            .iter_mut()
            .find(|instance| instance.instance_id == "layer_05_repeat")
            .unwrap()
            .state_policy = StreamCircuitPedalInstanceStatePolicy::ShareWith {
            instance_id: "layer_05".to_string(),
        };
        patch.validate_against_graph(&resolved).unwrap();

        let mut cross_device_share = patch.clone();
        cross_device_share
            .instances
            .iter_mut()
            .find(|instance| instance.instance_id == "layer_05_repeat")
            .unwrap()
            .device_id = "gpu1".to_string();
        assert!(
            cross_device_share
                .validate_against_graph(&resolved)
                .unwrap_err()
                .0
                .contains("cannot share state across devices")
        );

        let mut cycle = patch;
        cycle
            .instances
            .iter_mut()
            .find(|instance| instance.instance_id == "layer_05")
            .unwrap()
            .state_policy = StreamCircuitPedalInstanceStatePolicy::CloneFrom {
            instance_id: "layer_05_repeat".to_string(),
        };
        assert!(
            cycle
                .validate_against_graph(&resolved)
                .unwrap_err()
                .0
                .contains("dependency cycle")
        );
    }

    #[test]
    fn runtime_patch_rejects_disconnected_and_implicit_fanout_graphs() {
        let resolved =
            ResolvedLoweredPedalboard::from_index_file(fixture_model_index_path()).unwrap();
        let patch = resolved.default_runtime_patch("gpu0").unwrap();

        let mut disconnected = patch.clone();
        disconnected.cables.remove(4);
        assert!(
            disconnected
                .validate_against_graph(&resolved)
                .unwrap_err()
                .0
                .contains("exactly one model input and one model output")
        );

        let mut fanout = patch;
        let mut duplicate = fanout.cables[0].clone();
        duplicate.id = "implicit_fanout".to_string();
        duplicate.destination = fanout.cables[1].destination.clone();
        fanout.cables.push(duplicate);
        assert!(
            fanout
                .validate_against_graph(&resolved)
                .unwrap_err()
                .0
                .contains("use an explicit splitter pedal")
        );
    }
}
