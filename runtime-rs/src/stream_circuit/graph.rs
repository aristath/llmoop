use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

pub const STREAM_CIRCUIT_SCHEMA: &str = "nerve.stream_circuit.v1";
pub const SEMANTIC_MODULE_TREE_SCHEMA: &str = "nerve.semantic_module_tree.v1";
pub const CIRCUIT_PARAMS_SCHEMA: &str = "nerve.circuit_params.v1";
pub const CIRCUIT_STATE_SCHEMA: &str = "nerve.circuit_state.v1";
pub const LOWERED_EXECUTION_GRAPH_SCHEMA: &str = "nerve.lowered_execution_graph.v1";
pub const STREAM_CIRCUIT_PLACEMENT_SCHEMA: &str = "nerve.stream_circuit_placement.v1";
pub const STREAM_CIRCUIT_RUNTIME_GRAPH_SCHEMA: &str = "nerve.stream_circuit_runtime_graph.v1";
pub const RUNTIME_DEFAULT_LOGICAL_DEVICE_ID: &str = "runtime_default";
pub const RUNTIME_EDGE_ROUTES_SCHEMA: &str = "nerve.runtime_edge_routes.v1";
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
    pub component_port: Option<String>,
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
    pub component_id: String,
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
pub struct SemanticModule {
    pub id: String,
    pub role: String,
    pub responsibility: String,
    #[serde(default)]
    pub parent_id: Option<String>,
    #[serde(default)]
    pub child_ids: Vec<String>,
    #[serde(default)]
    pub source_node_ids: Vec<String>,
    #[serde(default)]
    pub parameter_ref_ids: Vec<String>,
    #[serde(default)]
    pub owned_state_port_ids: Vec<String>,
    #[serde(default)]
    pub input_signals: Vec<String>,
    #[serde(default)]
    pub output_signals: Vec<String>,
    #[serde(default, rename = "virtual")]
    pub r#virtual: bool,
    #[serde(default)]
    pub attrs: Value,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SemanticModuleTree {
    pub schema: String,
    pub root_module_id: String,
    pub modules: Vec<SemanticModule>,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub semantic_module_tree: Option<SemanticModuleTree>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub semantic_execution_nodes: Vec<CircuitNode>,
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

        if let Some(tree) = &self.semantic_module_tree {
            let semantic_nodes = if self.semantic_execution_nodes.is_empty() {
                &self.nodes
            } else {
                &self.semantic_execution_nodes
            };
            let semantic_node_ids = semantic_nodes
                .iter()
                .map(|node| node.id.clone())
                .collect::<BTreeSet<_>>();
            let mut semantic_signals = self
                .boundary
                .inputs
                .iter()
                .map(|port| port.id.clone())
                .collect::<BTreeSet<_>>();
            semantic_signals.extend(
                semantic_nodes
                    .iter()
                    .flat_map(|node| node.outputs.iter().cloned()),
            );
            self.validate_semantic_module_tree(
                tree,
                semantic_nodes,
                &semantic_node_ids,
                &state_ids,
                &param_ids,
                &semantic_signals,
                &mut issues,
            );
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

    fn validate_semantic_module_tree(
        &self,
        tree: &SemanticModuleTree,
        semantic_nodes: &[CircuitNode],
        node_ids: &BTreeSet<String>,
        state_ids: &BTreeSet<&String>,
        param_ids: &BTreeSet<&String>,
        known_signals: &BTreeSet<String>,
        issues: &mut Vec<String>,
    ) {
        if tree.schema != SEMANTIC_MODULE_TREE_SCHEMA {
            issues.push(format!(
                "{} has unsupported semantic module tree schema {:?}",
                self.id, tree.schema
            ));
        }
        if tree.modules.is_empty() {
            issues.push(format!("{} semantic module tree must not be empty", self.id));
            return;
        }
        let modules = tree
            .modules
            .iter()
            .map(|module| (module.id.as_str(), module))
            .collect::<BTreeMap<_, _>>();
        if modules.len() != tree.modules.len() {
            issues.push(format!("{} has duplicate semantic module ids", self.id));
        }
        let Some(root) = modules.get(tree.root_module_id.as_str()) else {
            issues.push(format!(
                "{} semantic module root {:?} does not resolve",
                self.id, tree.root_module_id
            ));
            return;
        };
        if root.parent_id.is_some() {
            issues.push(format!(
                "{} semantic module root {:?} must not have a parent",
                self.id, tree.root_module_id
            ));
        }

        let mut visited = BTreeSet::new();
        let mut active = BTreeSet::new();
        visit_semantic_module(
            tree.root_module_id.as_str(),
            &modules,
            &mut visited,
            &mut active,
            issues,
        );
        if visited.len() != modules.len() {
            let unreachable = modules
                .keys()
                .filter(|module_id| !visited.contains(**module_id))
                .copied()
                .collect::<Vec<_>>();
            issues.push(format!(
                "{} semantic modules are unreachable from the root: {:?}",
                self.id, unreachable
            ));
        }

        let node_by_id = semantic_nodes
            .iter()
            .map(|node| (node.id.as_str(), node))
            .collect::<BTreeMap<_, _>>();
        let mut node_owners = BTreeMap::new();
        let mut state_owners = BTreeMap::new();
        for module in &tree.modules {
            if module.id.is_empty()
                || module.role.is_empty()
                || module.responsibility.is_empty()
            {
                issues.push(format!(
                    "{} semantic modules require non-empty id, role, and responsibility",
                    self.id
                ));
            }
            for node_id in &module.source_node_ids {
                if !node_ids.contains(node_id) {
                    issues.push(format!(
                        "{} semantic module {} references unknown source node {:?}",
                        self.id, module.id, node_id
                    ));
                } else if let Some(previous) =
                    node_owners.insert(node_id.as_str(), module.id.as_str())
                {
                    issues.push(format!(
                        "{} source node {:?} belongs to semantic modules {:?} and {:?}",
                        self.id, node_id, previous, module.id
                    ));
                }
                if let Some(node) = node_by_id.get(node_id.as_str()) {
                    for parameter in &node.params {
                        if !module.parameter_ref_ids.contains(parameter) {
                            issues.push(format!(
                                "{} semantic module {} omits parameter {:?} used by source node {:?}",
                                self.id, module.id, parameter, node_id
                            ));
                        }
                    }
                }
            }
            for parameter in &module.parameter_ref_ids {
                if !param_ids.contains(parameter) {
                    issues.push(format!(
                        "{} semantic module {} references unknown parameter {:?}",
                        self.id, module.id, parameter
                    ));
                }
            }
            for state in &module.owned_state_port_ids {
                if !state_ids.contains(state) {
                    issues.push(format!(
                        "{} semantic module {} owns unknown state {:?}",
                        self.id, module.id, state
                    ));
                } else if let Some(previous) =
                    state_owners.insert(state.as_str(), module.id.as_str())
                {
                    issues.push(format!(
                        "{} state {:?} belongs to semantic modules {:?} and {:?}",
                        self.id, state, previous, module.id
                    ));
                }
            }
            for signal in module
                .input_signals
                .iter()
                .chain(module.output_signals.iter())
            {
                if !known_signals.contains(signal) {
                    issues.push(format!(
                        "{} semantic module {} references unknown signal {:?}",
                        self.id, module.id, signal
                    ));
                }
            }
        }
        if node_owners.keys().copied().collect::<BTreeSet<_>>()
            != node_ids.iter().map(String::as_str).collect()
        {
            issues.push(format!(
                "{} semantic module tree does not cover every source node exactly once",
                self.id
            ));
        }
        if state_owners.keys().copied().collect::<BTreeSet<_>>()
            != state_ids.iter().map(|state| state.as_str()).collect()
        {
            issues.push(format!(
                "{} semantic module tree does not own every state port exactly once",
                self.id
            ));
        }
    }
}

fn visit_semantic_module<'a>(
    module_id: &'a str,
    modules: &BTreeMap<&'a str, &'a SemanticModule>,
    visited: &mut BTreeSet<&'a str>,
    active: &mut BTreeSet<&'a str>,
    issues: &mut Vec<String>,
) {
    if !active.insert(module_id) {
        issues.push(format!(
            "semantic module tree contains a cycle at {module_id:?}"
        ));
        return;
    }
    if !visited.insert(module_id) {
        active.remove(module_id);
        return;
    }
    let module = modules[module_id];
    for child_id in &module.child_ids {
        let Some(child) = modules.get(child_id.as_str()) else {
            issues.push(format!(
                "semantic module {:?} references unknown child {:?}",
                module.id, child_id
            ));
            continue;
        };
        if child.parent_id.as_deref() != Some(module_id) {
            issues.push(format!(
                "semantic module {:?} child {:?} does not point back to its parent",
                module.id, child_id
            ));
        }
        visit_semantic_module(child_id, modules, visited, active, issues);
    }
    active.remove(module_id);
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
    if port.component_port.as_deref().is_none_or(str::is_empty) {
        issues.push(format!(
            "boundary {direction} port {:?} must map to a non-empty component_port",
            port.id
        ));
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LoweredExecutionGraphSource {
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
pub struct StreamCircuitEdgeEndpoint {
    pub component_id: String,
    pub port_id: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StreamCircuitGraphEdge {
    pub id: String,
    pub source: StreamCircuitEdgeEndpoint,
    pub destination: StreamCircuitEdgeEndpoint,
    pub connection: StreamCircuitConnection,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StreamCircuitGraphBoundaryPort {
    pub id: String,
    pub endpoint: StreamCircuitEdgeEndpoint,
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

    pub fn validate(&self, edge_id: &str) -> Result<(), CircuitPlacementError> {
        if let Self::TemporalFeedback { delay_activations } = self
            && *delay_activations == 0
        {
            return Err(CircuitPlacementError(format!(
                "runtime graph temporal feedback edge {edge_id} must delay at least one activation"
            )));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LoweredExecutionGraphGraph {
    pub topology: String,
    #[serde(default)]
    pub circuits: Vec<LoweredCircuitRef>,
    pub edges: Vec<StreamCircuitGraphEdge>,
    pub boundary: StreamCircuitGraphBoundary,
    #[serde(default)]
    pub input_transducer: Value,
    #[serde(default)]
    pub output_transducer: Value,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LoweredExecutionGraphSummary {
    pub circuit_count: usize,
    #[serde(default)]
    pub operator_counts: BTreeMap<String, usize>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LoweredExecutionGraph {
    pub schema: String,
    pub source: LoweredExecutionGraphSource,
    #[serde(default)]
    pub architecture: Value,
    #[serde(default)]
    pub dimensions: Value,
    pub graph: LoweredExecutionGraphGraph,
    pub summary: LoweredExecutionGraphSummary,
    #[serde(default)]
    pub notes: Vec<String>,
}

impl LoweredExecutionGraph {
    pub fn from_json_file(path: impl AsRef<Path>) -> Result<Self, CircuitArtifactError> {
        read_json(path)
    }

    pub fn validate_index(&self) -> Result<(), CircuitArtifactError> {
        let mut issues = Vec::new();
        if self.schema != LOWERED_EXECUTION_GRAPH_SCHEMA {
            issues.push(format!(
                "unsupported lowered execution_graph schema {:?}",
                self.schema
            ));
        }
        if self.graph.circuits.is_empty() {
            issues.push("lowered execution_graph contains no circuits".to_string());
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

        let mut edge_ids = BTreeSet::new();
        for edge in &self.graph.edges {
            if edge.id.is_empty() || !edge_ids.insert(edge.id.clone()) {
                issues.push(format!(
                    "invalid or duplicate graph edge id {:?}",
                    edge.id
                ));
            }
            if !ids.contains(&edge.source.component_id) {
                issues.push(format!(
                    "graph edge {} references unknown source component {:?}",
                    edge.id, edge.source.component_id
                ));
            }
            if !ids.contains(&edge.destination.component_id) {
                issues.push(format!(
                    "graph edge {} references unknown destination component {:?}",
                    edge.id, edge.destination.component_id
                ));
            }
            if matches!(
                edge.connection,
                StreamCircuitConnection::TemporalFeedback {
                    delay_activations: 0
                }
            ) {
                issues.push(format!(
                    "graph temporal feedback edge {} must delay at least one activation",
                    edge.id
                ));
            }
        }

        if issues.is_empty() {
            Ok(())
        } else {
            Err(CircuitArtifactError(format!(
                "lowered execution_graph validation failed:\n- {}",
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
        issues.push(format!("lowered execution_graph declares no {kind}s"));
        return;
    }
    let mut ids = BTreeSet::new();
    let mut endpoints = BTreeSet::new();
    for port in ports {
        if port.id.is_empty() || !ids.insert(port.id.as_str()) {
            issues.push(format!("invalid or duplicate {kind} id {:?}", port.id));
        }
        if !circuit_ids.contains(&port.endpoint.component_id) {
            issues.push(format!(
                "{kind} {} references unknown component {:?}",
                port.id, port.endpoint.component_id
            ));
        }
        if port.endpoint.port_id.is_empty()
            || !endpoints.insert((
                port.endpoint.component_id.as_str(),
                port.endpoint.port_id.as_str(),
            ))
        {
            issues.push(format!(
                "{kind} {} has an empty or duplicate endpoint {}.{}",
                port.id, port.endpoint.component_id, port.endpoint.port_id
            ));
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedLoweredExecutionGraph {
    pub artifact_root: PathBuf,
    pub index: LoweredExecutionGraph,
    pub circuits: Vec<ResolvedCircuitArtifact>,
}

impl ResolvedLoweredExecutionGraph {
    pub fn from_index_file(path: impl AsRef<Path>) -> Result<Self, CircuitArtifactError> {
        let path = path.as_ref();
        let artifact_root = path
            .parent()
            .ok_or_else(|| {
                CircuitArtifactError(format!(
                    "lowered execution_graph path {:?} does not have a parent directory",
                    path
                ))
            })?
            .to_path_buf();
        let index = LoweredExecutionGraph::from_json_file(path)?;
        index.validate_index()?;

        let mut circuits = Vec::with_capacity(index.graph.circuits.len());
        for component in &index.graph.circuits {
            let circuit = StreamCircuit::from_json_file(resolve_artifact_path(
                &artifact_root,
                &component.circuit,
            ))?;
            let params = CircuitParamsArtifact::from_json_file(resolve_artifact_path(
                &artifact_root,
                &component.params,
            ))?;
            let state = CircuitStateArtifact::from_json_file(resolve_artifact_path(
                &artifact_root,
                &component.state,
            ))?;
            let resolved = ResolvedCircuitArtifact {
                component: component.clone(),
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

    pub fn default_runtime_graph(
        &self,
        default_device_id: impl Into<String>,
    ) -> Result<StreamCircuitRuntimeGraph, CircuitPlacementError> {
        StreamCircuitRuntimeGraph::from_source_series(self, default_device_id)
    }

    pub fn runtime_graph_from_placement(
        &self,
        spec: &StreamCircuitPlacementSpec,
    ) -> Result<StreamCircuitRuntimeGraph, CircuitPlacementError> {
        StreamCircuitRuntimeGraph::from_placement_spec(self, spec)
    }

    pub fn instantiate_runtime_graph(
        &self,
        runtime_graph: &StreamCircuitRuntimeGraph,
    ) -> Result<ResolvedLoweredExecutionGraph, CircuitPlacementError> {
        runtime_graph.instantiate_graph(self)
    }
}
