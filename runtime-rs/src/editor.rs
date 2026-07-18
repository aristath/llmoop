use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    CircuitPlacementError, PedalPlacement, ResolvedLoweredPedalboard, RuntimeAvailableDevice,
    RuntimeAvailableMemoryHeap, StreamCircuitPedalInstance, StreamCircuitPedalInstanceStatePolicy,
    StreamCircuitPlacementPlan, StreamCircuitRuntimePatch, VulkanComputeDevice,
    VulkanResidentGreedyModelPackageManifest,
};

pub const RUNTIME_PACKAGE_MANIFEST_FILE: &str = "vulkan_resident_greedy_package.json";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RuntimeEditorError(pub String);

impl Display for RuntimeEditorError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for RuntimeEditorError {}

impl From<std::io::Error> for RuntimeEditorError {
    fn from(error: std::io::Error) -> Self {
        Self(error.to_string())
    }
}

impl From<CircuitPlacementError> for RuntimeEditorError {
    fn from(error: CircuitPlacementError) -> Self {
        Self(error.to_string())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeModelPathKind {
    CompiledPackage { manifest: PathBuf },
    SafetensorsSource { model_dir: PathBuf },
}

pub fn classify_runtime_model_path(
    path: impl AsRef<Path>,
) -> Result<RuntimeModelPathKind, RuntimeEditorError> {
    let path = path.as_ref();
    if path.is_file()
        && path.file_name().and_then(|name| name.to_str()) == Some(RUNTIME_PACKAGE_MANIFEST_FILE)
    {
        return Ok(RuntimeModelPathKind::CompiledPackage {
            manifest: path.to_path_buf(),
        });
    }
    if !path.is_dir() {
        return Err(RuntimeEditorError(format!(
            "model path does not exist or is not a directory: {}",
            path.display()
        )));
    }
    let manifest = path.join(RUNTIME_PACKAGE_MANIFEST_FILE);
    if manifest.is_file() {
        return Ok(RuntimeModelPathKind::CompiledPackage { manifest });
    }
    if path.join("config.json").is_file()
        && path.join("tokenizer.json").is_file()
        && path.read_dir()?.filter_map(Result::ok).any(|entry| {
            entry.path().extension().and_then(|value| value.to_str()) == Some("safetensors")
        })
    {
        return Ok(RuntimeModelPathKind::SafetensorsSource {
            model_dir: path.to_path_buf(),
        });
    }
    Err(RuntimeEditorError(format!(
        "{} is neither an llmoop package nor a discoverable Safetensors model",
        path.display()
    )))
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RuntimeEditorSourcePedal {
    pub source_id: String,
    pub layer_index: usize,
    pub operator_type: String,
    pub implementation: String,
    pub behavioral_role: String,
    pub input_shape: Vec<usize>,
    pub output_shape: Vec<usize>,
    pub state_ports: Vec<Value>,
    pub controls: Vec<Value>,
    pub control_schemas: Vec<RuntimeEditorControlSchema>,
    pub parameter_ref_count: usize,
    pub node_count: usize,
    pub kernel_count: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RuntimeEditorControlChoice {
    pub value: Value,
    pub label: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum RuntimeEditorControlKind {
    Boolean,
    Integer,
    Number,
    Text,
    Enumeration {
        choices: Vec<RuntimeEditorControlChoice>,
    },
    ReadOnly,
    Unsupported {
        declared_type: String,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RuntimeEditorControlSchema {
    pub id: String,
    pub name: String,
    pub description: Option<String>,
    pub kind: RuntimeEditorControlKind,
    pub current_value: Option<Value>,
    pub default_value: Option<Value>,
    pub minimum: Option<f64>,
    pub maximum: Option<f64>,
    pub step: Option<f64>,
    pub units: Option<String>,
    pub editable_at_runtime: bool,
    pub requires_state_reset: bool,
    pub requires_remount: bool,
    pub requires_recompile: bool,
    pub scope: String,
    pub raw: Value,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeEditorInstance {
    pub instance_id: String,
    pub source_id: String,
    pub layer_index: usize,
    pub occurrence: usize,
    pub device_id: String,
    pub enabled: bool,
    pub control_values: BTreeMap<String, Value>,
    pub state_policy: StreamCircuitPedalInstanceStatePolicy,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RuntimeEditorValidation {
    pub valid: bool,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    pub placement: Option<StreamCircuitPlacementPlan>,
}

#[derive(Clone, Debug)]
pub struct RuntimeModelEditor {
    package_manifest_path: PathBuf,
    package_root: PathBuf,
    manifest: VulkanResidentGreedyModelPackageManifest,
    source_graph: ResolvedLoweredPedalboard,
    source_pedals: Vec<RuntimeEditorSourcePedal>,
    source_by_layer: BTreeMap<usize, String>,
    available_devices: Vec<RuntimeAvailableDevice>,
    draft: StreamCircuitRuntimePatch,
}

impl RuntimeModelEditor {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, RuntimeEditorError> {
        Self::load_with_device_provider(path, |default_device_id| {
            discover_runtime_devices(default_device_id, None)
        })
    }

    pub fn load_with_available_devices(
        path: impl AsRef<Path>,
        available_devices: Vec<RuntimeAvailableDevice>,
    ) -> Result<Self, RuntimeEditorError> {
        Self::load_with_device_provider(path, |_| available_devices)
    }

    fn load_with_device_provider(
        path: impl AsRef<Path>,
        devices: impl FnOnce(&str) -> Vec<RuntimeAvailableDevice>,
    ) -> Result<Self, RuntimeEditorError> {
        let manifest_path = match classify_runtime_model_path(path)? {
            RuntimeModelPathKind::CompiledPackage { manifest } => manifest,
            RuntimeModelPathKind::SafetensorsSource { .. } => {
                return Err(RuntimeEditorError(
                    "Safetensors sources must be compiled before loading the runtime editor"
                        .to_string(),
                ));
            }
        };
        let package_root = manifest_path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));
        let manifest = VulkanResidentGreedyModelPackageManifest::from_json_file(&manifest_path)?;
        let source_graph = manifest
            .resolved_source_graph(package_root.clone())
            .map_err(|error| RuntimeEditorError(error.to_string()))?;
        let draft =
            StreamCircuitRuntimePatch::from_placement_spec(&source_graph, &manifest.placement)?;
        let source_pedals = source_pedals(&manifest);
        let source_by_layer = source_pedals
            .iter()
            .map(|pedal| (pedal.layer_index, pedal.source_id.clone()))
            .collect::<BTreeMap<_, _>>();
        if source_by_layer.len() != source_pedals.len() {
            return Err(RuntimeEditorError(
                "compiled package contains more than one source pedal for a layer index"
                    .to_string(),
            ));
        }
        let available_devices = devices(&manifest.placement.default_device_id);
        Ok(Self {
            package_manifest_path: manifest_path,
            package_root,
            manifest,
            source_graph,
            source_pedals,
            source_by_layer,
            available_devices,
            draft,
        })
    }

    pub fn package_manifest_path(&self) -> &Path {
        &self.package_manifest_path
    }

    pub fn package_root(&self) -> &Path {
        &self.package_root
    }

    pub fn package_id(&self) -> &str {
        &self.manifest.package_id
    }

    pub fn max_context_activations(&self) -> usize {
        self.manifest.max_context_activations
    }

    pub fn source_pedals(&self) -> &[RuntimeEditorSourcePedal] {
        &self.source_pedals
    }

    pub fn available_devices(&self) -> &[RuntimeAvailableDevice] {
        &self.available_devices
    }

    pub fn refresh_devices(&mut self) {
        self.available_devices = discover_runtime_devices(&self.draft.default_device_id, None);
    }

    pub fn draft(&self) -> &StreamCircuitRuntimePatch {
        &self.draft
    }

    pub fn layer_sequence(&self) -> Vec<usize> {
        let layer_by_source = self
            .source_pedals
            .iter()
            .map(|pedal| (pedal.source_id.as_str(), pedal.layer_index))
            .collect::<BTreeMap<_, _>>();
        self.draft
            .instances
            .iter()
            .filter_map(|instance| {
                layer_by_source
                    .get(instance.source_pedal_id.as_str())
                    .copied()
            })
            .collect()
    }

    pub fn instances(&self) -> Vec<RuntimeEditorInstance> {
        let layer_by_source = self
            .source_pedals
            .iter()
            .map(|pedal| (pedal.source_id.as_str(), pedal.layer_index))
            .collect::<BTreeMap<_, _>>();
        let mut occurrences = BTreeMap::<&str, usize>::new();
        self.draft
            .instances
            .iter()
            .filter_map(|instance| {
                let layer_index = layer_by_source
                    .get(instance.source_pedal_id.as_str())
                    .copied()?;
                let occurrence = occurrences
                    .entry(instance.source_pedal_id.as_str())
                    .and_modify(|value| *value += 1)
                    .or_insert(1);
                Some(RuntimeEditorInstance {
                    instance_id: instance.instance_id.clone(),
                    source_id: instance.source_pedal_id.clone(),
                    layer_index,
                    occurrence: *occurrence,
                    device_id: instance.device_id.clone(),
                    enabled: instance.enabled,
                    control_values: instance.control_values.clone(),
                    state_policy: instance.state_policy.clone(),
                })
            })
            .collect()
    }

    pub fn replace_layer_sequence(
        &mut self,
        layer_sequence: &[usize],
    ) -> Result<(), RuntimeEditorError> {
        if layer_sequence.is_empty() {
            return Err(RuntimeEditorError(
                "layer sequence must contain at least one layer".to_string(),
            ));
        }
        let mut previous_by_source =
            BTreeMap::<String, VecDeque<StreamCircuitPedalInstance>>::new();
        for instance in &self.draft.instances {
            previous_by_source
                .entry(instance.source_pedal_id.clone())
                .or_default()
                .push_back(instance.clone());
        }
        let mut occurrence_by_source = BTreeMap::<String, usize>::new();
        let mut used_instance_ids = BTreeSet::new();
        let mut instances = Vec::with_capacity(layer_sequence.len());
        for layer_index in layer_sequence {
            let source_id = self.source_by_layer.get(layer_index).ok_or_else(|| {
                RuntimeEditorError(format!(
                    "unknown layer {layer_index}; available layers: {}",
                    available_layer_range(&self.source_by_layer)
                ))
            })?;
            let occurrence = occurrence_by_source
                .entry(source_id.clone())
                .and_modify(|value| *value += 1)
                .or_insert(1);
            let previous = previous_by_source
                .get_mut(source_id)
                .and_then(VecDeque::pop_front);
            let instance = if let Some(previous) = previous {
                used_instance_ids.insert(previous.instance_id.clone());
                previous
            } else {
                let instance_id = allocate_instance_id(source_id, *occurrence, &used_instance_ids);
                used_instance_ids.insert(instance_id.clone());
                StreamCircuitPedalInstance {
                    instance_id,
                    source_pedal_id: source_id.clone(),
                    device_id: self.draft.default_device_id.clone(),
                    enabled: true,
                    control_values: BTreeMap::new(),
                    state_policy: StreamCircuitPedalInstanceStatePolicy::Fresh,
                }
            };
            instances.push(instance);
        }
        let candidate = StreamCircuitRuntimePatch {
            schema: self.draft.schema.clone(),
            wiring: self.draft.wiring.clone(),
            default_device_id: self.draft.default_device_id.clone(),
            instances,
        };
        candidate.validate_against_graph(&self.source_graph)?;
        self.draft = candidate;
        Ok(())
    }

    pub fn set_instance_device(
        &mut self,
        instance_id: &str,
        device_id: &str,
    ) -> Result<(), RuntimeEditorError> {
        let available = self.available_devices.iter().any(|device| {
            device.device_id == device_id
                && device.available
                && device.can_host_runtime_pedals_on_physical_device != Some(false)
        });
        if !available {
            return Err(RuntimeEditorError(format!(
                "runtime device {device_id:?} is unavailable or cannot host this pedal"
            )));
        }
        self.draft = self
            .draft
            .clone()
            .with_instance_device(instance_id, device_id)?;
        Ok(())
    }

    pub fn set_instance_enabled(
        &mut self,
        instance_id: &str,
        enabled: bool,
    ) -> Result<(), RuntimeEditorError> {
        let candidate = self
            .draft
            .clone()
            .with_instance_enabled(instance_id, enabled)?;
        candidate.validate_against_graph(&self.source_graph)?;
        self.draft = candidate;
        Ok(())
    }

    pub fn set_instance_control_value(
        &mut self,
        instance_id: &str,
        control_id: &str,
        value: Value,
    ) -> Result<(), RuntimeEditorError> {
        let source = self.source_pedal_for_instance(instance_id).ok_or_else(|| {
            RuntimeEditorError(format!(
                "runtime patch has no pedal instance {instance_id:?}"
            ))
        })?;
        let schema = source
            .control_schemas
            .iter()
            .find(|schema| schema.id == control_id)
            .ok_or_else(|| {
                RuntimeEditorError(format!(
                    "source pedal {} declares no control {control_id:?}",
                    source.source_id
                ))
            })?;
        validate_runtime_editor_control_value(schema, &value)?;
        let instance = self
            .draft
            .instances
            .iter_mut()
            .find(|instance| instance.instance_id == instance_id)
            .ok_or_else(|| {
                RuntimeEditorError(format!(
                    "runtime patch has no pedal instance {instance_id:?}"
                ))
            })?;
        instance
            .control_values
            .insert(control_id.to_string(), value);
        Ok(())
    }

    pub fn effective_instance_control_value(
        &self,
        instance_id: &str,
        control_id: &str,
    ) -> Option<Value> {
        let instance = self
            .draft
            .instances
            .iter()
            .find(|instance| instance.instance_id == instance_id)?;
        if let Some(value) = instance.control_values.get(control_id) {
            return Some(value.clone());
        }
        self.source_pedal_for_instance(instance_id)?
            .control_schemas
            .iter()
            .find(|schema| schema.id == control_id)
            .and_then(|schema| {
                schema
                    .current_value
                    .clone()
                    .or_else(|| schema.default_value.clone())
            })
    }

    pub fn set_instance_state_policy(
        &mut self,
        instance_id: &str,
        state_policy: StreamCircuitPedalInstanceStatePolicy,
    ) -> Result<(), RuntimeEditorError> {
        let instance = self
            .draft
            .instances
            .iter_mut()
            .find(|instance| instance.instance_id == instance_id)
            .ok_or_else(|| {
                RuntimeEditorError(format!(
                    "runtime patch has no pedal instance {instance_id:?}"
                ))
            })?;
        instance.state_policy = state_policy;
        Ok(())
    }

    pub fn validation(&self) -> RuntimeEditorValidation {
        let mut errors = Vec::new();
        for instance in &self.draft.instances {
            if !self.available_devices.iter().any(|device| {
                device.device_id == instance.device_id
                    && device.available
                    && device.can_host_runtime_pedals_on_physical_device != Some(false)
            }) {
                errors.push(format!(
                    "instance {} is assigned to unavailable device {}",
                    instance.instance_id, instance.device_id
                ));
            }
            if let Some(source) = self
                .source_pedals
                .iter()
                .find(|source| source.source_id == instance.source_pedal_id)
            {
                for (control_id, value) in &instance.control_values {
                    match source
                        .control_schemas
                        .iter()
                        .find(|schema| schema.id == *control_id)
                    {
                        Some(schema) => {
                            if let Err(error) = validate_runtime_editor_control_value(schema, value)
                            {
                                errors.push(format!(
                                    "instance {} control {}: {}",
                                    instance.instance_id, control_id, error
                                ));
                            }
                        }
                        None => errors.push(format!(
                            "instance {} has undeclared control {}",
                            instance.instance_id, control_id
                        )),
                    }
                }
            }
        }
        if let Err(error) = self.draft.validate_against_graph(&self.source_graph) {
            errors.push(error.to_string());
        }
        let placement = if errors.is_empty() {
            self.source_graph
                .instantiate_runtime_patch(&self.draft)
                .and_then(|graph| graph.placement_plan(&self.draft.placement_spec()))
                .map_err(|error| errors.push(error.to_string()))
                .ok()
        } else {
            None
        };
        RuntimeEditorValidation {
            valid: errors.is_empty(),
            errors,
            warnings: Vec::new(),
            placement,
        }
    }

    pub fn source_pedal_for_instance(
        &self,
        instance_id: &str,
    ) -> Option<&RuntimeEditorSourcePedal> {
        let source_id = self
            .draft
            .instances
            .iter()
            .find(|instance| instance.instance_id == instance_id)?
            .source_pedal_id
            .as_str();
        self.source_pedals
            .iter()
            .find(|pedal| pedal.source_id == source_id)
    }
}

#[cfg(test)]
pub(crate) fn load_runtime_model_editor_without_hardware(
    path: impl AsRef<Path>,
) -> Result<RuntimeModelEditor, RuntimeEditorError> {
    let path = path.as_ref();
    let manifest_path = match classify_runtime_model_path(path)? {
        RuntimeModelPathKind::CompiledPackage { manifest } => manifest,
        RuntimeModelPathKind::SafetensorsSource { .. } => {
            return Err(RuntimeEditorError(
                "test editor requires a compiled package".to_string(),
            ));
        }
    };
    let manifest = VulkanResidentGreedyModelPackageManifest::from_json_file(&manifest_path)?;
    let device_id = manifest.placement.default_device_id.clone();
    RuntimeModelEditor::load_with_available_devices(
        manifest_path,
        vec![RuntimeAvailableDevice {
            device_id: device_id.clone(),
            backend: "test".to_string(),
            available: true,
            runtime_device_id: Some(device_id),
            physical_device_id: Some("test:0".to_string()),
            physical_device_index: Some(0),
            device_name: Some("Deterministic test device".to_string()),
            device_type: Some("test".to_string()),
            vendor_id: None,
            raw_device_id: None,
            api_version: None,
            driver_version: None,
            compute_queue_family_indices: Some(vec![0]),
            memory_heaps: Some(Vec::new()),
            selected_by_default: Some(true),
            selected_by_runtime: Some(true),
            runtime_binding: Some("test_only".to_string()),
            can_host_runtime_pedals_on_physical_device: Some(true),
            notes: vec!["hardware discovery disabled for this test".to_string()],
            error: None,
        }],
    )
}

fn source_pedals(
    manifest: &VulkanResidentGreedyModelPackageManifest,
) -> Vec<RuntimeEditorSourcePedal> {
    let execution_by_pedal = manifest
        .pedal_executions
        .iter()
        .map(|execution| (execution.pedal_id.as_str(), execution))
        .collect::<BTreeMap<_, _>>();
    manifest
        .circuit_graph
        .pedals
        .iter()
        .map(|pedal| RuntimeEditorSourcePedal {
            source_id: pedal.pedal_id.clone(),
            layer_index: pedal.circuit.source.source_layer_index,
            operator_type: pedal.operator_type.clone(),
            implementation: pedal.implementation.clone(),
            behavioral_role: pedal.behavioral_role.clone(),
            input_shape: pedal
                .circuit
                .boundary
                .inputs
                .first()
                .map(|port| port.shape.clone())
                .unwrap_or_default(),
            output_shape: pedal
                .circuit
                .boundary
                .outputs
                .first()
                .map(|port| port.shape.clone())
                .unwrap_or_default(),
            state_ports: pedal
                .circuit
                .state_ports
                .iter()
                .filter_map(|state| serde_json::to_value(state).ok())
                .collect(),
            controls: pedal.circuit.boundary.controls.clone(),
            control_schemas: pedal
                .circuit
                .boundary
                .controls
                .iter()
                .enumerate()
                .map(|(index, control)| runtime_editor_control_schema(index, control))
                .collect(),
            parameter_ref_count: pedal.params.refs.len(),
            node_count: pedal.circuit.nodes.len(),
            kernel_count: execution_by_pedal
                .get(pedal.pedal_id.as_str())
                .map(|execution| execution.kernels.len())
                .unwrap_or(0),
        })
        .collect()
}

pub fn runtime_editor_control_schema(index: usize, raw: &Value) -> RuntimeEditorControlSchema {
    let object = raw.as_object();
    let string_field = |names: &[&str]| {
        object.and_then(|object| {
            names
                .iter()
                .find_map(|name| object.get(*name).and_then(Value::as_str))
                .map(ToOwned::to_owned)
        })
    };
    let bool_field = |names: &[&str]| {
        object.and_then(|object| {
            names
                .iter()
                .find_map(|name| object.get(*name).and_then(Value::as_bool))
        })
    };
    let value_field = |names: &[&str]| {
        object.and_then(|object| names.iter().find_map(|name| object.get(*name).cloned()))
    };
    let number_field = |names: &[&str]| {
        object.and_then(|object| {
            names
                .iter()
                .find_map(|name| object.get(*name).and_then(Value::as_f64))
        })
    };
    let id = string_field(&["id", "property_id"]).unwrap_or_else(|| format!("control_{index}"));
    let name = string_field(&["name", "label"]).unwrap_or_else(|| id.clone());
    let declared_type = string_field(&["value_type", "type"])
        .unwrap_or_else(|| "unspecified".to_string())
        .to_lowercase();
    let choices = object
        .and_then(|object| {
            ["choices", "values", "enum"]
                .into_iter()
                .find_map(|name| object.get(name).and_then(Value::as_array))
        })
        .map(|choices| {
            choices
                .iter()
                .map(|choice| {
                    if let Some(object) = choice.as_object() {
                        let value = object.get("value").cloned().unwrap_or(Value::Null);
                        let label = object
                            .get("label")
                            .and_then(Value::as_str)
                            .map(ToOwned::to_owned)
                            .unwrap_or_else(|| display_json_value(&value));
                        RuntimeEditorControlChoice { value, label }
                    } else {
                        RuntimeEditorControlChoice {
                            value: choice.clone(),
                            label: display_json_value(choice),
                        }
                    }
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let kind = match declared_type.as_str() {
        "bool" | "boolean" | "toggle" => RuntimeEditorControlKind::Boolean,
        "int" | "integer" | "u32" | "u64" | "i32" | "i64" => RuntimeEditorControlKind::Integer,
        "float" | "number" | "f32" | "f64" => RuntimeEditorControlKind::Number,
        "string" | "text" => RuntimeEditorControlKind::Text,
        "enum" | "enumeration" | "select" if !choices.is_empty() => {
            RuntimeEditorControlKind::Enumeration { choices }
        }
        "readonly" | "read_only" => RuntimeEditorControlKind::ReadOnly,
        _ => RuntimeEditorControlKind::Unsupported {
            declared_type: declared_type.clone(),
        },
    };
    RuntimeEditorControlSchema {
        id,
        name,
        description: string_field(&["description", "help"]),
        kind,
        current_value: value_field(&["current_value", "current", "value"]),
        default_value: value_field(&["default_value", "default"]),
        minimum: number_field(&["minimum", "min"]),
        maximum: number_field(&["maximum", "max"]),
        step: number_field(&["step"]),
        units: string_field(&["units", "unit"]),
        editable_at_runtime: bool_field(&["editable_at_runtime", "runtime_editable"])
            .unwrap_or(false),
        requires_state_reset: bool_field(&["requires_state_reset"]).unwrap_or(false),
        requires_remount: bool_field(&["requires_remount"]).unwrap_or(false),
        requires_recompile: bool_field(&["requires_recompile"]).unwrap_or(false),
        scope: string_field(&["scope"])
            .unwrap_or_else(|| "instance".to_string())
            .to_lowercase(),
        raw: raw.clone(),
    }
}

pub fn validate_runtime_editor_control_value(
    schema: &RuntimeEditorControlSchema,
    value: &Value,
) -> Result<(), RuntimeEditorError> {
    if !schema.editable_at_runtime {
        return Err(RuntimeEditorError(format!(
            "control {:?} is not editable at runtime",
            schema.id
        )));
    }
    if schema.scope != "instance" {
        return Err(RuntimeEditorError(format!(
            "control {:?} has {:?} scope; this editor changes pedal instances",
            schema.id, schema.scope
        )));
    }
    let numeric = match &schema.kind {
        RuntimeEditorControlKind::Boolean if value.is_boolean() => None,
        RuntimeEditorControlKind::Integer
            if value.as_i64().is_some() || value.as_u64().is_some() =>
        {
            value.as_f64()
        }
        RuntimeEditorControlKind::Number if value.as_f64().is_some() => value.as_f64(),
        RuntimeEditorControlKind::Text if value.is_string() => None,
        RuntimeEditorControlKind::Enumeration { choices }
            if choices.iter().any(|choice| choice.value == *value) =>
        {
            None
        }
        RuntimeEditorControlKind::ReadOnly => {
            return Err(RuntimeEditorError(format!(
                "control {:?} is read-only",
                schema.id
            )));
        }
        RuntimeEditorControlKind::Unsupported { declared_type } => {
            return Err(RuntimeEditorError(format!(
                "control {:?} uses unsupported value type {:?}",
                schema.id, declared_type
            )));
        }
        _ => {
            return Err(RuntimeEditorError(format!(
                "control {:?} received a value incompatible with its declared type",
                schema.id
            )));
        }
    };
    if let Some(numeric) = numeric {
        if schema.minimum.is_some_and(|minimum| numeric < minimum) {
            return Err(RuntimeEditorError(format!(
                "value {numeric} is below minimum {}",
                schema.minimum.unwrap_or_default()
            )));
        }
        if schema.maximum.is_some_and(|maximum| numeric > maximum) {
            return Err(RuntimeEditorError(format!(
                "value {numeric} is above maximum {}",
                schema.maximum.unwrap_or_default()
            )));
        }
        if let Some(step) = schema.step {
            if !step.is_finite() || step <= 0.0 {
                return Err(RuntimeEditorError(format!(
                    "control {:?} declares a non-positive or non-finite step",
                    schema.id
                )));
            }
            if matches!(schema.kind, RuntimeEditorControlKind::Integer)
                && (step - step.round()).abs() > f64::EPSILON
            {
                return Err(RuntimeEditorError(format!(
                    "integer control {:?} declares non-whole step {step}",
                    schema.id
                )));
            }
            let anchor = schema.minimum.unwrap_or(0.0);
            let step_position = (numeric - anchor) / step;
            let tolerance = 1e-9 * step_position.abs().max(1.0);
            if (step_position - step_position.round()).abs() > tolerance {
                return Err(RuntimeEditorError(format!(
                    "value {numeric} does not align to step {step} from {anchor}",
                )));
            }
        }
    }
    Ok(())
}

fn display_json_value(value: &Value) -> String {
    value
        .as_str()
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| value.to_string())
}

fn allocate_instance_id(
    source_id: &str,
    occurrence: usize,
    used_instance_ids: &BTreeSet<String>,
) -> String {
    let preferred = if occurrence == 1 {
        source_id.to_string()
    } else {
        format!("{source_id}@{occurrence}")
    };
    if !used_instance_ids.contains(&preferred) {
        return preferred;
    }
    let mut suffix = occurrence.max(2);
    loop {
        let candidate = format!("{source_id}@{suffix}");
        if !used_instance_ids.contains(&candidate) {
            return candidate;
        }
        suffix += 1;
    }
}

fn available_layer_range(source_by_layer: &BTreeMap<usize, String>) -> String {
    match (
        source_by_layer.keys().next().copied(),
        source_by_layer.keys().next_back().copied(),
    ) {
        (Some(first), Some(last)) if first != last => format!("{first}-{last}"),
        (Some(only), Some(_)) => only.to_string(),
        _ => "none".to_string(),
    }
}

pub fn discover_runtime_devices(
    default_device_id: &str,
    selected_vulkan_device_index: Option<usize>,
) -> Vec<RuntimeAvailableDevice> {
    match VulkanComputeDevice::available_compute_devices() {
        Ok(devices) if devices.is_empty() => vec![unavailable_device(
            default_device_id,
            "no compute-capable Vulkan physical devices were found",
            None,
        )],
        Ok(devices) => {
            let mut cpu_device_ordinal = 0usize;
            devices
                .iter()
                .map(|device| {
                    let selected_by_runtime = selected_vulkan_device_index
                        .map(|index| index == device.physical_device_index)
                        .unwrap_or(device.selected_by_default);
                    let cpu_runtime_device_id = if device.device_type == "cpu" {
                        let runtime_device_id = format!("cpu{cpu_device_ordinal}");
                        cpu_device_ordinal += 1;
                        Some(runtime_device_id)
                    } else {
                        None
                    };
                    let runtime_device_id = selected_by_runtime
                        .then(|| default_device_id.to_string())
                        .or(cpu_runtime_device_id.clone());
                    let device_id = runtime_device_id
                        .clone()
                        .unwrap_or_else(|| device.physical_device_id.clone());
                    RuntimeAvailableDevice {
                        device_id,
                        backend: "vulkan_compute".to_string(),
                        available: true,
                        runtime_device_id,
                        physical_device_id: Some(device.physical_device_id.clone()),
                        physical_device_index: Some(device.physical_device_index),
                        device_name: Some(device.device_name.clone()),
                        device_type: Some(device.device_type.clone()),
                        vendor_id: Some(device.vendor_id),
                        raw_device_id: Some(device.device_id),
                        api_version: Some(device.api_version),
                        driver_version: Some(device.driver_version),
                        compute_queue_family_indices: Some(
                            device.compute_queue_family_indices.clone(),
                        ),
                        memory_heaps: Some(
                            device
                                .memory_heaps
                                .iter()
                                .map(|heap| RuntimeAvailableMemoryHeap {
                                    heap_index: heap.heap_index,
                                    size_bytes: heap.size_bytes,
                                    device_local: heap.device_local,
                                })
                                .collect(),
                        ),
                        selected_by_default: Some(device.selected_by_default),
                        selected_by_runtime: Some(selected_by_runtime),
                        runtime_binding: Some(if selected_by_runtime {
                            "default_local_vulkan_target".to_string()
                        } else {
                            "inventory_only".to_string()
                        }),
                        can_host_runtime_pedals_on_physical_device: Some(true),
                        notes: if selected_by_runtime {
                            vec!["default target for unbound pedal instances".to_string()]
                        } else if let Some(cpu_runtime_device_id) = cpu_runtime_device_id {
                            vec![format!(
                                "CPU runtime target {cpu_runtime_device_id} backed by {}",
                                device.physical_device_id
                            )]
                        } else {
                            vec!["available runtime placement target".to_string()]
                        },
                        error: None,
                    }
                })
                .collect()
        }
        Err(error) => vec![unavailable_device(
            default_device_id,
            "Vulkan device discovery failed",
            Some(error.to_string()),
        )],
    }
}

fn unavailable_device(
    device_id: &str,
    note: &str,
    error: Option<String>,
) -> RuntimeAvailableDevice {
    RuntimeAvailableDevice {
        device_id: device_id.to_string(),
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
        notes: vec![note.to_string()],
        error,
    }
}

pub fn placement_pedals_by_instance(
    validation: &RuntimeEditorValidation,
) -> BTreeMap<&str, &PedalPlacement> {
    validation
        .placement
        .as_ref()
        .map(|placement| {
            placement
                .pedals
                .iter()
                .map(|pedal| (pedal.pedal_id.as_str(), pedal))
                .collect()
        })
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;

    #[test]
    fn classifies_compiled_packages_and_safetensors_sources() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "llmoop-editor-path-{}-{unique}",
            std::process::id()
        ));
        let package = root.join("package");
        let source = root.join("source");
        fs::create_dir_all(&package).unwrap();
        fs::create_dir_all(&source).unwrap();
        fs::write(package.join(RUNTIME_PACKAGE_MANIFEST_FILE), b"{}").unwrap();
        fs::write(source.join("config.json"), b"{}").unwrap();
        fs::write(source.join("tokenizer.json"), b"{}").unwrap();
        fs::write(source.join("model.safetensors"), b"").unwrap();

        assert_eq!(
            classify_runtime_model_path(&package).unwrap(),
            RuntimeModelPathKind::CompiledPackage {
                manifest: package.join(RUNTIME_PACKAGE_MANIFEST_FILE)
            }
        );
        assert_eq!(
            classify_runtime_model_path(&source).unwrap(),
            RuntimeModelPathKind::SafetensorsSource {
                model_dir: source.clone()
            }
        );

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn configured_package_editor_preserves_instances_while_reordering() {
        let package = std::env::var("LLMOOP_TEST_PACKAGE_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("..")
                    .join("packages")
                    .join("model_7760f415")
            });
        if !package.join(RUNTIME_PACKAGE_MANIFEST_FILE).is_file() {
            return;
        }
        let mut editor = load_runtime_model_editor_without_hardware(&package).unwrap();
        let original_first = editor.instances()[0].instance_id.clone();

        editor.replace_layer_sequence(&[0, 1, 1, 2]).unwrap();

        let instances = editor.instances();
        assert_eq!(editor.layer_sequence(), vec![0, 1, 1, 2]);
        assert_eq!(instances[0].instance_id, original_first);
        assert_eq!(instances[1].occurrence, 1);
        assert_eq!(instances[2].occurrence, 2);
        assert_ne!(instances[1].instance_id, instances[2].instance_id);
        assert!(editor.validation().valid);
    }

    #[test]
    fn generic_control_schema_preserves_constraints_and_rejects_bad_values() {
        let raw = serde_json::json!({
            "id": "attention_window",
            "name": "Attention window",
            "description": "Local temporal span",
            "value_type": "integer",
            "current": 4096,
            "default": 2048,
            "min": 128,
            "max": 8192,
            "step": 128,
            "units": "tokens",
            "editable_at_runtime": true,
            "requires_state_reset": true,
            "scope": "instance"
        });
        let schema = runtime_editor_control_schema(0, &raw);
        assert_eq!(schema.id, "attention_window");
        assert_eq!(schema.kind, RuntimeEditorControlKind::Integer);
        assert_eq!(schema.current_value, Some(serde_json::json!(4096)));
        assert_eq!(schema.minimum, Some(128.0));
        assert_eq!(schema.maximum, Some(8192.0));
        assert!(schema.requires_state_reset);
        assert!(validate_runtime_editor_control_value(&schema, &serde_json::json!(1024)).is_ok());
        assert!(
            validate_runtime_editor_control_value(&schema, &serde_json::json!(64))
                .unwrap_err()
                .to_string()
                .contains("below minimum")
        );
        assert!(
            validate_runtime_editor_control_value(&schema, &serde_json::json!(1000))
                .unwrap_err()
                .to_string()
                .contains("does not align to step")
        );

        let unsupported = runtime_editor_control_schema(
            1,
            &serde_json::json!({"id":"control","type":"pedal_control"}),
        );
        assert!(matches!(
            unsupported.kind,
            RuntimeEditorControlKind::Unsupported { .. }
        ));
        assert!(!unsupported.editable_at_runtime);
    }
}
