#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RuntimeEditorSourceComponent {
    pub source_id: String,
    pub layer_index: Option<usize>,
    pub operator_type: String,
    pub runtime_role: CircuitRuntimeRole,
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
    pub layer_index: Option<usize>,
    pub occurrence: usize,
    pub device_id: String,
    pub enabled: bool,
    pub control_values: BTreeMap<String, Value>,
    pub state_policy: StreamCircuitNodeInstanceStatePolicy,
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
    manifest: VulkanResidentModelPackageManifest,
    source_graph: ResolvedLoweredExecutionGraph,
    source_components: Vec<RuntimeEditorSourceComponent>,
    source_by_layer: BTreeMap<usize, Vec<String>>,
    source_ids: BTreeSet<String>,
    available_devices: Vec<RuntimeAvailableDevice>,
    draft: StreamCircuitRuntimeGraph,
}
