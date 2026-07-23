#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StreamCircuitPlacementSpec {
    pub schema: String,
    pub default_device_id: String,
    #[serde(default)]
    pub node_devices: BTreeMap<String, String>,
}

impl StreamCircuitPlacementSpec {
    pub fn new(default_device_id: impl Into<String>) -> Self {
        Self {
            schema: STREAM_CIRCUIT_PLACEMENT_SCHEMA.to_string(),
            default_device_id: default_device_id.into(),
            node_devices: BTreeMap::new(),
        }
    }

    pub fn with_component_device(
        mut self,
        component_id: impl Into<String>,
        device_id: impl Into<String>,
    ) -> Self {
        self.node_devices.insert(component_id.into(), device_id.into());
        self
    }

    pub fn device_for_component(&self, component_id: &str) -> &str {
        self.node_devices
            .get(component_id)
            .map(String::as_str)
            .unwrap_or(&self.default_device_id)
    }
}
