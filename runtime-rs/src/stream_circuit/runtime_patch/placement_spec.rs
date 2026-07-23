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
