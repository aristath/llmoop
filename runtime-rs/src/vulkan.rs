use serde::{Deserialize, Serialize};

pub const VULKAN_SPIRV_BACKEND_ID: &str = "vulkan_spirv";
pub const DEFAULT_SPIRV_ENTRY_POINT: &str = "main";

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpirvPedalProgram {
    pub pedal_id: String,
    pub operator_type: String,
    pub entry_point: String,
    pub specialization: Option<String>,
    pub words: Vec<u32>,
}

impl SpirvPedalProgram {
    pub fn new(
        pedal_id: impl Into<String>,
        operator_type: impl Into<String>,
        words: Vec<u32>,
    ) -> Self {
        Self {
            pedal_id: pedal_id.into(),
            operator_type: operator_type.into(),
            entry_point: DEFAULT_SPIRV_ENTRY_POINT.to_string(),
            specialization: None,
            words,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VulkanBackendDescriptor {
    pub backend_id: String,
    pub device_id: String,
    pub queue_family: Option<u32>,
    pub programs: Vec<SpirvPedalProgram>,
}

impl VulkanBackendDescriptor {
    pub fn empty(device_id: impl Into<String>) -> Self {
        Self {
            backend_id: VULKAN_SPIRV_BACKEND_ID.to_string(),
            device_id: device_id.into(),
            queue_family: None,
            programs: Vec::new(),
        }
    }
}
