use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    CircuitPlacementError, CircuitRuntimeRole, ComponentPlacement, RUNTIME_DEFAULT_LOGICAL_DEVICE_ID,
    ResolvedLoweredExecutionGraph, RuntimeAvailableDevice, RuntimeAvailableMemoryHeap,
    StreamCircuitNodeInstance, StreamCircuitNodeInstanceStatePolicy, StreamCircuitPlacementPlan,
    StreamCircuitRuntimeGraph, VulkanComputeDevice, VulkanResidentModelPackageManifest,
};

pub const RUNTIME_PACKAGE_MANIFEST_FILE: &str = "vulkan_resident_package.json";

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
