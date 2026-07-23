use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::fs;
use std::path::Path;

use serde::Deserialize;

use crate::stream_circuit::{
    CircuitNode, CircuitPort, ParameterRef, ResolvedCircuitArtifact, ResolvedLoweredPedalboard,
    StatePort, StreamCircuit,
};

pub const TENSOR_INDEX_SCHEMA: &str = "nerve.tensor_index.v1";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CircuitPlanError(pub String);

impl Display for CircuitPlanError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for CircuitPlanError {}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
pub struct TensorIndex {
    pub schema: String,
    #[serde(default)]
    pub tensors: BTreeMap<String, TensorMetadata>,
}

impl TensorIndex {
    pub fn from_json_file(path: impl AsRef<Path>) -> Result<Self, CircuitPlanError> {
        Self::from_json_file_with_package_containment(path, false)
    }

    pub fn from_package_json_file(path: impl AsRef<Path>) -> Result<Self, CircuitPlanError> {
        Self::from_json_file_with_package_containment(path, true)
    }

    fn from_json_file_with_package_containment(
        path: impl AsRef<Path>,
        require_package_containment: bool,
    ) -> Result<Self, CircuitPlanError> {
        let path = path.as_ref();
        let bytes = fs::read(path).map_err(|error| CircuitPlanError(error.to_string()))?;
        let mut index: Self =
            serde_json::from_slice(&bytes).map_err(|error| CircuitPlanError(error.to_string()))?;
        if index.schema != TENSOR_INDEX_SCHEMA {
            return Err(CircuitPlanError(format!(
                "unsupported tensor index schema {:?}",
                index.schema
            )));
        }
        let root = path.parent().unwrap_or_else(|| Path::new("."));
        for (tensor, metadata) in &mut index.tensors {
            if let Some(source_file) = &metadata.source_file {
                let source_path = Path::new(source_file);
                if require_package_containment
                    && (source_file.is_empty()
                        || source_path.is_absolute()
                        || source_path.components().any(|component| {
                            matches!(
                                component,
                                std::path::Component::ParentDir
                                    | std::path::Component::RootDir
                                    | std::path::Component::Prefix(_)
                            )
                        }))
                {
                    return Err(CircuitPlanError(format!(
                        "package tensor {tensor:?} source path {source_file:?} must stay inside the package"
                    )));
                }
                if !source_path.is_absolute() {
                    metadata.source_file =
                        Some(root.join(source_path).to_string_lossy().into_owned());
                }
            } else if require_package_containment {
                return Err(CircuitPlanError(format!(
                    "package tensor {tensor:?} has no source file"
                )));
            }
            if require_package_containment
                && metadata.data_sha256.as_deref().is_none_or(|digest| {
                    digest.len() != 64
                        || !digest
                            .bytes()
                            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
                })
            {
                return Err(CircuitPlanError(format!(
                    "package tensor {tensor:?} has no valid data SHA-256"
                )));
            }
        }
        Ok(index)
    }

    pub fn tensor_shape(&self, tensor: &str) -> Option<&[usize]> {
        self.tensors.get(tensor).map(|metadata| {
            metadata
                .logical_shape
                .as_deref()
                .unwrap_or(metadata.shape.as_slice())
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
pub struct TensorMetadata {
    pub dtype: String,
    pub shape: Vec<usize>,
    #[serde(default)]
    pub logical_shape: Option<Vec<usize>>,
    #[serde(default)]
    pub parameter_count: Option<usize>,
    #[serde(default)]
    pub byte_count: Option<usize>,
    #[serde(default)]
    pub data_offsets: Option<Vec<usize>>,
    #[serde(default)]
    pub source_file: Option<String>,
    #[serde(default)]
    pub data_sha256: Option<String>,
    #[serde(default)]
    pub layout: Option<String>,
}
