use std::fs;
use std::io;
use std::path::Path;

use serde::{Deserialize, Serialize};

pub const VULKAN_SPIRV_BACKEND_ID: &str = "vulkan_spirv";
pub const DEFAULT_SPIRV_ENTRY_POINT: &str = "main";
pub const DEFAULT_COMPUTE_LOCAL_SIZE_X: u32 = 64;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpirvPedalProgram {
    pub pedal_id: String,
    pub operator_type: String,
    pub entry_point: String,
    pub specialization: Option<String>,
    pub local_size_x: u32,
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
            local_size_x: DEFAULT_COMPUTE_LOCAL_SIZE_X,
            words,
        }
    }

    pub fn with_local_size_x(mut self, local_size_x: u32) -> Self {
        self.local_size_x = local_size_x;
        self
    }

    pub fn from_spirv_file(
        pedal_id: impl Into<String>,
        operator_type: impl Into<String>,
        path: impl AsRef<Path>,
    ) -> io::Result<Self> {
        Ok(Self::new(pedal_id, operator_type, read_spirv_words(path)?))
    }

    pub fn write_spirv_file(&self, path: impl AsRef<Path>) -> io::Result<()> {
        write_spirv_words(path, &self.words)
    }
}

pub fn read_spirv_words(path: impl AsRef<Path>) -> io::Result<Vec<u32>> {
    let bytes = fs::read(path)?;
    if bytes.len() % 4 != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "SPIR-V byte length must be divisible by 4",
        ));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

pub fn write_spirv_words(path: impl AsRef<Path>, words: &[u32]) -> io::Result<()> {
    let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    fs::write(path, bytes)
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VulkanBackendDescriptor {
    pub backend_id: String,
    pub device_id: String,
    pub queue_family: Option<u32>,
    pub programs: Vec<SpirvPedalProgram>,
}

impl VulkanBackendDescriptor {
    pub fn new(device_id: impl Into<String>, programs: Vec<SpirvPedalProgram>) -> Self {
        Self {
            backend_id: VULKAN_SPIRV_BACKEND_ID.to_string(),
            device_id: device_id.into(),
            queue_family: None,
            programs,
        }
    }

    pub fn empty(device_id: impl Into<String>) -> Self {
        Self::new(device_id, Vec::new())
    }

    pub fn with_program(mut self, program: SpirvPedalProgram) -> Self {
        self.programs.push(program);
        self
    }

    pub fn from_json_file(path: impl AsRef<Path>) -> io::Result<Self> {
        let bytes = fs::read(path)?;
        serde_json::from_slice(&bytes)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
    }

    pub fn write_json_file(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let bytes = serde_json::to_vec_pretty(self)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        fs::write(path, bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spirv_words_round_trip_through_file() {
        let words = vec![0x0723_0203, 0x0001_0000, 0x0008_000b, 42];
        let path = std::env::temp_dir().join(format!(
            "llmoop-spirv-round-trip-{}.spv",
            std::process::id()
        ));

        write_spirv_words(&path, &words).unwrap();
        let read = read_spirv_words(&path).unwrap();
        let program = SpirvPedalProgram::from_spirv_file("pedal_0", "test_kernel", &path).unwrap();
        program.write_spirv_file(&path).unwrap();

        assert_eq!(read, words);
        assert_eq!(program.pedal_id, "pedal_0");
        assert_eq!(program.operator_type, "test_kernel");
        assert_eq!(program.entry_point, DEFAULT_SPIRV_ENTRY_POINT);
        assert_eq!(program.local_size_x, DEFAULT_COMPUTE_LOCAL_SIZE_X);
        assert_eq!(read_spirv_words(&path).unwrap(), words);

        let _ = fs::remove_file(path);
    }

    #[test]
    fn rejects_spirv_files_with_partial_words() {
        let path =
            std::env::temp_dir().join(format!("llmoop-invalid-spirv-{}.spv", std::process::id()));
        fs::write(&path, [1_u8, 2, 3]).unwrap();

        let error = read_spirv_words(&path).unwrap_err();

        assert_eq!(error.kind(), io::ErrorKind::InvalidData);

        let _ = fs::remove_file(path);
    }

    #[test]
    fn backend_descriptor_round_trips_as_json() {
        let program = SpirvPedalProgram::new("pedal_a", "u32_add_one", vec![0x0723_0203, 1, 2, 3])
            .with_local_size_x(128);
        let descriptor = VulkanBackendDescriptor::empty("device_0").with_program(program);
        let path = std::env::temp_dir().join(format!(
            "llmoop-vulkan-descriptor-{}.json",
            std::process::id()
        ));

        descriptor.write_json_file(&path).unwrap();
        let read = VulkanBackendDescriptor::from_json_file(&path).unwrap();

        assert_eq!(read.backend_id, VULKAN_SPIRV_BACKEND_ID);
        assert_eq!(read.device_id, "device_0");
        assert_eq!(read.programs.len(), 1);
        assert_eq!(read.programs[0].pedal_id, "pedal_a");
        assert_eq!(read.programs[0].local_size_x, 128);

        let _ = fs::remove_file(path);
    }
}
