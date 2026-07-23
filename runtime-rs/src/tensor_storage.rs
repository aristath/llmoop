use std::error::Error;
use std::fmt::{Display, Formatter};
use std::fs;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use crate::stream_plan::TensorIndex;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorStorage {
    pub tensor: String,
    pub source_file: PathBuf,
    pub data_start: usize,
    pub data_end: usize,
    pub byte_count: usize,
    pub expected_sha256: Option<String>,
}

impl TensorStorage {
    pub fn from_index(
        tensor_index: &TensorIndex,
        tensor: &str,
    ) -> Result<Self, TensorStorageError> {
        let metadata = tensor_index.tensors.get(tensor).ok_or_else(|| {
            TensorStorageError(format!(
                "tensor index has no metadata for tensor {tensor:?}"
            ))
        })?;
        let source_file = metadata.source_file.as_ref().ok_or_else(|| {
            TensorStorageError(format!("tensor metadata for {tensor:?} has no source_file"))
        })?;
        let offsets = metadata.data_offsets.as_ref().ok_or_else(|| {
            TensorStorageError(format!(
                "tensor metadata for {tensor:?} has no data_offsets"
            ))
        })?;
        let [data_start, data_end] = offsets.as_slice() else {
            return Err(TensorStorageError(format!(
                "tensor metadata for {tensor:?} has invalid data_offsets {offsets:?}"
            )));
        };
        let byte_count = data_end.checked_sub(*data_start).ok_or_else(|| {
            TensorStorageError(format!(
                "tensor metadata for {tensor:?} has reversed data_offsets {offsets:?}"
            ))
        })?;
        if metadata.byte_count != Some(byte_count) {
            return Err(TensorStorageError(format!(
                "tensor {tensor:?} metadata byte_count {:?} does not match data_offsets byte count {byte_count}",
                metadata.byte_count
            )));
        }

        Ok(Self {
            tensor: tensor.to_string(),
            source_file: PathBuf::from(source_file),
            data_start: *data_start,
            data_end: *data_end,
            byte_count,
            expected_sha256: metadata.data_sha256.clone(),
        })
    }

    pub fn read_all(&self) -> Result<Vec<u8>, TensorStorageError> {
        self.read_partitions(&[TensorStorageRange {
            byte_offset: 0,
            byte_count: self.byte_count,
        }])
        .map(|mut partitions| partitions.remove(0))
    }

    pub fn read_partitions(
        &self,
        partitions: &[TensorStorageRange],
    ) -> Result<Vec<Vec<u8>>, TensorStorageError> {
        validate_partition(self, partitions)?;
        let data_base = safetensors_data_start(&self.source_file)?;
        let absolute_start = data_base
            .checked_add(u64::try_from(self.data_start).map_err(|_| {
                TensorStorageError(format!(
                    "tensor {:?} data_start {} cannot fit in u64",
                    self.tensor, self.data_start
                ))
            })?)
            .ok_or_else(|| {
                TensorStorageError(format!(
                    "tensor {:?} absolute data offset overflowed",
                    self.tensor
                ))
            })?;
        let mut file = fs::File::open(&self.source_file).map_err(|error| {
            TensorStorageError(format!(
                "failed to open safetensors source {:?}: {error}",
                self.source_file
            ))
        })?;
        file.seek(SeekFrom::Start(absolute_start))
            .map_err(|error| {
                TensorStorageError(format!(
                    "failed to seek safetensors source {:?} to tensor {:?}: {error}",
                    self.source_file, self.tensor
                ))
            })?;

        let mut digest = Sha256::new();
        let mut bytes = Vec::with_capacity(partitions.len());
        for partition in partitions {
            let mut partition_bytes = Vec::new();
            partition_bytes
                .try_reserve_exact(partition.byte_count)
                .map_err(|error| {
                    TensorStorageError(format!(
                        "failed to reserve {} bytes for tensor {:?}: {error}",
                        partition.byte_count, self.tensor
                    ))
                })?;
            partition_bytes.resize(partition.byte_count, 0);
            file.read_exact(&mut partition_bytes).map_err(|error| {
                TensorStorageError(format!(
                    "failed to read tensor {:?} from safetensors source {:?}: {error}",
                    self.tensor, self.source_file
                ))
            })?;
            digest.update(&partition_bytes);
            bytes.push(partition_bytes);
        }
        validate_tensor_data_sha256(
            &self.tensor,
            self.expected_sha256.as_deref(),
            &digest.finalize(),
        )?;
        Ok(bytes)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TensorStorageRange {
    pub byte_offset: usize,
    pub byte_count: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorStorageError(pub String);

impl Display for TensorStorageError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for TensorStorageError {}

fn validate_partition(
    storage: &TensorStorage,
    partitions: &[TensorStorageRange],
) -> Result<(), TensorStorageError> {
    if partitions.is_empty() {
        return Err(TensorStorageError(format!(
            "tensor {:?} storage partition must not be empty",
            storage.tensor
        )));
    }
    let mut next_offset = 0usize;
    for (index, partition) in partitions.iter().enumerate() {
        if partition.byte_offset != next_offset {
            return Err(TensorStorageError(format!(
                "tensor {:?} storage partition {index} starts at {} instead of {next_offset}",
                storage.tensor, partition.byte_offset
            )));
        }
        next_offset = next_offset
            .checked_add(partition.byte_count)
            .ok_or_else(|| {
                TensorStorageError(format!(
                    "tensor {:?} storage partition byte count overflowed",
                    storage.tensor
                ))
            })?;
    }
    if next_offset != storage.byte_count {
        return Err(TensorStorageError(format!(
            "tensor {:?} storage partition covers {next_offset} of {} bytes",
            storage.tensor, storage.byte_count
        )));
    }
    Ok(())
}

fn validate_tensor_data_sha256(
    tensor: &str,
    expected: Option<&str>,
    actual: &[u8],
) -> Result<(), TensorStorageError> {
    let Some(expected) = expected else {
        return Ok(());
    };
    let actual = actual
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    if actual != expected {
        return Err(TensorStorageError(format!(
            "tensor {tensor:?} data SHA-256 does not match its compiled package contract"
        )));
    }
    Ok(())
}

fn safetensors_data_start(path: &Path) -> Result<u64, TensorStorageError> {
    let mut file = fs::File::open(path).map_err(|error| {
        TensorStorageError(format!("failed to open safetensors file {path:?}: {error}"))
    })?;
    let mut header_len_bytes = [0u8; 8];
    file.read_exact(&mut header_len_bytes).map_err(|error| {
        TensorStorageError(format!(
            "failed to read safetensors header length from {path:?}: {error}"
        ))
    })?;
    let header_len = u64::from_le_bytes(header_len_bytes);
    8u64.checked_add(header_len).ok_or_else(|| {
        TensorStorageError(format!("safetensors data start overflowed for {path:?}"))
    })
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;
    use crate::stream_plan::TensorMetadata;

    #[test]
    fn reads_and_verifies_a_tensor_as_storage_partitions() {
        let fixture = Fixture::new(b"abcdefgh");
        let storage = TensorStorage::from_index(&fixture.index, "weight").unwrap();

        let partitions = storage
            .read_partitions(&[
                TensorStorageRange {
                    byte_offset: 0,
                    byte_count: 3,
                },
                TensorStorageRange {
                    byte_offset: 3,
                    byte_count: 5,
                },
            ])
            .unwrap();

        assert_eq!(partitions, vec![b"abc".to_vec(), b"defgh".to_vec()]);
    }

    #[test]
    fn rejects_partitions_that_do_not_cover_the_tensor_exactly() {
        let fixture = Fixture::new(b"abcdefgh");
        let storage = TensorStorage::from_index(&fixture.index, "weight").unwrap();

        let error = storage
            .read_partitions(&[
                TensorStorageRange {
                    byte_offset: 0,
                    byte_count: 3,
                },
                TensorStorageRange {
                    byte_offset: 4,
                    byte_count: 4,
                },
            ])
            .unwrap_err();

        assert!(error.to_string().contains("starts at 4 instead of 3"));
    }

    #[test]
    fn rejects_same_length_tensor_corruption_after_partitioned_read() {
        let mut fixture = Fixture::new(b"abcdefgh");
        let metadata = fixture.index.tensors.get_mut("weight").unwrap();
        metadata.data_sha256 = Some(
            Sha256::digest(b"ABCDEFGH")
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect(),
        );
        let storage = TensorStorage::from_index(&fixture.index, "weight").unwrap();

        let error = storage.read_all().unwrap_err();

        assert!(
            error
                .to_string()
                .contains("does not match its compiled package contract")
        );
    }

    struct Fixture {
        root: PathBuf,
        index: TensorIndex,
    }

    impl Fixture {
        fn new(tensor_bytes: &[u8]) -> Self {
            let unique = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let root = std::env::temp_dir().join(format!(
                "nerve-tensor-storage-{}-{unique}",
                std::process::id()
            ));
            fs::create_dir_all(&root).unwrap();
            let source = root.join("weights.safetensors");
            let header = b"{}";
            let mut payload = Vec::new();
            payload.extend_from_slice(&(header.len() as u64).to_le_bytes());
            payload.extend_from_slice(header);
            payload.extend_from_slice(b"prefix");
            payload.extend_from_slice(tensor_bytes);
            payload.extend_from_slice(b"suffix");
            fs::write(&source, payload).unwrap();
            let digest = Sha256::digest(tensor_bytes)
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect();
            let index = TensorIndex {
                schema: "nerve.tensor_index.v1".to_string(),
                tensors: BTreeMap::from([(
                    "weight".to_string(),
                    TensorMetadata {
                        dtype: "BF16".to_string(),
                        shape: vec![2, 2],
                        logical_shape: None,
                        parameter_count: Some(4),
                        byte_count: Some(tensor_bytes.len()),
                        data_offsets: Some(vec![6, 6 + tensor_bytes.len()]),
                        source_file: Some(source.to_string_lossy().into_owned()),
                        data_sha256: Some(digest),
                        layout: Some("row_major".to_string()),
                    },
                )]),
            };
            Self { root, index }
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.root);
        }
    }
}
