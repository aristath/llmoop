use std::fs;
use std::io;
use std::path::Path;

pub const DEFAULT_SPIRV_ENTRY_POINT: &str = "main";
pub const DEFAULT_COMPUTE_LOCAL_SIZE_X: u32 = 64;

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

#[cfg(test)]
pub(crate) fn write_spirv_words(path: impl AsRef<Path>, words: &[u32]) -> io::Result<()> {
    let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    fs::write(path, bytes)
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
}
