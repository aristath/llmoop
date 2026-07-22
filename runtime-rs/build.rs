use std::fs;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

const COMPILER_FINGERPRINT_SCHEMA: &str = "llmoop.package_compiler_sha256.v1";

fn directory_files(path: &Path, prefix: &str) -> Vec<(String, PathBuf)> {
    fs::read_dir(path)
        .unwrap_or_else(|error| panic!("failed to read compiler input directory {path:?}: {error}"))
        .map(|entry| entry.expect("failed to read compiler input entry").path())
        .filter(|path| path.is_file())
        .filter_map(|path| {
            let name = path.file_name()?.to_str()?;
            Some((format!("{prefix}/{name}"), path))
        })
        .collect()
}

fn main() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let repository_root = manifest_dir
        .parent()
        .expect("runtime crate must live inside the repository");
    println!(
        "cargo:rerun-if-changed={}",
        repository_root.join("llmoop").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        manifest_dir.join("shaders").display()
    );
    let mut inputs = directory_files(&repository_root.join("llmoop"), "llmoop")
        .into_iter()
        .filter(|(relative, _)| relative.ends_with(".py"))
        .chain(directory_files(
            &manifest_dir.join("shaders"),
            "runtime-rs/shaders",
        ))
        .collect::<Vec<_>>();
    inputs.sort_by(|left, right| left.0.cmp(&right.0));

    let mut digest = Sha256::new();
    for (relative, path) in inputs {
        println!("cargo:rerun-if-changed={}", path.display());
        let relative_bytes = relative.as_bytes();
        let source_bytes = fs::read(&path)
            .unwrap_or_else(|error| panic!("failed to read compiler input {path:?}: {error}"));
        digest.update((relative_bytes.len() as u64).to_le_bytes());
        digest.update(relative_bytes);
        digest.update((source_bytes.len() as u64).to_le_bytes());
        digest.update(source_bytes);
    }
    println!(
        "cargo:rustc-env=LLMOOP_PACKAGE_COMPILER_FINGERPRINT={COMPILER_FINGERPRINT_SCHEMA}:{:x}",
        digest.finalize()
    );
}
