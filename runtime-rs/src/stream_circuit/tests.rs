fn read_json<T: for<'de> Deserialize<'de>>(
    path: impl AsRef<Path>,
) -> Result<T, CircuitArtifactError> {
    let bytes = fs::read(path)?;
    Ok(serde_json::from_slice(&bytes)?)
}

fn resolve_artifact_path(artifact_root: &Path, path: &str) -> PathBuf {
    let path = Path::new(path);
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        artifact_root.join(path)
    }
}

fn product(shape: &[usize]) -> Option<usize> {
    shape
        .iter()
        .try_fold(1usize, |total, value| total.checked_mul(*value))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::compiled_artifact_dir;

    fn fixture_model_index_path() -> PathBuf {
        compiled_artifact_dir(
            "NERVE_TEST_LOWERED_DIR",
            "lowered",
            "execution_graph.circuits.json",
        )
        .join("execution_graph.circuits.json")
    }

    include!("tests/circuit_contracts.rs");
    include!("tests/placement_routes.rs");
    include!("tests/runtime_reports.rs");
    include!("tests/runtime_graph.rs");
}
