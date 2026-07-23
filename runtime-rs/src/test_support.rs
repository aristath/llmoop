use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use crate::stream_circuit::ResolvedLoweredPedalboard;

pub(crate) fn compiled_artifact_dir(env_var: &str, root_name: &str, marker_file: &str) -> PathBuf {
    if let Ok(path) = std::env::var(env_var) {
        let path = PathBuf::from(path);
        assert!(
            path.join(marker_file).is_file(),
            "{env_var}={} does not contain {marker_file}",
            path.display()
        );
        return path;
    }

    let path = repository_root()
        .join(root_name)
        .join(fixture_artifact_id());
    assert!(
        path.join(marker_file).is_file(),
        "set {env_var}; the structurally selected test fixture {} has no {root_name}/{marker_file}",
        fixture_artifact_id()
    );
    path
}

fn fixture_artifact_id() -> &'static str {
    static FIXTURE_ARTIFACT_ID: OnceLock<String> = OnceLock::new();
    FIXTURE_ARTIFACT_ID.get_or_init(discover_fixture_artifact_id)
}

fn discover_fixture_artifact_id() -> String {
    let root = repository_root().join("lowered");
    let mut candidates = std::fs::read_dir(&root)
        .unwrap_or_else(|_| {
            panic!(
                "set NERVE_TEST_LOWERED_DIR or compile the structural test model into {}",
                root.display()
            )
        })
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| is_structural_test_fixture(path))
        .collect::<Vec<_>>();
    candidates.sort();

    match candidates.as_slice() {
        [path] => path
            .file_name()
            .and_then(|name| name.to_str())
            .expect("compiled fixture directory must have a UTF-8 name")
            .to_string(),
        [] => panic!(
            "set NERVE_TEST_LOWERED_DIR or compile exactly one compatible structural test model"
        ),
        _ => panic!(
            "multiple compatible structural test models were found ({}); set NERVE_TEST_LOWERED_DIR, NERVE_TEST_TRANSPILED_DIR, and NERVE_TEST_PACKAGE_DIR explicitly",
            candidates
                .iter()
                .map(|path| path.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

fn is_structural_test_fixture(path: &Path) -> bool {
    let index_path = path.join("pedalboard.circuits.json");
    let Ok(graph) = ResolvedLoweredPedalboard::from_index_file(index_path) else {
        return false;
    };
    let operator_types = graph
        .circuits
        .iter()
        .filter(|artifact| artifact.circuit.runtime_role.is_signal_processor())
        .map(|artifact| artifact.pedal.operator_type.as_str())
        .collect::<Vec<_>>();

    graph
        .index
        .dimensions
        .get("hidden_size")
        .and_then(|value| value.as_u64())
        == Some(1_024)
        && operator_types
            == [
                "conv",
                "conv",
                "full_attention",
                "conv",
                "full_attention",
                "conv",
                "full_attention",
                "conv",
                "full_attention",
                "conv",
                "full_attention",
                "conv",
                "full_attention",
                "conv",
            ]
}

fn repository_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..")
}
