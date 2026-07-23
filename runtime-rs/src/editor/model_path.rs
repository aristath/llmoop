#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeModelPathKind {
    CompiledPackage { manifest: PathBuf },
    SafetensorsSource { model_dir: PathBuf },
}

pub fn classify_runtime_model_path(
    path: impl AsRef<Path>,
) -> Result<RuntimeModelPathKind, RuntimeEditorError> {
    let path = path.as_ref();
    if path.is_file()
        && path.file_name().and_then(|name| name.to_str()) == Some(RUNTIME_PACKAGE_MANIFEST_FILE)
    {
        return Ok(RuntimeModelPathKind::CompiledPackage {
            manifest: path.to_path_buf(),
        });
    }
    if !path.is_dir() {
        return Err(RuntimeEditorError(format!(
            "model path does not exist or is not a directory: {}",
            path.display()
        )));
    }
    let manifest = path.join(RUNTIME_PACKAGE_MANIFEST_FILE);
    if manifest.is_file() {
        return Ok(RuntimeModelPathKind::CompiledPackage { manifest });
    }
    if path.join("config.json").is_file()
        && path.join("tokenizer.json").is_file()
        && path.read_dir()?.filter_map(Result::ok).any(|entry| {
            entry.path().extension().and_then(|value| value.to_str()) == Some("safetensors")
        })
    {
        return Ok(RuntimeModelPathKind::SafetensorsSource {
            model_dir: path.to_path_buf(),
        });
    }
    Err(RuntimeEditorError(format!(
        "{} is neither a NERVE package nor a discoverable Safetensors model",
        path.display()
    )))
}
