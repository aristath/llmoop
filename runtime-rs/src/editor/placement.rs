pub fn placement_components_by_instance(
    validation: &RuntimeEditorValidation,
) -> BTreeMap<&str, &ComponentPlacement> {
    validation
        .placement
        .as_ref()
        .map(|placement| {
            placement
                .components
                .iter()
                .map(|component| (component.component_id.as_str(), component))
                .collect()
        })
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;

    #[test]
    fn classifies_compiled_packages_and_safetensors_sources() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "nerve-editor-path-{}-{unique}",
            std::process::id()
        ));
        let package = root.join("package");
        let source = root.join("source");
        fs::create_dir_all(&package).unwrap();
        fs::create_dir_all(&source).unwrap();
        fs::write(package.join(RUNTIME_PACKAGE_MANIFEST_FILE), b"{}").unwrap();
        fs::write(source.join("config.json"), b"{}").unwrap();
        fs::write(source.join("tokenizer.json"), b"{}").unwrap();
        fs::write(source.join("model.safetensors"), b"").unwrap();

        assert_eq!(
            classify_runtime_model_path(&package).unwrap(),
            RuntimeModelPathKind::CompiledPackage {
                manifest: package.join(RUNTIME_PACKAGE_MANIFEST_FILE)
            }
        );
        assert_eq!(
            classify_runtime_model_path(&source).unwrap(),
            RuntimeModelPathKind::SafetensorsSource {
                model_dir: source.clone()
            }
        );

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn configured_package_editor_preserves_instances_while_reordering() {
        let package = std::env::var("NERVE_TEST_PACKAGE_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("..")
                    .join("packages")
                    .join("model_7760f415")
            });
        if !package.join(RUNTIME_PACKAGE_MANIFEST_FILE).is_file() {
            return;
        }
        let mut editor = load_runtime_model_editor_without_hardware(&package).unwrap();
        let original_first = editor
            .instances()
            .iter()
            .find(|instance| instance.layer_index == Some(0))
            .unwrap()
            .instance_id
            .clone();

        editor.replace_layer_sequence(&[0, 1, 1, 2]).unwrap();

        let instances = editor
            .instances()
            .into_iter()
            .filter(|instance| instance.layer_index.is_some())
            .collect::<Vec<_>>();
        assert_eq!(editor.layer_sequence(), vec![0, 1, 1, 2]);
        assert_eq!(instances[0].instance_id, original_first);
        assert_eq!(instances[1].occurrence, 1);
        assert_eq!(instances[2].occurrence, 2);
        assert_ne!(instances[1].instance_id, instances[2].instance_id);
        assert!(editor.validation().valid);
    }

    #[test]
    fn generic_editor_preserves_explicit_system_component_placement() {
        let package = std::env::var("NERVE_TEST_PACKAGE_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("..")
                    .join("packages")
                    .join("model_7760f415")
            });
        if !package.join(RUNTIME_PACKAGE_MANIFEST_FILE).is_file() {
            return;
        }
        let initial = load_runtime_model_editor_without_hardware(&package).unwrap();
        let mut secondary = initial.available_devices()[0].clone();
        secondary.device_id = "gpu1".to_string();
        secondary.runtime_device_id = Some("gpu1".to_string());
        secondary.physical_device_id = Some("test:1".to_string());
        secondary.physical_device_index = Some(1);
        let mut editor = RuntimeModelEditor::load_with_available_devices(
            &package,
            vec![initial.available_devices()[0].clone(), secondary],
        )
        .unwrap();

        editor
            .set_instance_device("input_transducer", "gpu1")
            .unwrap();

        assert_eq!(
            editor
                .draft()
                .instances
                .iter()
                .find(|instance| instance.instance_id == "input_transducer")
                .unwrap()
                .device_id,
            "gpu1"
        );
        assert!(editor.validation().valid);
    }

    #[test]
    fn generic_control_schema_preserves_constraints_and_rejects_bad_values() {
        let raw = serde_json::json!({
            "id": "attention_window",
            "name": "Attention window",
            "description": "Local temporal span",
            "value_type": "integer",
            "current": 4096,
            "default": 2048,
            "min": 128,
            "max": 8192,
            "step": 128,
            "units": "tokens",
            "editable_at_runtime": true,
            "requires_state_reset": true,
            "scope": "instance"
        });
        let schema = runtime_editor_control_schema(0, &raw);
        assert_eq!(schema.id, "attention_window");
        assert_eq!(schema.kind, RuntimeEditorControlKind::Integer);
        assert_eq!(schema.current_value, Some(serde_json::json!(4096)));
        assert_eq!(schema.minimum, Some(128.0));
        assert_eq!(schema.maximum, Some(8192.0));
        assert!(schema.requires_state_reset);
        assert!(validate_runtime_editor_control_value(&schema, &serde_json::json!(1024)).is_ok());
        assert!(
            validate_runtime_editor_control_value(&schema, &serde_json::json!(64))
                .unwrap_err()
                .to_string()
                .contains("below minimum")
        );
        assert!(
            validate_runtime_editor_control_value(&schema, &serde_json::json!(1000))
                .unwrap_err()
                .to_string()
                .contains("does not align to step")
        );

        let unsupported = runtime_editor_control_schema(
            1,
            &serde_json::json!({"id":"control","type":"component_control"}),
        );
        assert!(matches!(
            unsupported.kind,
            RuntimeEditorControlKind::Unsupported { .. }
        ));
        assert!(!unsupported.editable_at_runtime);
    }
}
