impl Default for App {
    fn default() -> Self {
        Self::new()
    }
}

fn move_text_cursor(buffer: &mut TextBuffer, motion: CursorMotion, selecting: bool) {
    match motion {
        CursorMotion::Left => buffer.move_left(selecting),
        CursorMotion::Right => buffer.move_right(selecting),
        CursorMotion::Home => buffer.move_home(selecting),
        CursorMotion::End => buffer.move_end(selecting),
    }
}

fn change_node_modal(modal: &mut NodeModalState, delta: i32) {
    match modal.focus_row {
        0 if !modal.device_ids.is_empty() => {
            modal.device_index = cycle_index(modal.device_index, modal.device_ids.len(), delta)
        }
        1 => modal.enabled = !modal.enabled,
        2 => {
            let choices = if modal.policy_targets.is_empty() {
                1
            } else {
                3
            };
            let current = match modal.policy {
                NodePolicyKind::Independent => 0,
                NodePolicyKind::Clone => 1,
                NodePolicyKind::Share => 2,
            };
            modal.policy = match cycle_index(current, choices, delta) {
                0 => NodePolicyKind::Independent,
                1 => NodePolicyKind::Clone,
                _ => NodePolicyKind::Share,
            };
        }
        3 if !modal.policy_targets.is_empty() => {
            modal.policy_target_index =
                cycle_index(modal.policy_target_index, modal.policy_targets.len(), delta);
        }
        _ => {
            if let Some(property) = focused_node_property_mut(modal) {
                property.change(delta);
            }
        }
    }
}

fn focused_node_property_mut(modal: &mut NodeModalState) -> Option<&mut NodePropertyDraft> {
    let index = modal.property_index()?;
    modal.properties.get_mut(index)
}

fn cycle_index(current: usize, len: usize, delta: i32) -> usize {
    if len == 0 {
        return 0;
    }
    (current as i32 + delta).rem_euclid(len as i32) as usize
}

fn control_value_text(value: &Value) -> String {
    value
        .as_str()
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| value.to_string())
}

fn browser_entries(directory: &Path) -> Vec<BrowserEntry> {
    let mut entries = Vec::new();
    if let Some(parent) = directory.parent() {
        entries.push(BrowserEntry {
            path: parent.to_path_buf(),
            label: "../".to_string(),
            is_directory: true,
        });
    }
    let Ok(read_dir) = fs::read_dir(directory) else {
        return entries;
    };
    let mut children = read_dir
        .filter_map(Result::ok)
        .filter_map(|entry| {
            let file_type = entry.file_type().ok()?;
            let is_directory = file_type.is_dir();
            let name = entry.file_name().to_string_lossy().into_owned();
            Some(BrowserEntry {
                path: entry.path(),
                label: if is_directory {
                    format!("{name}/")
                } else {
                    name
                },
                is_directory,
            })
        })
        .collect::<Vec<_>>();
    children.sort_by(|left, right| {
        right
            .is_directory
            .cmp(&left.is_directory)
            .then_with(|| left.label.to_lowercase().cmp(&right.label.to_lowercase()))
    });
    entries.extend(children);
    entries
}

fn expand_home(path: &str) -> PathBuf {
    if path == "~" {
        return env::var_os("HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(path));
    }
    if let Some(rest) = path.strip_prefix("~/")
        && let Some(home) = env::var_os("HOME")
    {
        return PathBuf::from(home).join(rest);
    }
    PathBuf::from(path)
}

fn format_layer_sequence(sequence: &[usize]) -> String {
    format!(
        "[{}]",
        sequence
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(",")
    )
}

fn compiler_stage_label(kind: &str) -> &str {
    match kind {
        "DiscoveryStarted" => "Discovering source structure",
        "SourceDiscovered" => "Source structure discovered",
        "ValidationStarted" => "Validating source artifacts",
        "ComponentTranspiled" => "Transpiling source components",
        "ComponentLoweringStarted" => "Lowering component circuits",
        "ArtifactWritingStarted" => "Writing package artifacts",
        "TensorPackagingStarted" => "Packaging tensors",
        "ShaderCompilationStarted" => "Compiling GPU circuits",
        "PackageValidationStarted" => "Validating compiled package",
        _ => kind,
    }
}

fn source_discovery_from_value(raw: Value) -> Option<SourceDiscovery> {
    let event = CompilerEvent {
        schema: "nerve.compiler_event.v1".to_string(),
        sequence: 0,
        kind: "SourceDiscovered".to_string(),
        payload: [("source".to_string(), raw)].into_iter().collect(),
    };
    SourceDiscovery::from_event(&event)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn visual_sequence_format_is_zero_based_and_has_no_ceremonial_prefix() {
        assert_eq!(format_layer_sequence(&[0, 1, 5, 5, 7]), "[0,1,5,5,7]");
    }

    #[test]
    fn property_draft_tracks_real_changes_and_schema_validation() {
        let schema = crate::runtime_editor_control_schema(
            0,
            &serde_json::json!({
                "id": "window",
                "name": "Window",
                "type": "integer",
                "current": 4,
                "min": 2,
                "max": 10,
                "step": 2,
                "editable_at_runtime": true,
                "scope": "INSTANCE"
            }),
        );
        let mut property = NodePropertyDraft::new(schema, serde_json::json!(4));
        assert!(property.editable());
        assert!(!property.changed());
        assert!(property.error.is_none());

        property.change(1);
        assert_eq!(property.value, serde_json::json!(6));
        assert!(property.changed());
        assert!(property.error.is_none());

        property.buffer.set("5");
        property.reparse_buffer();
        assert!(
            property
                .error
                .as_deref()
                .is_some_and(|error| error.contains("step"))
        );

        property.buffer.set("4");
        property.reparse_buffer();
        assert!(!property.changed());
        assert!(property.error.is_none());

        let missing = crate::runtime_editor_control_schema(
            1,
            &serde_json::json!({
                "id": "missing",
                "type": "number",
                "editable_at_runtime": true,
                "scope": "instance"
            }),
        );
        let missing = NodePropertyDraft::new(missing, Value::Null);
        assert!(
            missing
                .error
                .as_deref()
                .is_some_and(|error| error.contains("declared type"))
        );
    }

    #[test]
    fn help_and_mouse_capture_remain_global_without_losing_the_open_overlay() {
        let mut app = App::new();
        assert!(matches!(app.overlay, Some(Overlay::ModelSelector(_))));
        let original_mouse_capture = app.mouse_capture;

        app.dispatch(AppAction::ToggleHelp);
        assert!(matches!(app.overlay, Some(Overlay::Help)));
        app.dispatch(AppAction::ToggleMouseCapture);
        assert_eq!(app.mouse_capture, !original_mouse_capture);

        app.dispatch(AppAction::ToggleHelp);
        assert!(matches!(app.overlay, Some(Overlay::ModelSelector(_))));
    }

    #[test]
    fn model_selector_diagnostics_scroll_without_changing_browser_selection() {
        let mut app = App::new();
        let Some(Overlay::ModelSelector(selector)) = &mut app.overlay else {
            panic!("model selector did not open");
        };
        selector.focus = ModelSelectorFocus::Action;
        selector.diagnostics = vec!["first".to_string(), "second".to_string()];
        let selected_entry = selector.selected_entry;

        app.dispatch(AppAction::ModalNext);
        let Some(Overlay::ModelSelector(selector)) = &app.overlay else {
            panic!("model selector closed");
        };
        assert_eq!(selector.diagnostic_scroll, 1);
        assert_eq!(selector.selected_entry, selected_entry);
    }

    #[test]
    fn browser_lists_directories_before_files() {
        let root = env::temp_dir().join(format!("nerve-tui-browser-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("model")).unwrap();
        fs::write(root.join("readme"), "x").unwrap();
        let entries = browser_entries(&root);
        let model = entries
            .iter()
            .position(|entry| entry.label == "model/")
            .unwrap();
        let readme = entries
            .iter()
            .position(|entry| entry.label == "readme")
            .unwrap();
        assert!(model < readme);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn loaded_graph_visual_actions_keep_one_authoritative_numeric_sequence() {
        let Some(package) = env::var_os("NERVE_TEST_PACKAGE_DIR") else {
            return;
        };
        let editor = crate::editor::load_runtime_model_editor_without_hardware(package).unwrap();
        let mut app = App::new();
        app.install_editor(editor);
        assert!(app.overlay.is_none());
        let original = app.last_valid_sequence.clone();
        let original_selected = app.selected_instance_id.clone();

        app.dispatch(AppAction::DuplicateSelected);
        assert_eq!(app.last_valid_sequence.len(), original.len() + 1);
        assert_eq!(app.last_valid_sequence[0..2], [original[0], original[0]]);
        assert!(app.sequence.text().starts_with("[0,0,"));
        assert_ne!(app.selected_instance_id, original_selected);

        app.dispatch(AppAction::RemoveSelected);
        assert_eq!(app.last_valid_sequence, original);
        assert_eq!(
            parse_layer_sequence(
                app.sequence.text(),
                &app.editor
                    .as_ref()
                    .unwrap()
                    .source_components()
                    .iter()
                    .filter_map(|component| component.layer_index)
                    .collect()
            )
            .unwrap(),
            original
        );
    }
}
