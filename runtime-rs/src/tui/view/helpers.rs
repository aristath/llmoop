fn buffer_line(buffer: &TextBuffer, width: usize) -> (usize, Line<'static>, usize) {
    let chars = buffer.text().chars().collect::<Vec<_>>();
    let cursor = buffer.cursor().min(chars.len());
    let start = cursor.saturating_sub(width.saturating_sub(1));
    let end = (start + width).min(chars.len());
    let selection = buffer.selection();
    let mut spans = Vec::new();
    let mut run_start = start;
    let mut run_selected = selection.is_some_and(|(left, right)| start >= left && start < right);
    for index in start..end {
        let selected = selection.is_some_and(|(left, right)| index >= left && index < right);
        if selected != run_selected {
            spans.push(buffer_span(&chars[run_start..index], run_selected));
            run_start = index;
            run_selected = selected;
        }
    }
    spans.push(buffer_span(&chars[run_start..end], run_selected));
    (start, Line::from(spans), cursor.saturating_sub(start))
}

fn buffer_span(chars: &[char], selected: bool) -> Span<'static> {
    let text = chars.iter().collect::<String>();
    if selected {
        Span::styled(text, Style::default().fg(Color::Black).bg(SIGNAL))
    } else {
        Span::styled(text, Style::default().fg(TEXT))
    }
}

fn device_color_map(instances: &[RuntimeEditorInstance]) -> BTreeMap<String, Color> {
    let colors = [
        Color::Rgb(103, 194, 255),
        Color::Rgb(150, 220, 150),
        Color::Rgb(199, 151, 255),
        Color::Rgb(92, 214, 196),
        Color::Rgb(255, 150, 190),
    ];
    let mut map = BTreeMap::new();
    for instance in instances {
        let next = colors[map.len() % colors.len()];
        map.entry(instance.device_id.clone()).or_insert(next);
    }
    map
}

fn centered_rect(
    area: Rect,
    percent_x: u16,
    percent_y: u16,
    min_width: u16,
    min_height: u16,
) -> Rect {
    let width = ((area.width as u32 * percent_x as u32 / 100) as u16)
        .max(min_width.min(area.width))
        .min(area.width);
    let height = ((area.height as u32 * percent_y as u32 / 100) as u16)
        .max(min_height.min(area.height))
        .min(area.height);
    Rect::new(
        area.x + area.width.saturating_sub(width) / 2,
        area.y + area.height.saturating_sub(height) / 2,
        width,
        height,
    )
}

fn centered_line_area(area: Rect, height: u16) -> Rect {
    Rect::new(
        area.x,
        area.y + area.height.saturating_sub(height) / 2,
        area.width,
        height.min(area.height),
    )
}

fn truncate(value: &str, width: usize) -> String {
    if width == 0 {
        return String::new();
    }
    if value.width() <= width {
        return value.to_string();
    }
    if width == 1 {
        return "…".to_string();
    }
    let mut result = String::new();
    for character in value.chars() {
        let next = character.width().unwrap_or(0);
        if result.width() + next + 1 > width {
            break;
        }
        result.push(character);
    }
    result.push('…');
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    fn rendered_text(app: &mut App, width: u16, height: u16) -> String {
        let backend = TestBackend::new(width, height);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|frame| render(frame, app)).unwrap();
        let buffer = terminal.backend().buffer();
        (0..height)
            .flat_map(|y| (0..width).map(move |x| buffer[(x, y)].symbol()))
            .collect::<String>()
    }

    #[test]
    fn truncation_respects_terminal_cell_width() {
        assert_eq!(truncate("abcdef", 4), "abc…");
        assert_eq!(truncate("λx", 4), "λx");
    }

    #[test]
    fn text_buffer_view_keeps_cursor_visible() {
        let buffer = TextBuffer::new("[0,1,2,3]");
        let (start, line, cursor) = buffer_line(&buffer, 5);
        assert_eq!(start, 5);
        assert_eq!(cursor, 4);
        assert_eq!(line.to_string(), "2,3]");
    }

    #[test]
    fn loaded_workspace_renders_signal_chain_at_normal_and_small_sizes() {
        let Some(package) = std::env::var_os("NERVE_TEST_PACKAGE_DIR") else {
            return;
        };
        let editor = crate::editor::load_runtime_model_editor_without_hardware(package).unwrap();
        let mut app = App::new();
        app.install_editor(editor);
        for (width, height) in [(80, 24), (40, 12)] {
            let rendered = rendered_text(&mut app, width, height);
            assert!(rendered.contains("SIGNAL BOARD"));
            assert!(rendered.contains("PEDALBOARD"));
            assert!(rendered.contains("ZERO-BASED"));
        }
    }

    #[test]
    fn model_selector_and_pedal_modal_keep_actions_visible_in_small_terminals() {
        let mut app = App::new();
        let rendered = rendered_text(&mut app, 40, 12);
        assert!(rendered.contains("OPEN MODEL"));
        assert!(rendered.contains("Cancel"));

        let schema = crate::runtime_editor_control_schema(
            0,
            &serde_json::json!({
                "id": "window",
                "name": "Window",
                "description": "Local temporal span",
                "type": "integer",
                "current": 4,
                "min": 2,
                "max": 10,
                "step": 2,
                "editable_at_runtime": true,
                "scope": "instance"
            }),
        );
        let property = super::super::app::PedalPropertyDraft::new(schema, serde_json::json!(4));
        let mut modal = PedalModalState {
            instance_id: "layer_00".to_string(),
            source: crate::RuntimeEditorSourcePedal {
                source_id: "layer_00".to_string(),
                layer_index: Some(0),
                operator_type: "transformer".to_string(),
                runtime_role: crate::CircuitRuntimeRole::SignalProcessor,
                implementation: "compiled_circuit".to_string(),
                behavioral_role: "stream_transform".to_string(),
                input_shape: vec![64],
                output_shape: vec![64],
                state_ports: Vec::new(),
                controls: Vec::new(),
                control_schemas: Vec::new(),
                parameter_ref_count: 4,
                node_count: 8,
                kernel_count: 3,
            },
            occurrence: 1,
            device_ids: vec!["gpu0".to_string()],
            device_labels: vec!["gpu0 · fixture".to_string()],
            device_index: 0,
            original_device_id: "gpu0".to_string(),
            enabled: true,
            policy: PedalPolicyKind::Independent,
            policy_targets: Vec::new(),
            policy_target_index: 0,
            properties: vec![property],
            focus_row: 5,
            error: None,
        };
        modal.focus_row = modal.apply_row();
        app.overlay = Some(Overlay::Pedal(modal));
        let rendered = rendered_text(&mut app, 40, 12);
        assert!(rendered.contains("LAYER 0"));
        assert!(rendered.contains("Apply"));
        assert!(rendered.contains("Cancel"));
    }
}
