fn render_node_modal(frame: &mut Frame<'_>, app: &mut App, modal: &NodeModalState) {
    let area = centered_rect(frame.area(), 70, 78, 54, 20);
    frame.render_widget(Clear, area);
    let title = format!(
        " NODE {} · OCCURRENCE {} ",
        modal.source.source_id, modal.occurrence
    );
    let block = Block::default()
        .title(Span::styled(title, Style::default().fg(SIGNAL).bold()))
        .borders(Borders::ALL)
        .border_type(BorderType::Double)
        .border_style(Style::default().fg(SIGNAL));
    let inner = block.inner(area);
    frame.render_widget(block, area);
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(4),
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Min(4),
            Constraint::Length(2),
        ])
        .split(inner);
    frame.render_widget(
        Paragraph::new(vec![
            Line::from(vec![
                Span::styled("Source       ", Style::default().fg(META)),
                Span::styled(&modal.source.source_id, Style::default().fg(TEXT)),
            ]),
            Line::from(vec![
                Span::styled("Circuit      ", Style::default().fg(META)),
                Span::styled(&modal.source.operator_type, Style::default().fg(COOL)),
            ]),
            Line::from(vec![
                Span::styled("Implementation ", Style::default().fg(META)),
                Span::styled(
                    truncate(
                        &modal.source.implementation,
                        rows[0].width.saturating_sub(16) as usize,
                    ),
                    Style::default().fg(META),
                ),
            ]),
            Line::styled(
                format!(
                    "{} nodes · {} kernels · {} parameter refs",
                    modal.source.node_count,
                    modal.source.kernel_count,
                    modal.source.parameter_ref_count
                ),
                Style::default().fg(META),
            ),
        ]),
        rows[0],
    );
    let device = modal
        .device_labels
        .get(modal.device_index)
        .map(String::as_str)
        .unwrap_or("no compatible device");
    let policy = match modal.policy {
        NodePolicyKind::Independent => "Independent",
        NodePolicyKind::Clone => "Clone from another instance",
        NodePolicyKind::Share => "Share with another instance",
    };
    let policy_target = modal
        .policy_targets
        .get(modal.policy_target_index)
        .map(String::as_str)
        .unwrap_or("no other instance");
    let control_lines = [
        format!("Device       ‹ {device} ›"),
        format!("Enabled      [{}]", if modal.enabled { "x" } else { " " }),
        format!("State policy ‹ {policy} ›"),
        format!("State source ‹ {policy_target} ›"),
    ];
    for (index, line) in control_lines.into_iter().enumerate() {
        let area = rows[index + 1];
        frame.render_widget(
            Paragraph::new(line).style(modal_row_style(modal.focus_row == index)),
            area,
        );
        app.hit_map.insert(area, HitTarget::ModalRow(index));
    }
    render_node_properties(frame, app, modal, rows[6]);
    if let Some(error) = &modal.error {
        frame.render_widget(
            Paragraph::new(truncate(error, rows[6].width as usize))
                .style(Style::default().fg(FAULT)),
            Rect::new(
                rows[6].x,
                rows[6].bottom().saturating_sub(1),
                rows[6].width,
                1,
            ),
        );
    }
    let apply_style = modal_row_style(modal.focus_row == modal.apply_row());
    let cancel_style = modal_row_style(modal.focus_row == modal.cancel_row());
    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled("[ Apply ]", apply_style),
            Span::raw("    "),
            Span::styled("[ Cancel ]", cancel_style),
        ]))
        .alignment(Alignment::Center),
        rows[7],
    );
    let middle = rows[7].x + rows[7].width / 2;
    app.hit_map.insert(
        Rect::new(middle.saturating_sub(10), rows[7].y, 9, 1),
        HitTarget::ModalApply,
    );
    app.hit_map.insert(
        Rect::new(middle + 2, rows[7].y, 10, 1),
        HitTarget::ModalCancel,
    );
}

fn render_node_properties(
    frame: &mut Frame<'_>,
    app: &mut App,
    modal: &NodeModalState,
    area: Rect,
) {
    let block = Block::default()
        .title(" Instance properties ")
        .borders(Borders::TOP)
        .border_style(Style::default().fg(QUIET));
    let inner = block.inner(area);
    frame.render_widget(block, area);
    if modal.properties.is_empty() {
        frame.render_widget(
            Paragraph::new("No control schema declared by this source component.")
                .style(Style::default().fg(META)),
            inner,
        );
        return;
    }
    let detail_height = u16::from(inner.height >= 2);
    let list_area = Rect::new(
        inner.x,
        inner.y,
        inner.width,
        inner.height.saturating_sub(detail_height),
    );
    let visible = list_area.height as usize;
    let selected = modal.property_index().unwrap_or(0);
    let start = selected.saturating_sub(visible.saturating_sub(1));
    for (visual_row, (property_index, property)) in modal
        .properties
        .iter()
        .enumerate()
        .skip(start)
        .take(visible)
        .enumerate()
    {
        let row = Rect::new(
            list_area.x,
            list_area.y + visual_row as u16,
            list_area.width,
            1,
        );
        let focused = modal.focus_row == property_index + 4;
        let label_width = (inner.width / 3).clamp(12, 22);
        let label_area = Rect::new(row.x, row.y, label_width.min(row.width), 1);
        let value_area = Rect::new(
            label_area.right().saturating_add(1),
            row.y,
            row.right()
                .saturating_sub(label_area.right().saturating_add(1)),
            1,
        );
        let label = format!(
            "{}{}",
            if focused { "› " } else { "  " },
            property.schema.name
        );
        frame.render_widget(
            Paragraph::new(truncate(&label, label_area.width as usize)).style(if focused {
                Style::default().fg(SIGNAL).bold()
            } else {
                Style::default().fg(META)
            }),
            label_area,
        );
        if property.accepts_text() {
            let width = value_area.width.saturating_sub(1) as usize;
            let (start, line, cursor) = buffer_line(&property.buffer, width.max(1));
            frame.render_widget(
                Paragraph::new(line).style(if property.error.is_some() {
                    Style::default().fg(FAULT)
                } else if focused {
                    Style::default().fg(SIGNAL).underlined()
                } else {
                    Style::default().fg(TEXT)
                }),
                value_area,
            );
            if focused && property.buffer.cursor() >= start {
                frame.set_cursor_position(Position::new(value_area.x + cursor as u16, row.y));
            }
        } else {
            let value = property_display(property);
            frame.render_widget(
                Paragraph::new(truncate(&value, value_area.width as usize)).style(
                    if property.error.is_some() {
                        Style::default().fg(FAULT)
                    } else if focused {
                        Style::default().fg(SIGNAL).underlined()
                    } else if property.editable() {
                        Style::default().fg(TEXT)
                    } else {
                        Style::default().fg(META)
                    },
                ),
                value_area,
            );
        }
        app.hit_map
            .insert(row, HitTarget::ModalRow(property_index + 4));
    }
    if detail_height > 0 {
        let selected = modal
            .property_index()
            .and_then(|index| modal.properties.get(index));
        let detail = selected
            .and_then(|property| {
                property
                    .error
                    .as_deref()
                    .map(|error| (format!("! {error}"), FAULT))
                    .or_else(|| {
                        property
                            .schema
                            .description
                            .as_deref()
                            .map(|description| (description.to_string(), META))
                    })
            })
            .unwrap_or_else(|| {
                (
                    "Arrows change controls · text fields accept typing and paste".to_string(),
                    QUIET,
                )
            });
        frame.render_widget(
            Paragraph::new(truncate(&detail.0, inner.width as usize))
                .style(Style::default().fg(detail.1)),
            Rect::new(inner.x, inner.bottom().saturating_sub(1), inner.width, 1),
        );
    }
}

fn property_display(property: &super::app::NodePropertyDraft) -> String {
    let units = property
        .schema
        .units
        .as_deref()
        .map(|units| format!(" {units}"))
        .unwrap_or_default();
    let lifecycle = [
        property.schema.requires_state_reset.then_some("reset"),
        property.schema.requires_remount.then_some("remount"),
        property.schema.requires_recompile.then_some("recompile"),
    ]
    .into_iter()
    .flatten()
    .collect::<Vec<_>>()
    .join("+");
    let lifecycle = if lifecycle.is_empty() {
        String::new()
    } else {
        format!(" · {lifecycle}")
    };
    match &property.schema.kind {
        RuntimeEditorControlKind::Unsupported { declared_type } => {
            format!("unsupported {declared_type} · {}", property.schema.raw)
        }
        RuntimeEditorControlKind::ReadOnly => {
            format!("{}{} · read-only", property.display_value(), units)
        }
        _ if !property.editable() => {
            format!("{}{} · read-only", property.display_value(), units)
        }
        _ => format!("‹ {}{} ›{lifecycle}", property.display_value(), units),
    }
}

fn short_device_label(
    instance: &RuntimeEditorInstance,
    devices: Option<&[RuntimeAvailableDevice]>,
) -> String {
    let Some(device) = devices.and_then(|devices| {
        devices
            .iter()
            .find(|device| device.device_id == instance.device_id)
    }) else {
        return instance.device_id.clone();
    };
    let ordinal = device.physical_device_index.unwrap_or(0);
    match device.device_type.as_deref() {
        Some("cpu") => format!("CPU {ordinal}"),
        Some("discrete_gpu" | "integrated_gpu" | "virtual_gpu" | "other") => {
            format!("GPU {ordinal}")
        }
        _ => device
            .device_name
            .clone()
            .unwrap_or_else(|| instance.device_id.clone()),
    }
}

fn modal_row_style(focused: bool) -> Style {
    if focused {
        Style::default().fg(SIGNAL).bold().underlined()
    } else {
        Style::default().fg(TEXT)
    }
}
