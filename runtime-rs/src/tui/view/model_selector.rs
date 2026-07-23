fn render_model_selector(frame: &mut Frame<'_>, app: &mut App, selector: &ModelSelectorState) {
    let area = centered_rect(frame.area(), 86, 88, 64, 18);
    frame.render_widget(Clear, area);
    let block = Block::default()
        .title(Span::styled(
            " OPEN MODEL ",
            Style::default().fg(SIGNAL).bold(),
        ))
        .borders(Borders::ALL)
        .border_type(BorderType::Double)
        .border_style(Style::default().fg(SIGNAL));
    let inner = block.inner(area);
    frame.render_widget(block, area);
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(5),
            Constraint::Length(6),
            Constraint::Length(2),
        ])
        .split(inner);
    render_path_field(frame, app, selector, rows[0]);
    render_browser(frame, app, selector, rows[1]);
    render_detection(frame, selector, rows[2]);
    render_model_actions(frame, app, selector, rows[3]);
}

fn render_path_field(
    frame: &mut Frame<'_>,
    app: &mut App,
    selector: &ModelSelectorState,
    area: Rect,
) {
    let focused = selector.focus == ModelSelectorFocus::Path;
    let block = Block::default()
        .title(" Source path ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(if focused { SIGNAL } else { QUIET }));
    let inner = block.inner(area);
    frame.render_widget(block, area);
    app.hit_map.insert(area, HitTarget::ModelPath);
    let width = inner.width.saturating_sub(1) as usize;
    let (_, line, cursor) = buffer_line(&selector.path, width.max(1));
    frame.render_widget(Paragraph::new(line), inner);
    if focused {
        frame.set_cursor_position(Position::new(inner.x + cursor as u16, inner.y));
    }
}

fn render_browser(frame: &mut Frame<'_>, app: &mut App, selector: &ModelSelectorState, area: Rect) {
    let columns = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(58), Constraint::Percentage(42)])
        .split(area);
    let focused = selector.focus == ModelSelectorFocus::Browser;
    let list_block = Block::default()
        .title(Span::styled(
            format!(" {} ", selector.browser_directory.display()),
            Style::default().fg(META),
        ))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(if focused { SIGNAL } else { QUIET }));
    let list_inner = list_block.inner(columns[0]);
    frame.render_widget(list_block, columns[0]);
    let visible_height = list_inner.height as usize;
    let start = selector
        .selected_entry
        .saturating_sub(visible_height.saturating_sub(1));
    let entries = selector
        .entries
        .iter()
        .enumerate()
        .skip(start)
        .take(visible_height)
        .map(|(index, entry)| browser_item(index, entry, selector.selected_entry, focused))
        .collect::<Vec<_>>();
    frame.render_widget(List::new(entries), list_inner);
    for (row, index) in (start..selector.entries.len())
        .take(visible_height)
        .enumerate()
    {
        app.hit_map.insert(
            Rect::new(list_inner.x, list_inner.y + row as u16, list_inner.width, 1),
            HitTarget::BrowserEntry(index),
        );
    }
    let info = vec![
        Line::styled("Compiler boundary", Style::default().fg(SIGNAL).bold()),
        Line::styled(
            "Architecture discovery stays outside the TUI.",
            Style::default().fg(META),
        ),
        Line::raw(""),
        Line::styled(
            "Directories first · Enter opens · paste works",
            Style::default().fg(META),
        ),
    ];
    frame.render_widget(
        Paragraph::new(info)
            .block(
                Block::default()
                    .title(" Selection ")
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(QUIET)),
            )
            .wrap(Wrap { trim: true }),
        columns[1],
    );
}

fn browser_item(
    index: usize,
    entry: &BrowserEntry,
    selected: usize,
    focused: bool,
) -> ListItem<'static> {
    let marker = if index == selected { "› " } else { "  " };
    let style = if index == selected {
        Style::default()
            .fg(if focused { SIGNAL } else { TEXT })
            .add_modifier(Modifier::BOLD)
    } else if entry.is_directory {
        Style::default().fg(COOL)
    } else {
        Style::default().fg(META)
    };
    ListItem::new(format!("{marker}{}", entry.label)).style(style)
}

fn render_detection(frame: &mut Frame<'_>, selector: &ModelSelectorState, area: Rect) {
    let mut lines = Vec::new();
    match selector.detected.as_ref() {
        Some(Ok(crate::RuntimeModelPathKind::CompiledPackage { manifest })) => {
            lines.push(Line::from(vec![
                Span::styled("Detected  ", Style::default().fg(META)),
                Span::styled("compiled nerve package", Style::default().fg(TEXT).bold()),
            ]));
            lines.push(Line::styled(
                truncate(
                    &manifest.display().to_string(),
                    area.width.saturating_sub(4) as usize,
                ),
                Style::default().fg(META),
            ));
        }
        Some(Ok(crate::RuntimeModelPathKind::SafetensorsSource { .. })) => {
            lines.push(Line::from(vec![
                Span::styled("Detected  ", Style::default().fg(META)),
                Span::styled("Safetensors source", Style::default().fg(TEXT).bold()),
            ]));
            if let Some(discovery) = &selector.discovery {
                lines.push(Line::styled(
                    format!(
                        "{} · {} weight file(s) · tokenizer {} · chat template {}",
                        discovery.model_type,
                        discovery.weight_files.len(),
                        if discovery.tokenizer_files.is_empty() {
                            "missing"
                        } else {
                            "ready"
                        },
                        if discovery.has_chat_template {
                            "ready"
                        } else {
                            "absent"
                        }
                    ),
                    Style::default().fg(META),
                ));
                if !discovery.architecture.is_empty() {
                    lines.push(Line::styled(
                        discovery.architecture.join(", "),
                        Style::default().fg(COOL),
                    ));
                }
            } else {
                lines.push(Line::styled(
                    "Run compiler discovery before transpilation.",
                    Style::default().fg(META),
                ));
            }
        }
        Some(Err(error)) => lines.push(Line::styled(error.clone(), Style::default().fg(FAULT))),
        None => lines.push(Line::styled("No path selected", Style::default().fg(META))),
    }
    for diagnostic in selector.diagnostics.iter().skip(selector.diagnostic_scroll) {
        lines.push(Line::styled(
            format!("! {diagnostic}"),
            Style::default().fg(FAULT),
        ));
    }
    frame.render_widget(
        Paragraph::new(lines)
            .block(
                Block::default()
                    .title(" Discovery ")
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(QUIET)),
            )
            .wrap(Wrap { trim: true }),
        area,
    );
}

fn render_model_actions(
    frame: &mut Frame<'_>,
    app: &mut App,
    selector: &ModelSelectorState,
    area: Rect,
) {
    let active = !matches!(selector.current_action_label(), "Unavailable");
    let action_style = if !active {
        Style::default().fg(QUIET)
    } else if selector.focus == ModelSelectorFocus::Action {
        Style::default().fg(SIGNAL).bold().underlined()
    } else {
        Style::default().fg(COOL)
    };
    let label = format!("[ {} ]", selector.current_action_label());
    let cancel = "[ Cancel ]";
    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(label.clone(), action_style),
            Span::raw("   "),
            Span::styled(cancel, Style::default().fg(META)),
        ]))
        .alignment(Alignment::Center),
        area,
    );
    let width = label.width() as u16;
    let x = area.x + area.width.saturating_sub(width + cancel.width() as u16 + 3) / 2;
    app.hit_map
        .insert(Rect::new(x, area.y, width, 1), HitTarget::ModelAction);
    app.hit_map.insert(
        Rect::new(x + width + 3, area.y, cancel.width() as u16, 1),
        HitTarget::ModalCancel,
    );
}

