fn render_compiler(frame: &mut Frame<'_>, app: &mut App, progress: &CompilerProgressState) {
    let area = centered_rect(frame.area(), 78, 66, 58, 15);
    frame.render_widget(Clear, area);
    let title = match progress.kind {
        super::compiler::CompilerJobKind::Discovery => " DISCOVERING MODEL ",
        super::compiler::CompilerJobKind::Compilation => " TRANSPILING MODEL ",
    };
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
            Constraint::Length(2),
            Constraint::Length(2),
            Constraint::Length(2),
            Constraint::Min(3),
            Constraint::Length(2),
        ])
        .split(inner);
    frame.render_widget(
        Paragraph::new(vec![
            Line::styled(
                truncate(
                    &progress.source_path.display().to_string(),
                    rows[0].width as usize,
                ),
                Style::default().fg(TEXT),
            ),
            Line::styled(&progress.stage, Style::default().fg(COOL).bold()),
        ]),
        rows[0],
    );
    if let (Some(current), Some(total)) = (progress.current, progress.total) {
        let ratio = if total == 0 {
            0.0
        } else {
            current.min(total) as f64 / total as f64
        };
        frame.render_widget(
            Gauge::default()
                .ratio(ratio)
                .label(format!("{current} / {total}"))
                .gauge_style(Style::default().fg(SIGNAL).bg(QUIET)),
            Rect::new(rows[1].x, rows[1].y, rows[1].width, 1),
        );
    } else {
        frame.render_widget(
            Paragraph::new("Progress is stage-based; no fabricated percentage")
                .style(Style::default().fg(META)),
            rows[1],
        );
    }
    let current = progress
        .current_item
        .as_deref()
        .map(|item| format!("Current circuit: {item}"))
        .unwrap_or_else(|| format!("Structured events: {}", progress.events.len()));
    frame.render_widget(
        Paragraph::new(current).style(Style::default().fg(META)),
        rows[2],
    );
    let diagnostics = if progress.diagnostics.is_empty() {
        vec![Line::styled(
            "Compiler diagnostics will appear here.",
            Style::default().fg(QUIET),
        )]
    } else {
        progress
            .diagnostics
            .iter()
            .skip(progress.diagnostic_scroll)
            .map(|line| Line::styled(line.clone(), Style::default().fg(FAULT)))
            .collect()
    };
    frame.render_widget(
        Paragraph::new(diagnostics)
            .block(
                Block::default()
                    .title(" Diagnostics ")
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(QUIET)),
            )
            .wrap(Wrap { trim: true }),
        rows[3],
    );
    let cancel_label = if progress.cancelling {
        "[ stopping safely… ]"
    } else {
        "[ Cancel ]"
    };
    frame.render_widget(
        Paragraph::new(cancel_label)
            .alignment(Alignment::Center)
            .style(Style::default().fg(if progress.cancelling { META } else { FAULT })),
        rows[4],
    );
    app.hit_map.insert(rows[4], HitTarget::CompilerCancel);
}

