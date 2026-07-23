fn render_help(frame: &mut Frame<'_>) {
    let area = centered_rect(frame.area(), 66, 72, 52, 18);
    frame.render_widget(Clear, area);
    let lines = vec![
        Line::styled("PEDALBOARD", Style::default().fg(SIGNAL).bold()),
        Line::raw("←/→ or h/l   select pedal"),
        Line::raw("Enter         edit selected instance"),
        Line::raw("Ctrl+D        duplicate selected instance"),
        Line::raw("Delete        remove selected instance"),
        Line::raw("Alt+←/→       move selected instance"),
        Line::raw(""),
        Line::styled("WORKSPACE", Style::default().fg(SIGNAL).bold()),
        Line::raw("Tab           change focus region"),
        Line::raw("Ctrl+O        open/replace model"),
        Line::raw("Ctrl+M        enable/disable mouse capture"),
        Line::raw("Ctrl+R        refresh runtime devices"),
        Line::raw("F1            toggle this help"),
        Line::raw("q / Ctrl+C    quit when not editing text"),
        Line::raw(""),
        Line::styled("Esc closes this help.", Style::default().fg(META)),
    ];
    frame.render_widget(
        Paragraph::new(lines)
            .block(
                Block::default()
                    .title(" HELP ")
                    .borders(Borders::ALL)
                    .border_type(BorderType::Double)
                    .border_style(Style::default().fg(SIGNAL)),
            )
            .wrap(Wrap { trim: false }),
        area,
    );
}

