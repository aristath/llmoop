use std::collections::BTreeMap;

use ratatui::Frame;
use ratatui::layout::{Alignment, Constraint, Direction, Layout, Position, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Borders, Clear, Gauge, List, ListItem, Paragraph, Wrap};
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

use crate::{RuntimeAvailableDevice, RuntimeEditorControlKind, RuntimeEditorInstance};

use super::app::{
    App, BrowserEntry, CompilerProgressState, FocusRegion, HitTarget, ModelSelectorFocus,
    ModelSelectorState, Overlay, PedalModalState, PedalPolicyKind,
};
use super::sequence::TextBuffer;

const SIGNAL: Color = Color::Rgb(240, 176, 64);
const META: Color = Color::Rgb(128, 137, 151);
const TEXT: Color = Color::Rgb(221, 225, 230);
const FAULT: Color = Color::Rgb(244, 114, 94);
const COOL: Color = Color::Rgb(103, 194, 255);
const QUIET: Color = Color::Rgb(76, 83, 96);

pub(crate) fn render(frame: &mut Frame<'_>, app: &mut App) {
    app.hit_map.clear();
    let area = frame.area();
    frame.render_widget(Block::default().style(Style::default()), area);
    render_workspace(frame, app, area);
    if let Some(overlay) = app.overlay.clone() {
        match overlay {
            Overlay::ModelSelector(selector) => render_model_selector(frame, app, &selector),
            Overlay::Compiler(progress) => render_compiler(frame, app, &progress),
            Overlay::Pedal(modal) => render_pedal_modal(frame, app, &modal),
            Overlay::Help => render_help(frame),
        }
    }
}

fn render_workspace(frame: &mut Frame<'_>, app: &mut App, area: Rect) {
    let sections = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Length(4),
            Constraint::Min(7),
            Constraint::Length(2),
        ])
        .split(area);
    render_header(frame, app, sections[0]);
    render_sequence(frame, app, sections[1]);
    render_board(frame, app, sections[2]);
    render_footer(frame, app, sections[3]);
}

fn render_header(frame: &mut Frame<'_>, app: &mut App, area: Rect) {
    let (identity, metadata) = if let Some(editor) = &app.editor {
        (
            editor.package_id().to_string(),
            format!(
                "{} source pedals  ·  {} instances  ·  context {}",
                editor.source_pedals().len(),
                editor.instances().len(),
                editor.max_context_activations()
            ),
        )
    } else {
        (
            "NO MODEL".to_string(),
            "Open a compiled package or transpile a Safetensors source".to_string(),
        )
    };
    let title = Line::from(vec![
        Span::styled(" llmoop ", Style::default().fg(SIGNAL).bold()),
        Span::styled("SIGNAL BOARD", Style::default().fg(META)),
        Span::raw("  "),
        Span::styled(identity, Style::default().fg(TEXT).bold()),
    ]);
    let block = Block::default()
        .borders(Borders::BOTTOM)
        .border_style(Style::default().fg(QUIET));
    let inner = block.inner(area);
    frame.render_widget(block, area);
    frame.render_widget(
        Paragraph::new(vec![
            title,
            Line::styled(metadata, Style::default().fg(META)),
        ]),
        inner,
    );
    let button_width = 14u16.min(inner.width);
    if button_width > 0 {
        let button = Rect::new(
            inner.right().saturating_sub(button_width),
            inner.y,
            button_width,
            1,
        );
        frame.render_widget(
            Paragraph::new("[ open model ]")
                .alignment(Alignment::Right)
                .style(Style::default().fg(COOL)),
            button,
        );
        app.hit_map.insert(button, HitTarget::OpenModel);
    }
}

fn render_sequence(frame: &mut Frame<'_>, app: &mut App, area: Rect) {
    let focused = app.focus == FocusRegion::Sequence && app.overlay.is_none();
    let border_color = if app.sequence_error.is_some() {
        FAULT
    } else if focused {
        SIGNAL
    } else {
        QUIET
    };
    let block = Block::default()
        .title(Span::styled(
            " NUMERIC SEQUENCE · ZERO-BASED ",
            Style::default().fg(if focused { SIGNAL } else { META }),
        ))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(border_color));
    let inner = block.inner(area);
    frame.render_widget(block, area);
    app.hit_map.insert(area, HitTarget::Sequence);
    if app.editor.is_none() {
        frame.render_widget(
            Paragraph::new("Load a model to edit its runtime layer sequence")
                .style(Style::default().fg(QUIET)),
            inner,
        );
        return;
    }
    let width = inner.width.saturating_sub(1) as usize;
    let (start, line, cursor_column) = buffer_line(&app.sequence, width.max(1));
    frame.render_widget(
        Paragraph::new(line),
        Rect::new(inner.x, inner.y, inner.width, 1),
    );
    if let Some(error) = &app.sequence_error {
        let message = format!("! {error}");
        frame.render_widget(
            Paragraph::new(truncate(&message, inner.width as usize))
                .style(Style::default().fg(FAULT)),
            Rect::new(inner.x, inner.y.saturating_add(1), inner.width, 1),
        );
    } else {
        frame.render_widget(
            Paragraph::new("valid draft · edits update the board immediately")
                .style(Style::default().fg(META)),
            Rect::new(inner.x, inner.y.saturating_add(1), inner.width, 1),
        );
    }
    if focused && app.sequence.cursor() >= start {
        frame.set_cursor_position(Position::new(
            inner.x.saturating_add(cursor_column as u16),
            inner.y,
        ));
    }
}

fn render_board(frame: &mut Frame<'_>, app: &mut App, area: Rect) {
    let focused = app.focus == FocusRegion::Board && app.overlay.is_none();
    let block = Block::default()
        .title(Span::styled(
            " LIVE PEDALBOARD · DRAFT ",
            Style::default().fg(if focused { SIGNAL } else { META }),
        ))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(if focused { SIGNAL } else { QUIET }));
    let inner = block.inner(area);
    frame.render_widget(block, area);
    let instances = app.instances();
    if instances.is_empty() {
        frame.render_widget(
            Paragraph::new(vec![
                Line::styled("The signal chain is empty.", Style::default().fg(TEXT)),
                Line::styled(
                    "Open a model to populate reusable source pedals.",
                    Style::default().fg(META),
                ),
            ])
            .alignment(Alignment::Center),
            centered_line_area(inner, 2),
        );
        return;
    }

    let pedal_width = if inner.width >= 80 { 14 } else { 10 };
    let cable_width = if inner.width >= 80 { 5 } else { 3 };
    let unit_width = pedal_width + cable_width;
    let visible_count = ((inner.width + cable_width) / unit_width).max(1) as usize;
    let selected_index = app
        .selected_instance_id
        .as_ref()
        .and_then(|id| {
            instances
                .iter()
                .position(|instance| &instance.instance_id == id)
        })
        .unwrap_or(0);
    if selected_index < app.board_scroll {
        app.board_scroll = selected_index;
    }
    if selected_index >= app.board_scroll + visible_count {
        app.board_scroll = selected_index + 1 - visible_count;
    }
    app.board_scroll = app
        .board_scroll
        .min(instances.len().saturating_sub(visible_count));

    let shown = instances
        .iter()
        .skip(app.board_scroll)
        .take(visible_count)
        .collect::<Vec<_>>();
    let board_y = inner.y + inner.height.saturating_sub(6) / 2;
    let device_colors = device_color_map(&instances);
    for (visible_index, instance) in shown.iter().enumerate() {
        let absolute_index = app.board_scroll + visible_index;
        let x = inner.x + visible_index as u16 * unit_width;
        if visible_index > 0 {
            let previous = &instances[absolute_index - 1];
            let transition = previous.device_id != instance.device_id;
            render_cable(
                frame,
                Rect::new(x.saturating_sub(cable_width), board_y + 2, cable_width, 1),
                transition,
            );
        }
        let pedal_area = Rect::new(x, board_y, pedal_width.min(inner.right() - x), 5);
        render_pedal(
            frame,
            pedal_area,
            instance,
            app.selected_instance_id.as_deref() == Some(instance.instance_id.as_str()),
            focused,
            *device_colors.get(&instance.device_id).unwrap_or(&COOL),
            app.editor.as_ref().map(|editor| editor.available_devices()),
        );
        app.hit_map
            .insert(pedal_area, HitTarget::Pedal(instance.instance_id.clone()));
    }
    if app.board_scroll > 0 {
        let left = Rect::new(inner.x, inner.bottom().saturating_sub(1), 3, 1);
        frame.render_widget(Paragraph::new("◀").style(Style::default().fg(SIGNAL)), left);
        app.hit_map.insert(left, HitTarget::PanLeft);
    }
    if app.board_scroll + shown.len() < instances.len() {
        let right = Rect::new(inner.right().saturating_sub(3), inner.bottom() - 1, 3, 1);
        frame.render_widget(
            Paragraph::new("▶")
                .alignment(Alignment::Right)
                .style(Style::default().fg(SIGNAL)),
            right,
        );
        app.hit_map.insert(right, HitTarget::PanRight);
    }
    let position = format!(
        "signal {}–{} / {}",
        app.board_scroll + 1,
        app.board_scroll + shown.len(),
        instances.len()
    );
    frame.render_widget(
        Paragraph::new(position)
            .alignment(Alignment::Center)
            .style(Style::default().fg(META)),
        Rect::new(inner.x, inner.bottom().saturating_sub(1), inner.width, 1),
    );
}

fn render_cable(frame: &mut Frame<'_>, area: Rect, transition: bool) {
    let cable = if transition {
        match area.width {
            0 => "".to_string(),
            1 => "◆".to_string(),
            2 => "◆┄".to_string(),
            width => format!(
                "{}◆{}",
                "┄".repeat((width as usize - 1) / 2),
                "┄".repeat(width as usize / 2)
            ),
        }
    } else {
        "─".repeat(area.width as usize)
    };
    frame.render_widget(
        Paragraph::new(cable).style(Style::default().fg(SIGNAL).bold()),
        area,
    );
}

fn render_pedal(
    frame: &mut Frame<'_>,
    area: Rect,
    instance: &RuntimeEditorInstance,
    selected: bool,
    board_focused: bool,
    device_color: Color,
    devices: Option<&[RuntimeAvailableDevice]>,
) {
    let device_available = devices.is_some_and(|devices| {
        devices.iter().any(|device| {
            device.device_id == instance.device_id
                && device.available
                && device.can_host_runtime_pedals_on_physical_device != Some(false)
        })
    });
    let border_color = if !device_available {
        FAULT
    } else if selected {
        SIGNAL
    } else {
        device_color
    };
    let title = if instance.occurrence > 1 {
        format!(" {}^{} ", instance.source_id, instance.occurrence)
    } else {
        format!(" {} ", instance.source_id)
    };
    let block = Block::default()
        .title(Span::styled(
            title,
            Style::default()
                .fg(if selected { SIGNAL } else { TEXT })
                .bold(),
        ))
        .borders(Borders::ALL)
        .border_type(if selected && board_focused {
            BorderType::Double
        } else {
            BorderType::Rounded
        })
        .border_style(Style::default().fg(border_color));
    let inner = block.inner(area);
    frame.render_widget(block, area);
    let state = if !device_available {
        "! UNAVAILABLE"
    } else if instance.enabled {
        "● ACTIVE"
    } else {
        "○ BYPASS"
    };
    frame.render_widget(
        Paragraph::new(vec![
            Line::styled(
                truncate(&short_device_label(instance, devices), inner.width as usize),
                Style::default().fg(device_color),
            ),
            Line::styled(
                truncate(state, inner.width as usize),
                Style::default().fg(if instance.enabled { META } else { FAULT }),
            ),
            Line::styled(
                format!("occ {}", instance.occurrence),
                Style::default().fg(QUIET),
            ),
        ]),
        inner,
    );
}

fn render_footer(frame: &mut Frame<'_>, app: &mut App, area: Rect) {
    let help = if app.overlay.is_some() {
        "Tab focus · Enter activate · Esc close/cancel · F1 help"
    } else {
        match app.focus {
            FocusRegion::Sequence => {
                "Type/paste sequence · Shift+arrows select · Tab board · Ctrl+O model · F1 help"
            }
            FocusRegion::Board => {
                "←/→ select · Enter edit · Ctrl+D duplicate · Del remove · Alt+←/→ reorder · Tab sequence"
            }
        }
    };
    frame.render_widget(
        Paragraph::new(vec![
            Line::styled(
                truncate(&app.status, area.width as usize),
                Style::default().fg(TEXT),
            ),
            Line::styled(
                truncate(help, area.width as usize),
                Style::default().fg(META),
            ),
        ]),
        area,
    );
}

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
                Span::styled("compiled llmoop package", Style::default().fg(TEXT).bold()),
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

fn render_pedal_modal(frame: &mut Frame<'_>, app: &mut App, modal: &PedalModalState) {
    let area = centered_rect(frame.area(), 70, 78, 54, 20);
    frame.render_widget(Clear, area);
    let title = format!(
        " PEDAL {} · OCCURRENCE {} ",
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
        PedalPolicyKind::Independent => "Independent",
        PedalPolicyKind::Clone => "Clone from another instance",
        PedalPolicyKind::Share => "Share with another instance",
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
    render_pedal_properties(frame, app, modal, rows[6]);
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

fn render_pedal_properties(
    frame: &mut Frame<'_>,
    app: &mut App,
    modal: &PedalModalState,
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
            Paragraph::new("No control schema declared by this source pedal.")
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

fn property_display(property: &super::app::PedalPropertyDraft) -> String {
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
        let Some(package) = std::env::var_os("LLMOOP_TEST_PACKAGE_DIR") else {
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
