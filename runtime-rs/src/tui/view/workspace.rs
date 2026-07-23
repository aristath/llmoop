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
        Span::styled(" nerve ", Style::default().fg(SIGNAL).bold()),
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

