use std::error::Error;
use std::io::{self, Stdout, stdout};
use std::time::Duration;

use crossterm::cursor::{Hide, Show};
use crossterm::event::{
    self, DisableBracketedPaste, DisableFocusChange, DisableMouseCapture, EnableBracketedPaste,
    EnableFocusChange, EnableMouseCapture, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers,
    MouseButton, MouseEvent, MouseEventKind,
};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;

use super::app::{App, AppAction, CursorMotion, FocusRegion, ModelSelectorFocus, Overlay};
use super::view;

pub fn run() -> Result<(), Box<dyn Error>> {
    let mut guard = TerminalGuard::enter(true)?;
    let backend = CrosstermBackend::new(stdout());
    let mut terminal = Terminal::new(backend)?;
    let mut app = App::new();
    let mut dirty = true;
    while !app.should_quit() {
        dirty |= app.poll_compiler();
        if app.take_terminal_reset_request() {
            let area = terminal.size()?;
            terminal.resize(area.into())?;
            dirty = true;
        }
        if dirty {
            terminal.draw(|frame| view::render(frame, &mut app))?;
            dirty = false;
        }
        if !event::poll(Duration::from_millis(50))? {
            continue;
        }
        let event = event::read()?;
        let redraw_for_event = matches!(&event, Event::Resize(_, _));
        let action = action_from_event(&app, event);
        if let Some(action) = action {
            let previous_mouse_capture = app.mouse_capture();
            app.dispatch(action);
            if app.mouse_capture() != previous_mouse_capture {
                guard.set_mouse_capture(app.mouse_capture())?;
            }
            dirty = true;
        } else if redraw_for_event {
            dirty = true;
        }
    }
    Ok(())
}

fn action_from_event(app: &App, event: Event) -> Option<AppAction> {
    match event {
        Event::Key(key) if matches!(key.kind, KeyEventKind::Press | KeyEventKind::Repeat) => {
            action_from_key(app, key)
        }
        Event::Paste(text) => Some(AppAction::InsertText(text)),
        Event::Mouse(mouse) => action_from_mouse(app, mouse),
        Event::Resize(_, _) | Event::FocusGained | Event::FocusLost | Event::Key(_) => None,
    }
}

fn action_from_key(app: &App, key: KeyEvent) -> Option<AppAction> {
    let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
    let alt = key.modifiers.contains(KeyModifiers::ALT);
    let shift = key.modifiers.contains(KeyModifiers::SHIFT);
    if ctrl {
        match key.code {
            KeyCode::Char('c') => return Some(AppAction::Quit),
            KeyCode::Char('o') => return Some(AppAction::OpenModelSelector),
            KeyCode::Char('m') => return Some(AppAction::ToggleMouseCapture),
            KeyCode::Char('r') => return Some(AppAction::RefreshDevices),
            KeyCode::Char('a') => return Some(AppAction::SelectAllText),
            KeyCode::Char('d') => return Some(AppAction::DuplicateSelected),
            _ => {}
        }
    }
    if key.code == KeyCode::F(1) {
        return Some(AppAction::ToggleHelp);
    }
    if key.code == KeyCode::Tab {
        return Some(if shift {
            AppAction::FocusPrevious
        } else {
            AppAction::FocusNext
        });
    }
    if key.code == KeyCode::BackTab {
        return Some(AppAction::FocusPrevious);
    }

    match app.overlay.as_ref() {
        Some(Overlay::ModelSelector(selector)) => match key.code {
            KeyCode::Esc => Some(AppAction::CloseOverlay),
            KeyCode::Enter => Some(AppAction::ActivateModal),
            KeyCode::Up => Some(AppAction::ModalPrevious),
            KeyCode::Down => Some(AppAction::ModalNext),
            KeyCode::Left if selector.focus == ModelSelectorFocus::Path => {
                Some(AppAction::MoveTextCursor {
                    motion: CursorMotion::Left,
                    selecting: shift,
                })
            }
            KeyCode::Right if selector.focus == ModelSelectorFocus::Path => {
                Some(AppAction::MoveTextCursor {
                    motion: CursorMotion::Right,
                    selecting: shift,
                })
            }
            KeyCode::Home if selector.focus == ModelSelectorFocus::Path => {
                Some(AppAction::MoveTextCursor {
                    motion: CursorMotion::Home,
                    selecting: shift,
                })
            }
            KeyCode::End if selector.focus == ModelSelectorFocus::Path => {
                Some(AppAction::MoveTextCursor {
                    motion: CursorMotion::End,
                    selecting: shift,
                })
            }
            KeyCode::Backspace => Some(AppAction::Backspace),
            KeyCode::Delete => Some(AppAction::DeleteForward),
            KeyCode::Char(character) if !ctrl && selector.focus == ModelSelectorFocus::Path => {
                Some(AppAction::InsertText(character.to_string()))
            }
            _ => None,
        },
        Some(Overlay::Compiler(_)) => match key.code {
            KeyCode::Esc => Some(AppAction::CancelCompiler),
            KeyCode::Up => Some(AppAction::ModalPrevious),
            KeyCode::Down => Some(AppAction::ModalNext),
            _ => None,
        },
        Some(Overlay::Pedal(_)) => match key.code {
            KeyCode::Esc => Some(AppAction::CloseOverlay),
            KeyCode::Up => Some(AppAction::ModalPrevious),
            KeyCode::Down => Some(AppAction::ModalNext),
            KeyCode::Left if app.modal_text_entry_active() && !alt => {
                Some(AppAction::MoveTextCursor {
                    motion: CursorMotion::Left,
                    selecting: shift,
                })
            }
            KeyCode::Right if app.modal_text_entry_active() && !alt => {
                Some(AppAction::MoveTextCursor {
                    motion: CursorMotion::Right,
                    selecting: shift,
                })
            }
            KeyCode::Left => Some(AppAction::ModalChange(-1)),
            KeyCode::Right => Some(AppAction::ModalChange(1)),
            KeyCode::Home if app.modal_text_entry_active() => Some(AppAction::MoveTextCursor {
                motion: CursorMotion::Home,
                selecting: shift,
            }),
            KeyCode::End if app.modal_text_entry_active() => Some(AppAction::MoveTextCursor {
                motion: CursorMotion::End,
                selecting: shift,
            }),
            KeyCode::Backspace if app.modal_text_entry_active() => Some(AppAction::Backspace),
            KeyCode::Delete if app.modal_text_entry_active() => Some(AppAction::DeleteForward),
            KeyCode::Char(character) if app.modal_text_entry_active() && !ctrl && !alt => {
                Some(AppAction::InsertText(character.to_string()))
            }
            KeyCode::Enter => Some(AppAction::ActivateModal),
            _ => None,
        },
        Some(Overlay::Help) => Some(AppAction::CloseOverlay),
        None if app.focus() == FocusRegion::Sequence => match key.code {
            KeyCode::Esc => Some(AppAction::FocusNext),
            KeyCode::Left => Some(AppAction::MoveTextCursor {
                motion: CursorMotion::Left,
                selecting: shift,
            }),
            KeyCode::Right => Some(AppAction::MoveTextCursor {
                motion: CursorMotion::Right,
                selecting: shift,
            }),
            KeyCode::Home => Some(AppAction::MoveTextCursor {
                motion: CursorMotion::Home,
                selecting: shift,
            }),
            KeyCode::End => Some(AppAction::MoveTextCursor {
                motion: CursorMotion::End,
                selecting: shift,
            }),
            KeyCode::Backspace => Some(AppAction::Backspace),
            KeyCode::Delete => Some(AppAction::DeleteForward),
            KeyCode::Char(character) if !ctrl && !alt => {
                Some(AppAction::InsertText(character.to_string()))
            }
            _ => None,
        },
        None => match key.code {
            KeyCode::Char('q') if !ctrl && !alt => Some(AppAction::Quit),
            KeyCode::Left if alt => Some(AppAction::MoveSelected(-1)),
            KeyCode::Right if alt => Some(AppAction::MoveSelected(1)),
            KeyCode::Left | KeyCode::Char('h') => Some(AppAction::SelectPreviousPedal),
            KeyCode::Right | KeyCode::Char('l') => Some(AppAction::SelectNextPedal),
            KeyCode::Home => Some(AppAction::SelectFirstPedal),
            KeyCode::End => Some(AppAction::SelectLastPedal),
            KeyCode::Enter => Some(AppAction::OpenSelectedPedal),
            KeyCode::Delete => Some(AppAction::RemoveSelected),
            _ => None,
        },
    }
}

fn action_from_mouse(app: &App, mouse: MouseEvent) -> Option<AppAction> {
    match mouse.kind {
        MouseEventKind::Down(MouseButton::Left) => app.action_at(mouse.column, mouse.row),
        MouseEventKind::ScrollUp => match app.overlay {
            Some(Overlay::ModelSelector(_)) | Some(Overlay::Compiler(_)) => {
                Some(AppAction::ModalPrevious)
            }
            Some(Overlay::Pedal(_)) => Some(AppAction::ModalChange(-1)),
            _ => Some(AppAction::PanBoard(-1)),
        },
        MouseEventKind::ScrollDown => match app.overlay {
            Some(Overlay::ModelSelector(_)) | Some(Overlay::Compiler(_)) => {
                Some(AppAction::ModalNext)
            }
            Some(Overlay::Pedal(_)) => Some(AppAction::ModalChange(1)),
            _ => Some(AppAction::PanBoard(1)),
        },
        _ => None,
    }
}

struct TerminalGuard {
    stdout: Stdout,
    mouse_capture: bool,
}

impl TerminalGuard {
    fn enter(mouse_capture: bool) -> io::Result<Self> {
        enable_raw_mode()?;
        let mut stdout = stdout();
        execute!(
            stdout,
            EnterAlternateScreen,
            EnableBracketedPaste,
            EnableFocusChange,
            Hide
        )?;
        if mouse_capture {
            execute!(stdout, EnableMouseCapture)?;
        }
        Ok(Self {
            stdout,
            mouse_capture,
        })
    }

    fn set_mouse_capture(&mut self, enabled: bool) -> io::Result<()> {
        if enabled == self.mouse_capture {
            return Ok(());
        }
        if enabled {
            execute!(self.stdout, EnableMouseCapture)?;
        } else {
            execute!(self.stdout, DisableMouseCapture)?;
        }
        self.mouse_capture = enabled;
        Ok(())
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        if self.mouse_capture {
            let _ = execute!(self.stdout, DisableMouseCapture);
        }
        let _ = execute!(
            self.stdout,
            DisableBracketedPaste,
            DisableFocusChange,
            Show,
            LeaveAlternateScreen
        );
        let _ = disable_raw_mode();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_mapping_keeps_board_and_text_meanings_separate() {
        let mut app = App::new();
        app.overlay = None;
        app.focus = FocusRegion::Board;
        let left = KeyEvent::new(KeyCode::Left, KeyModifiers::NONE);
        assert_eq!(
            action_from_key(&app, left),
            Some(AppAction::SelectPreviousPedal)
        );
        app.focus = FocusRegion::Sequence;
        assert_eq!(
            action_from_key(&app, left),
            Some(AppAction::MoveTextCursor {
                motion: CursorMotion::Left,
                selecting: false
            })
        );
    }

    #[test]
    fn mouse_hit_resolves_to_same_open_pedal_action() {
        let mut app = App::new();
        app.hit_map.insert(
            ratatui::layout::Rect::new(2, 3, 10, 4),
            super::super::app::HitTarget::Pedal("layer_00".to_string()),
        );
        let mouse = MouseEvent {
            kind: MouseEventKind::Down(MouseButton::Left),
            column: 5,
            row: 4,
            modifiers: KeyModifiers::NONE,
        };
        assert_eq!(
            action_from_mouse(&app, mouse),
            Some(AppAction::OpenPedal("layer_00".to_string()))
        );
    }
}
