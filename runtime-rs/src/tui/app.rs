use std::collections::BTreeSet;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitStatus;

use ratatui::layout::Rect;
use serde_json::Value;

use crate::{
    RuntimeEditorInstance, RuntimeEditorSourcePedal, RuntimeModelEditor, RuntimeModelPathKind,
    StreamCircuitPedalInstanceStatePolicy, classify_runtime_model_path,
};

use super::compiler::{
    CompilerEvent, CompilerJob, CompilerJobKind, CompilerLaunch, CompilerMessage,
};
use super::sequence::{SequenceParseError, TextBuffer, parse_layer_sequence};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FocusRegion {
    Sequence,
    Board,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CursorMotion {
    Left,
    Right,
    Home,
    End,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AppAction {
    Quit,
    OpenModelSelector,
    CloseOverlay,
    ToggleHelp,
    ToggleMouseCapture,
    RefreshDevices,
    FocusNext,
    FocusPrevious,
    FocusSequence,
    FocusModelPath,
    InsertText(String),
    Backspace,
    DeleteForward,
    MoveTextCursor {
        motion: CursorMotion,
        selecting: bool,
    },
    SelectAllText,
    SelectPreviousPedal,
    SelectNextPedal,
    SelectFirstPedal,
    SelectLastPedal,
    OpenSelectedPedal,
    SelectPedal(String),
    OpenPedal(String),
    PanBoard(i16),
    DuplicateSelected,
    RemoveSelected,
    MoveSelected(i32),
    ActivateModal,
    ModalPrevious,
    ModalNext,
    ModalChange(i32),
    ModalClickRow(usize),
    ModelBrowserSelect(usize),
    ModelBrowserOpen(usize),
    ModelAction,
    ApplyPedal,
    CancelCompiler,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum HitTarget {
    OpenModel,
    Sequence,
    Pedal(String),
    PanLeft,
    PanRight,
    ModalApply,
    ModalCancel,
    ModalRow(usize),
    BrowserEntry(usize),
    ModelAction,
    CompilerCancel,
    ModelPath,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct HitMap {
    entries: Vec<(Rect, HitTarget)>,
}

impl HitMap {
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    pub fn insert(&mut self, area: Rect, target: HitTarget) {
        if area.width > 0 && area.height > 0 {
            self.entries.push((area, target));
        }
    }

    pub fn resolve(&self, column: u16, row: u16) -> Option<&HitTarget> {
        self.entries.iter().rev().find_map(|(area, target)| {
            (column >= area.x
                && column < area.x.saturating_add(area.width)
                && row >= area.y
                && row < area.y.saturating_add(area.height))
            .then_some(target)
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ModelSelectorFocus {
    Path,
    Browser,
    Action,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct BrowserEntry {
    pub path: PathBuf,
    pub label: String,
    pub is_directory: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct SourceDiscovery {
    pub model_type: String,
    pub architecture: Vec<String>,
    pub weight_files: Vec<String>,
    pub tokenizer_files: Vec<String>,
    pub has_chat_template: bool,
    pub raw: Value,
}

impl SourceDiscovery {
    fn from_event(event: &CompilerEvent) -> Option<Self> {
        let source = event.value("source")?.clone();
        Some(Self {
            model_type: source
                .get("model_type")
                .and_then(Value::as_str)
                .unwrap_or("unknown")
                .to_string(),
            architecture: source
                .get("architecture")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
                .map(ToOwned::to_owned)
                .collect(),
            weight_files: source
                .get("weight_files")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
                .map(ToOwned::to_owned)
                .collect(),
            tokenizer_files: source
                .get("tokenizer_files")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
                .map(ToOwned::to_owned)
                .collect(),
            has_chat_template: source
                .get("has_chat_template")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            raw: source,
        })
    }
}

#[derive(Clone, Debug)]
pub(crate) struct ModelSelectorState {
    pub path: TextBuffer,
    pub browser_directory: PathBuf,
    pub entries: Vec<BrowserEntry>,
    pub selected_entry: usize,
    pub focus: ModelSelectorFocus,
    pub detected: Option<Result<RuntimeModelPathKind, String>>,
    pub discovery: Option<SourceDiscovery>,
    pub diagnostics: Vec<String>,
    pub diagnostic_scroll: usize,
}

impl ModelSelectorState {
    fn new(initial_path: PathBuf) -> Self {
        let mut state = Self {
            path: TextBuffer::new(initial_path.display().to_string()),
            browser_directory: initial_path,
            entries: Vec::new(),
            selected_entry: 0,
            focus: ModelSelectorFocus::Path,
            detected: None,
            discovery: None,
            diagnostics: Vec::new(),
            diagnostic_scroll: 0,
        };
        state.refresh();
        state
    }

    fn selected_path(&self) -> PathBuf {
        expand_home(self.path.text())
    }

    fn refresh(&mut self) {
        let selected = self.selected_path();
        self.detected = Some(classify_runtime_model_path(&selected).map_err(|error| error.0));
        self.discovery = None;
        let browser_directory = if selected.is_dir() {
            selected.clone()
        } else {
            selected
                .parent()
                .map(Path::to_path_buf)
                .unwrap_or_else(|| PathBuf::from("/"))
        };
        self.browser_directory = browser_directory;
        self.entries = browser_entries(&self.browser_directory);
        self.selected_entry = self
            .selected_entry
            .min(self.entries.len().saturating_sub(1));
    }

    fn set_path(&mut self, path: PathBuf) {
        self.path.set(path.display().to_string());
        self.diagnostics.clear();
        self.refresh();
    }

    pub(crate) fn current_action_label(&self) -> &'static str {
        match self.detected.as_ref() {
            Some(Ok(RuntimeModelPathKind::CompiledPackage { .. })) => "Load model",
            Some(Ok(RuntimeModelPathKind::SafetensorsSource { .. }))
                if self.discovery.is_some() =>
            {
                "Transpile and load"
            }
            Some(Ok(RuntimeModelPathKind::SafetensorsSource { .. })) => "Inspect source",
            _ => "Unavailable",
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct CompilerProgressState {
    pub kind: CompilerJobKind,
    pub source_path: PathBuf,
    pub selector: ModelSelectorState,
    pub stage: String,
    pub current: Option<u64>,
    pub total: Option<u64>,
    pub current_item: Option<String>,
    pub events: Vec<CompilerEvent>,
    pub diagnostics: Vec<String>,
    pub diagnostic_scroll: usize,
    pub cancelling: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PedalPolicyKind {
    Independent,
    Clone,
    Share,
}

#[derive(Clone, Debug)]
pub(crate) struct PedalModalState {
    pub instance_id: String,
    pub source: RuntimeEditorSourcePedal,
    pub layer_index: usize,
    pub occurrence: usize,
    pub device_ids: Vec<String>,
    pub device_labels: Vec<String>,
    pub device_index: usize,
    pub enabled: bool,
    pub policy: PedalPolicyKind,
    pub policy_targets: Vec<String>,
    pub policy_target_index: usize,
    pub focus_row: usize,
    pub error: Option<String>,
}

impl PedalModalState {
    fn row_count(&self) -> usize {
        6
    }

    fn state_policy(&self) -> StreamCircuitPedalInstanceStatePolicy {
        match self.policy {
            PedalPolicyKind::Independent => StreamCircuitPedalInstanceStatePolicy::Fresh,
            PedalPolicyKind::Clone => StreamCircuitPedalInstanceStatePolicy::CloneFrom {
                instance_id: self
                    .policy_targets
                    .get(self.policy_target_index)
                    .cloned()
                    .unwrap_or_default(),
            },
            PedalPolicyKind::Share => StreamCircuitPedalInstanceStatePolicy::ShareWith {
                instance_id: self
                    .policy_targets
                    .get(self.policy_target_index)
                    .cloned()
                    .unwrap_or_default(),
            },
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) enum Overlay {
    ModelSelector(ModelSelectorState),
    Compiler(CompilerProgressState),
    Pedal(PedalModalState),
    Help,
}

pub struct App {
    pub(crate) editor: Option<RuntimeModelEditor>,
    pub(crate) sequence: TextBuffer,
    pub(crate) sequence_error: Option<SequenceParseError>,
    pub(crate) last_valid_sequence: Vec<usize>,
    pub(crate) focus: FocusRegion,
    pub(crate) selected_instance_id: Option<String>,
    pub(crate) board_scroll: usize,
    pub(crate) overlay: Option<Overlay>,
    pub(crate) status: String,
    pub(crate) should_quit: bool,
    pub(crate) mouse_capture: bool,
    pub(crate) hit_map: HitMap,
    pub(crate) compiler_job: Option<CompilerJob>,
    compiler_launch: Result<CompilerLaunch, String>,
    terminal_reset_requested: bool,
}

impl App {
    pub fn new() -> Self {
        let initial_path = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        Self {
            editor: None,
            sequence: TextBuffer::new("[]"),
            sequence_error: None,
            last_valid_sequence: Vec::new(),
            focus: FocusRegion::Board,
            selected_instance_id: None,
            board_scroll: 0,
            overlay: Some(Overlay::ModelSelector(ModelSelectorState::new(
                initial_path,
            ))),
            status: "No model loaded · draft not mounted".to_string(),
            should_quit: false,
            mouse_capture: true,
            hit_map: HitMap::default(),
            compiler_job: None,
            compiler_launch: CompilerLaunch::from_environment(),
            terminal_reset_requested: false,
        }
    }

    pub fn should_quit(&self) -> bool {
        self.should_quit
    }

    pub fn mouse_capture(&self) -> bool {
        self.mouse_capture
    }

    pub fn has_overlay(&self) -> bool {
        self.overlay.is_some()
    }

    pub fn focus(&self) -> FocusRegion {
        self.focus
    }

    pub fn selected_instance(&self) -> Option<&str> {
        self.selected_instance_id.as_deref()
    }

    pub fn load_compiled_model(&mut self, path: impl AsRef<Path>) {
        self.load_model(path.as_ref().to_path_buf());
    }

    pub fn take_terminal_reset_request(&mut self) -> bool {
        std::mem::take(&mut self.terminal_reset_requested)
    }

    pub fn action_at(&self, column: u16, row: u16) -> Option<AppAction> {
        match self.hit_map.resolve(column, row)? {
            HitTarget::OpenModel => Some(AppAction::OpenModelSelector),
            HitTarget::Sequence => Some(AppAction::FocusSequence),
            HitTarget::Pedal(instance_id) => Some(AppAction::OpenPedal(instance_id.clone())),
            HitTarget::PanLeft => Some(AppAction::PanBoard(-1)),
            HitTarget::PanRight => Some(AppAction::PanBoard(1)),
            HitTarget::ModalApply => Some(AppAction::ApplyPedal),
            HitTarget::ModalCancel => Some(AppAction::CloseOverlay),
            HitTarget::ModalRow(row) => Some(AppAction::ModalClickRow(*row)),
            HitTarget::BrowserEntry(index) => Some(AppAction::ModelBrowserOpen(*index)),
            HitTarget::ModelAction => Some(AppAction::ModelAction),
            HitTarget::CompilerCancel => Some(AppAction::CancelCompiler),
            HitTarget::ModelPath => Some(AppAction::FocusModelPath),
        }
    }

    pub fn dispatch(&mut self, action: AppAction) {
        if self.overlay.is_some() {
            self.dispatch_overlay(action);
            return;
        }
        match action {
            AppAction::Quit => self.should_quit = true,
            AppAction::OpenModelSelector => self.open_model_selector(),
            AppAction::ToggleHelp => self.overlay = Some(Overlay::Help),
            AppAction::ToggleMouseCapture => self.mouse_capture = !self.mouse_capture,
            AppAction::RefreshDevices => {
                if let Some(editor) = &mut self.editor {
                    editor.refresh_devices();
                    self.terminal_reset_requested = true;
                    self.status = format!(
                        "Detected {} runtime device target(s)",
                        editor.available_devices().len()
                    );
                }
            }
            AppAction::FocusNext | AppAction::FocusPrevious => {
                self.focus = match self.focus {
                    FocusRegion::Sequence => FocusRegion::Board,
                    FocusRegion::Board => FocusRegion::Sequence,
                };
            }
            AppAction::FocusSequence => self.focus = FocusRegion::Sequence,
            AppAction::InsertText(value) if self.focus == FocusRegion::Sequence => {
                self.sequence.insert(&value);
                self.apply_sequence_text();
            }
            AppAction::Backspace if self.focus == FocusRegion::Sequence => {
                self.sequence.backspace();
                self.apply_sequence_text();
            }
            AppAction::DeleteForward if self.focus == FocusRegion::Sequence => {
                self.sequence.delete();
                self.apply_sequence_text();
            }
            AppAction::MoveTextCursor { motion, selecting }
                if self.focus == FocusRegion::Sequence =>
            {
                move_text_cursor(&mut self.sequence, motion, selecting);
            }
            AppAction::SelectAllText if self.focus == FocusRegion::Sequence => {
                self.sequence.select_all();
            }
            AppAction::SelectPreviousPedal => self.select_relative(-1),
            AppAction::SelectNextPedal => self.select_relative(1),
            AppAction::SelectFirstPedal => self.select_index(0),
            AppAction::SelectLastPedal => {
                let last = self.instance_count().saturating_sub(1);
                self.select_index(last);
            }
            AppAction::OpenSelectedPedal => self.open_selected_pedal(),
            AppAction::SelectPedal(instance_id) => self.select_instance(&instance_id),
            AppAction::OpenPedal(instance_id) => {
                self.select_instance(&instance_id);
                self.open_selected_pedal();
            }
            AppAction::PanBoard(delta) => {
                self.board_scroll = self.board_scroll.saturating_add_signed(delta as isize)
            }
            AppAction::DuplicateSelected => self.duplicate_selected(),
            AppAction::RemoveSelected => self.remove_selected(),
            AppAction::MoveSelected(delta) => self.move_selected(delta),
            _ => {}
        }
    }

    pub fn poll_compiler(&mut self) -> bool {
        let Some(mut job) = self.compiler_job.take() else {
            return false;
        };
        let messages = job.drain_messages();
        let mut changed = !messages.is_empty();
        for message in messages {
            self.handle_compiler_message(message);
        }
        let process_status = job.try_status();
        let keep_job = matches!(self.overlay, Some(Overlay::Compiler(_)));
        match process_status {
            Ok(Some(status)) if keep_job => {
                self.handle_compiler_exit(status, job.terminal_event_received());
                changed = true;
            }
            Err(error) if keep_job => {
                self.fail_compiler_job(error);
                changed = true;
            }
            _ if keep_job => self.compiler_job = Some(job),
            _ => {}
        }
        changed
    }

    fn dispatch_overlay(&mut self, action: AppAction) {
        match self.overlay.as_ref() {
            Some(Overlay::ModelSelector(_)) => self.dispatch_model_selector(action),
            Some(Overlay::Compiler(_)) => self.dispatch_compiler(action),
            Some(Overlay::Pedal(_)) => self.dispatch_pedal_modal(action),
            Some(Overlay::Help) => match action {
                AppAction::Quit => self.should_quit = true,
                AppAction::CloseOverlay | AppAction::ToggleHelp => self.overlay = None,
                _ => {}
            },
            None => {}
        }
    }

    fn dispatch_model_selector(&mut self, action: AppAction) {
        if matches!(action, AppAction::ActivateModal) {
            let browser_index = match &self.overlay {
                Some(Overlay::ModelSelector(selector))
                    if selector.focus == ModelSelectorFocus::Browser =>
                {
                    Some(selector.selected_entry)
                }
                _ => None,
            };
            if let Some(index) = browser_index {
                self.open_browser_entry(index);
                return;
            }
        }
        let Some(Overlay::ModelSelector(selector)) = &mut self.overlay else {
            return;
        };
        match action {
            AppAction::Quit => self.should_quit = true,
            AppAction::CloseOverlay => self.overlay = None,
            AppAction::ToggleMouseCapture => self.mouse_capture = !self.mouse_capture,
            AppAction::FocusNext => {
                selector.focus = match selector.focus {
                    ModelSelectorFocus::Path => ModelSelectorFocus::Browser,
                    ModelSelectorFocus::Browser => ModelSelectorFocus::Action,
                    ModelSelectorFocus::Action => ModelSelectorFocus::Path,
                }
            }
            AppAction::FocusPrevious => {
                selector.focus = match selector.focus {
                    ModelSelectorFocus::Path => ModelSelectorFocus::Action,
                    ModelSelectorFocus::Browser => ModelSelectorFocus::Path,
                    ModelSelectorFocus::Action => ModelSelectorFocus::Browser,
                }
            }
            AppAction::FocusModelPath => selector.focus = ModelSelectorFocus::Path,
            AppAction::InsertText(value) if selector.focus == ModelSelectorFocus::Path => {
                selector.path.insert(&value);
                selector.refresh();
            }
            AppAction::Backspace if selector.focus == ModelSelectorFocus::Path => {
                selector.path.backspace();
                selector.refresh();
            }
            AppAction::DeleteForward if selector.focus == ModelSelectorFocus::Path => {
                selector.path.delete();
                selector.refresh();
            }
            AppAction::MoveTextCursor { motion, selecting }
                if selector.focus == ModelSelectorFocus::Path =>
            {
                move_text_cursor(&mut selector.path, motion, selecting);
            }
            AppAction::SelectAllText if selector.focus == ModelSelectorFocus::Path => {
                selector.path.select_all();
            }
            AppAction::ModalPrevious if selector.focus == ModelSelectorFocus::Browser => {
                selector.selected_entry = selector.selected_entry.saturating_sub(1);
            }
            AppAction::ModalNext if selector.focus == ModelSelectorFocus::Browser => {
                selector.selected_entry =
                    (selector.selected_entry + 1).min(selector.entries.len().saturating_sub(1));
            }
            AppAction::ModelBrowserSelect(index) => {
                selector.selected_entry = index.min(selector.entries.len().saturating_sub(1));
                selector.focus = ModelSelectorFocus::Browser;
            }
            AppAction::ModelBrowserOpen(index) => self.open_browser_entry(index),
            AppAction::ActivateModal | AppAction::ModelAction => self.activate_model_action(),
            _ => {}
        }
    }

    fn dispatch_compiler(&mut self, action: AppAction) {
        match action {
            AppAction::Quit | AppAction::CancelCompiler | AppAction::CloseOverlay => {
                if let Some(job) = &mut self.compiler_job {
                    if let Err(error) = job.cancel() {
                        if let Some(Overlay::Compiler(progress)) = &mut self.overlay {
                            progress.diagnostics.push(error);
                        }
                    } else if let Some(Overlay::Compiler(progress)) = &mut self.overlay {
                        progress.cancelling = true;
                        progress.stage = "Cancellation requested".to_string();
                    }
                }
            }
            AppAction::ModalPrevious => {
                if let Some(Overlay::Compiler(progress)) = &mut self.overlay {
                    progress.diagnostic_scroll = progress.diagnostic_scroll.saturating_sub(1);
                }
            }
            AppAction::ModalNext => {
                if let Some(Overlay::Compiler(progress)) = &mut self.overlay {
                    progress.diagnostic_scroll = progress.diagnostic_scroll.saturating_add(1);
                }
            }
            _ => {}
        }
    }

    fn dispatch_pedal_modal(&mut self, action: AppAction) {
        let Some(Overlay::Pedal(modal)) = &mut self.overlay else {
            return;
        };
        match action {
            AppAction::Quit => self.should_quit = true,
            AppAction::CloseOverlay => self.overlay = None,
            AppAction::FocusNext | AppAction::ModalNext => {
                modal.focus_row = (modal.focus_row + 1) % modal.row_count()
            }
            AppAction::FocusPrevious | AppAction::ModalPrevious => {
                modal.focus_row = modal
                    .focus_row
                    .checked_sub(1)
                    .unwrap_or_else(|| modal.row_count() - 1)
            }
            AppAction::ModalChange(delta) => change_pedal_modal(modal, delta),
            AppAction::ModalClickRow(row) => {
                modal.focus_row = row.min(modal.row_count().saturating_sub(1));
                if row <= 3 {
                    change_pedal_modal(modal, 1);
                } else if row == 4 {
                    self.apply_pedal_modal();
                } else if row == 5 {
                    self.overlay = None;
                }
            }
            AppAction::ActivateModal => {
                if modal.focus_row == 1 {
                    modal.enabled = !modal.enabled;
                } else if matches!(modal.focus_row, 0 | 2 | 3) {
                    change_pedal_modal(modal, 1);
                } else if modal.focus_row == 4 {
                    self.apply_pedal_modal();
                } else if modal.focus_row == 5 {
                    self.overlay = None;
                }
            }
            AppAction::ApplyPedal => self.apply_pedal_modal(),
            _ => {}
        }
    }

    fn open_model_selector(&mut self) {
        let initial = self
            .editor
            .as_ref()
            .and_then(|editor| editor.package_manifest_path().parent())
            .map(Path::to_path_buf)
            .or_else(|| env::current_dir().ok())
            .unwrap_or_else(|| PathBuf::from("."));
        self.overlay = Some(Overlay::ModelSelector(ModelSelectorState::new(initial)));
    }

    fn open_browser_entry(&mut self, index: usize) {
        let Some(Overlay::ModelSelector(selector)) = &mut self.overlay else {
            return;
        };
        let Some(entry) = selector.entries.get(index).cloned() else {
            return;
        };
        selector.set_path(entry.path);
        selector.focus = if entry.is_directory {
            ModelSelectorFocus::Browser
        } else {
            ModelSelectorFocus::Path
        };
    }

    fn activate_model_action(&mut self) {
        let Some(Overlay::ModelSelector(selector)) = &self.overlay else {
            return;
        };
        let path = selector.selected_path();
        let detected = selector.detected.clone();
        let discovered = selector.discovery.is_some();
        match detected {
            Some(Ok(RuntimeModelPathKind::CompiledPackage { manifest })) => {
                self.load_model(manifest)
            }
            Some(Ok(RuntimeModelPathKind::SafetensorsSource { model_dir })) if discovered => {
                self.start_compiler_job(CompilerJobKind::Compilation, model_dir)
            }
            Some(Ok(RuntimeModelPathKind::SafetensorsSource { model_dir })) => {
                self.start_compiler_job(CompilerJobKind::Discovery, model_dir)
            }
            Some(Err(error)) => {
                if let Some(Overlay::ModelSelector(selector)) = &mut self.overlay {
                    selector.diagnostics = vec![error];
                }
            }
            None => {
                if let Some(Overlay::ModelSelector(selector)) = &mut self.overlay {
                    selector.diagnostics = vec![format!(
                        "Could not identify model source at {}",
                        path.display()
                    )];
                }
            }
        }
    }

    fn start_compiler_job(&mut self, kind: CompilerJobKind, source_path: PathBuf) {
        let launch = match &self.compiler_launch {
            Ok(launch) => launch,
            Err(error) => {
                if let Some(Overlay::ModelSelector(selector)) = &mut self.overlay {
                    selector.diagnostics = vec![error.clone()];
                }
                return;
            }
        };
        let job = match kind {
            CompilerJobKind::Discovery => launch.start_discovery(&source_path),
            CompilerJobKind::Compilation => launch.start_compilation(&source_path),
        };
        match job {
            Ok(job) => {
                let selector = match self.overlay.take() {
                    Some(Overlay::ModelSelector(selector)) => selector,
                    _ => ModelSelectorState::new(source_path.clone()),
                };
                self.overlay = Some(Overlay::Compiler(CompilerProgressState {
                    kind,
                    source_path,
                    selector,
                    stage: match kind {
                        CompilerJobKind::Discovery => "Starting model discovery",
                        CompilerJobKind::Compilation => "Starting model compilation",
                    }
                    .to_string(),
                    current: None,
                    total: None,
                    current_item: None,
                    events: Vec::new(),
                    diagnostics: Vec::new(),
                    diagnostic_scroll: 0,
                    cancelling: false,
                }));
                self.compiler_job = Some(job);
            }
            Err(error) => {
                if let Some(Overlay::ModelSelector(selector)) = &mut self.overlay {
                    selector.diagnostics = vec![error];
                }
            }
        }
    }

    fn handle_compiler_message(&mut self, message: CompilerMessage) {
        match message {
            CompilerMessage::Event(event) => self.handle_compiler_event(event),
            CompilerMessage::Diagnostic(diagnostic)
            | CompilerMessage::ProtocolError(diagnostic) => {
                if let Some(Overlay::Compiler(progress)) = &mut self.overlay {
                    progress.diagnostics.push(diagnostic);
                }
            }
        }
    }

    fn handle_compiler_event(&mut self, event: CompilerEvent) {
        let kind = match &self.overlay {
            Some(Overlay::Compiler(progress)) => progress.kind,
            _ => return,
        };
        if event.kind == "Completed" {
            match kind {
                CompilerJobKind::Discovery => {
                    let mut selector = match self.overlay.take() {
                        Some(Overlay::Compiler(progress)) => progress.selector,
                        _ => return,
                    };
                    selector.discovery = selector.discovery.or_else(|| {
                        event
                            .value("discovery")
                            .cloned()
                            .and_then(source_discovery_from_value)
                    });
                    selector.focus = ModelSelectorFocus::Action;
                    self.status = "Source discovery complete".to_string();
                    self.overlay = Some(Overlay::ModelSelector(selector));
                }
                CompilerJobKind::Compilation => {
                    let package = event
                        .nested_string("package", "package_manifest")
                        .map(PathBuf::from);
                    if let Some(package) = package {
                        self.load_model(package);
                    } else {
                        self.fail_compiler_job(
                            "Compiler completed without a package_manifest".to_string(),
                        );
                    }
                }
            }
            return;
        }
        if event.kind == "Cancelled" {
            let selector = match self.overlay.take() {
                Some(Overlay::Compiler(progress)) => progress.selector,
                _ => return,
            };
            self.status = "Model compiler cancelled; no package was published".to_string();
            self.overlay = Some(Overlay::ModelSelector(selector));
            return;
        }
        if event.kind == "Failed" {
            let diagnostics = event.diagnostics();
            let mut selector = match self.overlay.take() {
                Some(Overlay::Compiler(progress)) => progress.selector,
                _ => return,
            };
            selector.diagnostics.extend(diagnostics);
            self.status = "Model compiler failed".to_string();
            self.overlay = Some(Overlay::ModelSelector(selector));
            return;
        }
        if let Some(Overlay::Compiler(progress)) = &mut self.overlay {
            progress.stage = compiler_stage_label(&event.kind).to_string();
            if let Some((current, total)) = event.progress() {
                progress.current = Some(current);
                progress.total = Some(total);
            }
            progress.current_item = event.current_item().map(ToOwned::to_owned);
            if event.kind == "SourceDiscovered" {
                progress.selector.discovery = SourceDiscovery::from_event(&event);
            }
            progress.events.push(event);
        }
    }

    fn handle_compiler_exit(&mut self, status: ExitStatus, terminal_event_received: bool) {
        if !terminal_event_received {
            self.fail_compiler_job(format!(
                "Compiler exited with {status} without a terminal structured event"
            ));
        }
    }

    fn fail_compiler_job(&mut self, error: String) {
        let mut selector = match self.overlay.take() {
            Some(Overlay::Compiler(progress)) => {
                let mut selector = progress.selector;
                selector.diagnostics.extend(progress.diagnostics);
                selector
            }
            Some(Overlay::ModelSelector(selector)) => selector,
            overlay => {
                self.overlay = overlay;
                return;
            }
        };
        selector.diagnostics.push(error);
        self.status = "Model compiler failed".to_string();
        self.overlay = Some(Overlay::ModelSelector(selector));
    }

    fn load_model(&mut self, path: PathBuf) {
        let loaded = RuntimeModelEditor::load(&path);
        self.terminal_reset_requested = true;
        match loaded {
            Ok(editor) => {
                let sequence = editor.layer_sequence();
                let selected_instance_id = editor
                    .instances()
                    .first()
                    .map(|instance| instance.instance_id.clone());
                self.sequence.set(format_layer_sequence(&sequence));
                self.last_valid_sequence = sequence;
                self.sequence_error = None;
                self.selected_instance_id = selected_instance_id;
                self.board_scroll = 0;
                self.status = format!(
                    "Loaded {} · {} pedals · draft not mounted",
                    editor.package_id(),
                    editor.instances().len()
                );
                self.editor = Some(editor);
                self.overlay = None;
                self.focus = FocusRegion::Board;
            }
            Err(error) => {
                let mut selector = match self.overlay.take() {
                    Some(Overlay::Compiler(progress)) => progress.selector,
                    Some(Overlay::ModelSelector(selector)) => selector,
                    _ => ModelSelectorState::new(path.clone()),
                };
                selector.set_path(path);
                selector.diagnostics.push(error.to_string());
                self.status = "Compiled package could not be loaded".to_string();
                self.overlay = Some(Overlay::ModelSelector(selector));
            }
        }
    }

    fn apply_sequence_text(&mut self) {
        let Some(editor) = &mut self.editor else {
            return;
        };
        let available = editor
            .source_pedals()
            .iter()
            .map(|pedal| pedal.layer_index)
            .collect::<BTreeSet<_>>();
        match parse_layer_sequence(self.sequence.text(), &available) {
            Ok(sequence) => match editor.replace_layer_sequence(&sequence) {
                Ok(()) => {
                    self.last_valid_sequence = sequence;
                    self.sequence_error = None;
                    self.ensure_selection_exists();
                    self.status = "Board draft updated · not mounted".to_string();
                }
                Err(error) => {
                    self.sequence_error = Some(SequenceParseError {
                        message: error.to_string(),
                        byte: self.sequence.byte_cursor(),
                        column: self.sequence.cursor() + 1,
                    });
                }
            },
            Err(error) => self.sequence_error = Some(error),
        }
    }

    fn select_relative(&mut self, delta: isize) {
        let instances = self.instances();
        if instances.is_empty() {
            return;
        }
        let current = self
            .selected_instance_id
            .as_ref()
            .and_then(|selected| {
                instances
                    .iter()
                    .position(|instance| &instance.instance_id == selected)
            })
            .unwrap_or(0);
        self.select_index(
            current
                .saturating_add_signed(delta)
                .min(instances.len() - 1),
        );
    }

    fn select_index(&mut self, index: usize) {
        if let Some(instance) = self.instances().get(index) {
            self.selected_instance_id = Some(instance.instance_id.clone());
        }
    }

    fn select_instance(&mut self, instance_id: &str) {
        if self
            .instances()
            .iter()
            .any(|instance| instance.instance_id == instance_id)
        {
            self.selected_instance_id = Some(instance_id.to_string());
            self.focus = FocusRegion::Board;
        }
    }

    fn open_selected_pedal(&mut self) {
        let Some(editor) = &self.editor else {
            return;
        };
        let Some(instance_id) = &self.selected_instance_id else {
            return;
        };
        let instances = editor.instances();
        let Some(instance) = instances
            .iter()
            .find(|instance| &instance.instance_id == instance_id)
            .cloned()
        else {
            return;
        };
        let Some(source) = editor.source_pedal_for_instance(instance_id).cloned() else {
            return;
        };
        let devices = editor
            .available_devices()
            .iter()
            .filter(|device| {
                device.available && device.can_host_runtime_pedals_on_physical_device != Some(false)
            })
            .map(|device| {
                let name = device.device_name.as_deref().unwrap_or("unnamed device");
                let memory = device
                    .memory_heaps
                    .as_ref()
                    .and_then(|heaps| {
                        heaps
                            .iter()
                            .filter(|heap| heap.device_local)
                            .map(|heap| heap.size_bytes)
                            .max()
                    })
                    .map(|bytes| format!(" · {:.1} GiB", bytes as f64 / 1_073_741_824.0))
                    .unwrap_or_default();
                (
                    device.device_id.clone(),
                    format!("{} · {}{memory}", device.device_id, name),
                )
            })
            .collect::<Vec<_>>();
        let device_ids = devices.iter().map(|(id, _)| id.clone()).collect::<Vec<_>>();
        let device_labels = devices
            .iter()
            .map(|(_, label)| label.clone())
            .collect::<Vec<_>>();
        let device_index = device_ids
            .iter()
            .position(|device| device == &instance.device_id)
            .unwrap_or(0);
        let policy_targets = instances
            .iter()
            .filter(|candidate| candidate.instance_id != instance.instance_id)
            .map(|candidate| candidate.instance_id.clone())
            .collect::<Vec<_>>();
        let (policy, target) = match &instance.state_policy {
            StreamCircuitPedalInstanceStatePolicy::Fresh => (PedalPolicyKind::Independent, None),
            StreamCircuitPedalInstanceStatePolicy::CloneFrom { instance_id } => {
                (PedalPolicyKind::Clone, Some(instance_id))
            }
            StreamCircuitPedalInstanceStatePolicy::ShareWith { instance_id } => {
                (PedalPolicyKind::Share, Some(instance_id))
            }
        };
        let policy_target_index = target
            .and_then(|target| {
                policy_targets
                    .iter()
                    .position(|candidate| candidate == target)
            })
            .unwrap_or(0);
        self.overlay = Some(Overlay::Pedal(PedalModalState {
            instance_id: instance.instance_id,
            source,
            layer_index: instance.layer_index,
            occurrence: instance.occurrence,
            device_ids,
            device_labels,
            device_index,
            enabled: instance.enabled,
            policy,
            policy_targets,
            policy_target_index,
            focus_row: 0,
            error: None,
        }));
    }

    fn apply_pedal_modal(&mut self) {
        let Some(Overlay::Pedal(modal)) = &self.overlay else {
            return;
        };
        let modal = modal.clone();
        let Some(editor) = &self.editor else {
            return;
        };
        let Some(device_id) = modal.device_ids.get(modal.device_index) else {
            if let Some(Overlay::Pedal(modal)) = &mut self.overlay {
                modal.error = Some("No compatible runtime device is available".to_string());
            }
            return;
        };
        if modal.policy != PedalPolicyKind::Independent && modal.policy_targets.is_empty() {
            if let Some(Overlay::Pedal(modal)) = &mut self.overlay {
                modal.error = Some("This state policy needs another pedal instance".to_string());
            }
            return;
        }
        let mut candidate = editor.clone();
        let result = candidate
            .set_instance_device(&modal.instance_id, device_id)
            .and_then(|_| candidate.set_instance_enabled(&modal.instance_id, modal.enabled))
            .and_then(|_| {
                candidate.set_instance_state_policy(&modal.instance_id, modal.state_policy())
            });
        match result {
            Ok(()) => {
                self.editor = Some(candidate);
                self.overlay = None;
                self.status = format!("Updated {} · draft not mounted", modal.instance_id);
            }
            Err(error) => {
                if let Some(Overlay::Pedal(modal)) = &mut self.overlay {
                    modal.error = Some(error.to_string());
                }
            }
        }
    }

    fn duplicate_selected(&mut self) {
        let Some(index) = self.selected_index() else {
            return;
        };
        let mut sequence = self.last_valid_sequence.clone();
        if let Some(layer) = sequence.get(index).copied() {
            sequence.insert(index + 1, layer);
            self.replace_board_from_visual(sequence, Some(index + 1));
        }
    }

    fn remove_selected(&mut self) {
        let Some(index) = self.selected_index() else {
            return;
        };
        if self.last_valid_sequence.len() <= 1 {
            self.status = "A pedalboard must contain at least one pedal".to_string();
            return;
        }
        let mut sequence = self.last_valid_sequence.clone();
        sequence.remove(index);
        self.replace_board_from_visual(sequence, Some(index.saturating_sub(1)));
    }

    fn move_selected(&mut self, delta: i32) {
        let Some(index) = self.selected_index() else {
            return;
        };
        let target = index.saturating_add_signed(delta as isize);
        if target >= self.last_valid_sequence.len() || target == index {
            return;
        }
        let selected_id = self.selected_instance_id.clone();
        let mut sequence = self.last_valid_sequence.clone();
        sequence.swap(index, target);
        self.replace_board_from_visual(sequence, None);
        if let Some(selected_id) = selected_id {
            self.select_instance(&selected_id);
        }
    }

    fn replace_board_from_visual(&mut self, sequence: Vec<usize>, select_index: Option<usize>) {
        let Some(editor) = &mut self.editor else {
            return;
        };
        match editor.replace_layer_sequence(&sequence) {
            Ok(()) => {
                self.sequence.set(format_layer_sequence(&sequence));
                self.last_valid_sequence = sequence;
                self.sequence_error = None;
                if let Some(index) = select_index {
                    self.select_index(index);
                } else {
                    self.ensure_selection_exists();
                }
                self.status = "Board draft updated · not mounted".to_string();
            }
            Err(error) => self.status = error.to_string(),
        }
    }

    fn ensure_selection_exists(&mut self) {
        let instances = self.instances();
        if self.selected_instance_id.as_ref().is_none_or(|selected| {
            !instances
                .iter()
                .any(|instance| &instance.instance_id == selected)
        }) {
            self.selected_instance_id = instances
                .first()
                .map(|instance| instance.instance_id.clone());
        }
    }

    fn selected_index(&self) -> Option<usize> {
        let selected = self.selected_instance_id.as_ref()?;
        self.instances()
            .iter()
            .position(|instance| &instance.instance_id == selected)
    }

    fn instance_count(&self) -> usize {
        self.editor
            .as_ref()
            .map(|editor| editor.instances().len())
            .unwrap_or(0)
    }

    pub(crate) fn instances(&self) -> Vec<RuntimeEditorInstance> {
        self.editor
            .as_ref()
            .map(RuntimeModelEditor::instances)
            .unwrap_or_default()
    }
}

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

fn change_pedal_modal(modal: &mut PedalModalState, delta: i32) {
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
                PedalPolicyKind::Independent => 0,
                PedalPolicyKind::Clone => 1,
                PedalPolicyKind::Share => 2,
            };
            modal.policy = match cycle_index(current, choices, delta) {
                0 => PedalPolicyKind::Independent,
                1 => PedalPolicyKind::Clone,
                _ => PedalPolicyKind::Share,
            };
        }
        3 if !modal.policy_targets.is_empty() => {
            modal.policy_target_index =
                cycle_index(modal.policy_target_index, modal.policy_targets.len(), delta);
        }
        _ => {}
    }
}

fn cycle_index(current: usize, len: usize, delta: i32) -> usize {
    if len == 0 {
        return 0;
    }
    (current as i32 + delta).rem_euclid(len as i32) as usize
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
        "PedalTranspiled" => "Transpiling source pedals",
        "PedalLoweringStarted" => "Lowering pedal circuits",
        "ArtifactWritingStarted" => "Writing package artifacts",
        "TensorPackagingStarted" => "Packaging tensors",
        "ShaderCompilationStarted" => "Compiling GPU circuits",
        "PackageValidationStarted" => "Validating compiled package",
        _ => kind,
    }
}

fn source_discovery_from_value(raw: Value) -> Option<SourceDiscovery> {
    let event = CompilerEvent {
        schema: "llmoop.compiler_event.v1".to_string(),
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
    fn browser_lists_directories_before_files() {
        let root = env::temp_dir().join(format!("llmoop-tui-browser-{}", std::process::id()));
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
    fn loaded_board_visual_actions_keep_one_authoritative_numeric_sequence() {
        let Some(package) = env::var_os("LLMOOP_TEST_PACKAGE_DIR") else {
            return;
        };
        let mut app = App::new();
        app.load_compiled_model(package);
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
                    .source_pedals()
                    .iter()
                    .map(|pedal| pedal.layer_index)
                    .collect()
            )
            .unwrap(),
            original
        );
    }
}
