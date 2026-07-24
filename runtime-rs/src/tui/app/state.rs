use std::collections::BTreeSet;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitStatus;

use ratatui::layout::Rect;
use serde_json::Value;

use crate::{
    RuntimeEditorControlKind, RuntimeEditorControlSchema, RuntimeEditorInstance,
    RuntimeEditorSourceComponent, RuntimeModelEditor, RuntimeModelPathKind,
    StreamCircuitNodeInstanceStatePolicy, classify_runtime_model_path,
    validate_runtime_editor_control_value,
};

use super::compiler::{
    CompilerEvent, CompilerJob, CompilerJobKind, CompilerLaunch, CompilerMessage,
};
use super::sequence::{SequenceParseError, TextBuffer, parse_layer_sequence};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FocusRegion {
    Sequence,
    Graph,
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
    SelectPreviousNode,
    SelectNextNode,
    SelectFirstNode,
    SelectLastNode,
    OpenSelectedNode,
    SelectNode(String),
    OpenNode(String),
    PanGraph(i16),
    DuplicateSelected,
    RemoveSelected,
    MoveSelected(i32),
    ActivateModal,
    ModalPrevious,
    ModalNext,
    ModalChange(i32),
    ToggleModuleAnatomy,
    ScrollModuleAnatomy(i16),
    ModalClickRow(usize),
    ModelBrowserSelect(usize),
    ModelBrowserOpen(usize),
    ModelAction,
    ApplyNode,
    CancelCompiler,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum HitTarget {
    OpenModel,
    Sequence,
    Node(String),
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
pub(crate) enum NodePolicyKind {
    Independent,
    Clone,
    Share,
}

#[derive(Clone, Debug)]
pub(crate) struct NodeModalState {
    pub instance_id: String,
    pub source: RuntimeEditorSourceComponent,
    pub occurrence: usize,
    pub device_ids: Vec<String>,
    pub device_labels: Vec<String>,
    pub device_index: usize,
    pub original_device_id: String,
    pub enabled: bool,
    pub policy: NodePolicyKind,
    pub policy_targets: Vec<String>,
    pub policy_target_index: usize,
    pub properties: Vec<NodePropertyDraft>,
    pub anatomy_expanded: bool,
    pub anatomy_scroll: u16,
    pub focus_row: usize,
    pub error: Option<String>,
}

impl NodeModalState {
    fn row_count(&self) -> usize {
        6 + self.properties.len()
    }

    pub(crate) fn property_index(&self) -> Option<usize> {
        let index = self.focus_row.checked_sub(4)?;
        (index < self.properties.len()).then_some(index)
    }

    pub(crate) fn apply_row(&self) -> usize {
        4 + self.properties.len()
    }

    pub(crate) fn cancel_row(&self) -> usize {
        self.apply_row() + 1
    }

    fn state_policy(&self) -> StreamCircuitNodeInstanceStatePolicy {
        match self.policy {
            NodePolicyKind::Independent => StreamCircuitNodeInstanceStatePolicy::Fresh,
            NodePolicyKind::Clone => StreamCircuitNodeInstanceStatePolicy::CloneFrom {
                instance_id: self
                    .policy_targets
                    .get(self.policy_target_index)
                    .cloned()
                    .unwrap_or_default(),
            },
            NodePolicyKind::Share => StreamCircuitNodeInstanceStatePolicy::ShareWith {
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
pub(crate) struct NodePropertyDraft {
    pub schema: RuntimeEditorControlSchema,
    pub original_value: Value,
    pub value: Value,
    pub buffer: TextBuffer,
    pub error: Option<String>,
}

impl NodePropertyDraft {
    pub(crate) fn new(schema: RuntimeEditorControlSchema, value: Value) -> Self {
        let buffer = TextBuffer::new(control_value_text(&value));
        let mut property = Self {
            schema,
            original_value: value.clone(),
            value,
            buffer,
            error: None,
        };
        property.revalidate();
        property
    }

    pub fn editable(&self) -> bool {
        self.schema.editable_at_runtime
            && self.schema.scope == "instance"
            && !matches!(
                self.schema.kind,
                RuntimeEditorControlKind::ReadOnly | RuntimeEditorControlKind::Unsupported { .. }
            )
    }

    pub fn accepts_text(&self) -> bool {
        self.editable()
            && matches!(
                self.schema.kind,
                RuntimeEditorControlKind::Integer
                    | RuntimeEditorControlKind::Number
                    | RuntimeEditorControlKind::Text
            )
    }

    pub fn changed(&self) -> bool {
        self.value != self.original_value
    }

    fn revalidate(&mut self) {
        self.error = if self.editable() {
            validate_runtime_editor_control_value(&self.schema, &self.value)
                .map_err(|error| error.to_string())
                .err()
        } else {
            None
        };
    }

    fn reparse_buffer(&mut self) {
        let parsed = match self.schema.kind {
            RuntimeEditorControlKind::Integer => self
                .buffer
                .text()
                .parse::<i64>()
                .map(Value::from)
                .map_err(|_| "Expected a whole number".to_string()),
            RuntimeEditorControlKind::Number => self
                .buffer
                .text()
                .parse::<f64>()
                .map_err(|_| "Expected a number".to_string())
                .and_then(|number| {
                    serde_json::Number::from_f64(number)
                        .map(Value::Number)
                        .ok_or_else(|| "Number must be finite".to_string())
                }),
            RuntimeEditorControlKind::Text => Ok(Value::String(self.buffer.text().to_string())),
            _ => Ok(self.value.clone()),
        };
        match parsed {
            Ok(value) => {
                self.value = value;
                self.revalidate();
            }
            Err(error) => self.error = Some(error),
        }
    }

    fn change(&mut self, delta: i32) {
        if !self.editable() {
            return;
        }
        match &self.schema.kind {
            RuntimeEditorControlKind::Boolean => {
                self.value = Value::Bool(!self.value.as_bool().unwrap_or(false));
                self.buffer.set(control_value_text(&self.value));
                self.revalidate();
            }
            RuntimeEditorControlKind::Enumeration { choices } if !choices.is_empty() => {
                let current = choices
                    .iter()
                    .position(|choice| choice.value == self.value)
                    .unwrap_or(0);
                let next = cycle_index(current, choices.len(), delta);
                self.value = choices[next].value.clone();
                self.buffer.set(control_value_text(&self.value));
                self.revalidate();
            }
            RuntimeEditorControlKind::Integer => {
                let step = self.schema.step.unwrap_or(1.0).round() as i64;
                let current = self.value.as_i64().unwrap_or(0);
                let mut next = current.saturating_add(step.saturating_mul(delta as i64));
                if let Some(minimum) = self.schema.minimum {
                    next = next.max(minimum.ceil() as i64);
                }
                if let Some(maximum) = self.schema.maximum {
                    next = next.min(maximum.floor() as i64);
                }
                self.value = Value::from(next);
                self.buffer.set(next.to_string());
                self.revalidate();
            }
            RuntimeEditorControlKind::Number => {
                let step = self.schema.step.unwrap_or(0.1);
                let current = self.value.as_f64().unwrap_or(0.0);
                let mut next = current + step * delta as f64;
                if let Some(minimum) = self.schema.minimum {
                    next = next.max(minimum);
                }
                if let Some(maximum) = self.schema.maximum {
                    next = next.min(maximum);
                }
                if let Some(number) = serde_json::Number::from_f64(next) {
                    self.value = Value::Number(number);
                    self.buffer.set(next.to_string());
                    self.revalidate();
                }
            }
            _ => {}
        }
    }

    pub fn display_value(&self) -> String {
        if let RuntimeEditorControlKind::Enumeration { choices } = &self.schema.kind
            && let Some(choice) = choices.iter().find(|choice| choice.value == self.value)
        {
            return choice.label.clone();
        }
        control_value_text(&self.value)
    }
}

#[derive(Clone, Debug)]
pub(crate) enum Overlay {
    ModelSelector(ModelSelectorState),
    Compiler(CompilerProgressState),
    Node(NodeModalState),
    Help,
}

pub struct App {
    pub(crate) editor: Option<RuntimeModelEditor>,
    pub(crate) sequence: TextBuffer,
    pub(crate) sequence_error: Option<SequenceParseError>,
    pub(crate) last_valid_sequence: Vec<usize>,
    pub(crate) focus: FocusRegion,
    pub(crate) selected_instance_id: Option<String>,
    pub(crate) graph_scroll: usize,
    pub(crate) overlay: Option<Overlay>,
    help_return_overlay: Option<Overlay>,
    pub(crate) status: String,
    pub(crate) should_quit: bool,
    pub(crate) mouse_capture: bool,
    pub(crate) hit_map: HitMap,
    pub(crate) compiler_job: Option<CompilerJob>,
    compiler_launch: Result<CompilerLaunch, String>,
    terminal_reset_requested: bool,
}
