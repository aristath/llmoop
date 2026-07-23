impl App {
    fn dispatch_overlay(&mut self, action: AppAction) {
        match self.overlay.as_ref() {
            Some(Overlay::ModelSelector(_)) => self.dispatch_model_selector(action),
            Some(Overlay::Compiler(_)) => self.dispatch_compiler(action),
            Some(Overlay::Pedal(_)) => self.dispatch_pedal_modal(action),
            Some(Overlay::Help) => match action {
                AppAction::Quit => self.should_quit = true,
                AppAction::CloseOverlay => self.overlay = self.help_return_overlay.take(),
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
            AppAction::ModalPrevious if selector.focus == ModelSelectorFocus::Action => {
                selector.diagnostic_scroll = selector.diagnostic_scroll.saturating_sub(1);
            }
            AppAction::ModalNext if selector.focus == ModelSelectorFocus::Browser => {
                selector.selected_entry =
                    (selector.selected_entry + 1).min(selector.entries.len().saturating_sub(1));
            }
            AppAction::ModalNext if selector.focus == ModelSelectorFocus::Action => {
                selector.diagnostic_scroll = (selector.diagnostic_scroll + 1)
                    .min(selector.diagnostics.len().saturating_sub(1));
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
                } else if row == modal.apply_row() {
                    self.apply_pedal_modal();
                } else if row == modal.cancel_row() {
                    self.overlay = None;
                } else if modal
                    .property_index()
                    .and_then(|index| modal.properties.get(index))
                    .is_some_and(|property| !property.accepts_text())
                {
                    change_pedal_modal(modal, 1);
                }
            }
            AppAction::ActivateModal => {
                if modal.focus_row == 1 {
                    modal.enabled = !modal.enabled;
                } else if matches!(modal.focus_row, 0 | 2 | 3) {
                    change_pedal_modal(modal, 1);
                } else if modal.focus_row == modal.apply_row() {
                    self.apply_pedal_modal();
                } else if modal.focus_row == modal.cancel_row() {
                    self.overlay = None;
                } else if modal
                    .property_index()
                    .and_then(|index| modal.properties.get(index))
                    .is_some_and(|property| !property.accepts_text())
                {
                    change_pedal_modal(modal, 1);
                }
            }
            AppAction::InsertText(value) => {
                if let Some(property) = focused_property_mut(modal)
                    && property.accepts_text()
                {
                    property.buffer.insert(&value);
                    property.reparse_buffer();
                }
            }
            AppAction::Backspace => {
                if let Some(property) = focused_property_mut(modal)
                    && property.accepts_text()
                {
                    property.buffer.backspace();
                    property.reparse_buffer();
                }
            }
            AppAction::DeleteForward => {
                if let Some(property) = focused_property_mut(modal)
                    && property.accepts_text()
                {
                    property.buffer.delete();
                    property.reparse_buffer();
                }
            }
            AppAction::MoveTextCursor { motion, selecting } => {
                if let Some(property) = focused_property_mut(modal)
                    && property.accepts_text()
                {
                    move_text_cursor(&mut property.buffer, motion, selecting);
                }
            }
            AppAction::SelectAllText => {
                if let Some(property) = focused_property_mut(modal)
                    && property.accepts_text()
                {
                    property.buffer.select_all();
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
            Ok(editor) => self.install_editor(editor),
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

    pub(crate) fn install_editor(&mut self, editor: RuntimeModelEditor) {
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

}
