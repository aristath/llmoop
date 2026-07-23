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
            help_return_overlay: None,
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

    pub fn modal_text_entry_active(&self) -> bool {
        let Some(Overlay::Pedal(modal)) = &self.overlay else {
            return false;
        };
        modal
            .property_index()
            .and_then(|index| modal.properties.get(index))
            .is_some_and(PedalPropertyDraft::accepts_text)
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
        if matches!(action, AppAction::ToggleMouseCapture) {
            self.mouse_capture = !self.mouse_capture;
            return;
        }
        if matches!(action, AppAction::ToggleHelp) {
            if matches!(self.overlay, Some(Overlay::Help)) {
                self.overlay = self.help_return_overlay.take();
            } else {
                self.help_return_overlay = self.overlay.take();
                self.overlay = Some(Overlay::Help);
            }
            return;
        }
        if self.overlay.is_some() {
            self.dispatch_overlay(action);
            return;
        }
        match action {
            AppAction::Quit => self.should_quit = true,
            AppAction::OpenModelSelector => self.open_model_selector(),
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

}
