impl App {
    fn apply_sequence_text(&mut self) {
        let Some(editor) = &mut self.editor else {
            return;
        };
        let available = editor
            .source_pedals()
            .iter()
            .filter_map(|pedal| pedal.layer_index)
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

}
