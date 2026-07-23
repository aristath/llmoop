impl App {
    fn duplicate_selected(&mut self) {
        let Some(index) = self.selected_index() else {
            return;
        };
        let mut sequence = self.last_valid_sequence.clone();
        if let Some(layer) = sequence.get(index).copied() {
            sequence.insert(index + 1, layer);
            self.replace_graph_from_visual(sequence, Some(index + 1));
        }
    }

    fn remove_selected(&mut self) {
        let Some(index) = self.selected_index() else {
            return;
        };
        if self.last_valid_sequence.len() <= 1 {
            self.status = "An execution graph must contain at least one node".to_string();
            return;
        }
        let mut sequence = self.last_valid_sequence.clone();
        sequence.remove(index);
        self.replace_graph_from_visual(sequence, Some(index.saturating_sub(1)));
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
        self.replace_graph_from_visual(sequence, None);
        if let Some(selected_id) = selected_id {
            self.select_instance(&selected_id);
        }
    }

    fn replace_graph_from_visual(&mut self, sequence: Vec<usize>, select_index: Option<usize>) {
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
                self.status = "Graph draft updated · not mounted".to_string();
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
