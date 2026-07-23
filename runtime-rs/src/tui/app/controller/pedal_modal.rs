impl App {
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
                let status = if !device.available {
                    " · UNAVAILABLE"
                } else if device.can_host_runtime_pedals_on_physical_device == Some(false) {
                    " · INCOMPATIBLE"
                } else {
                    ""
                };
                (
                    device.device_id.clone(),
                    format!("{} · {}{memory}{status}", device.device_id, name),
                )
            })
            .chain(
                (!editor
                    .available_devices()
                    .iter()
                    .any(|device| device.device_id == instance.device_id))
                .then(|| {
                    (
                        instance.device_id.clone(),
                        format!("{} · UNAVAILABLE", instance.device_id),
                    )
                }),
            )
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
        let properties = source
            .control_schemas
            .iter()
            .cloned()
            .map(|schema| {
                let value = editor
                    .effective_instance_control_value(instance_id, &schema.id)
                    .unwrap_or(Value::Null);
                PedalPropertyDraft::new(schema, value)
            })
            .collect();
        self.overlay = Some(Overlay::Pedal(PedalModalState {
            instance_id: instance.instance_id,
            source,
            occurrence: instance.occurrence,
            device_ids,
            device_labels,
            device_index,
            original_device_id: instance.device_id,
            enabled: instance.enabled,
            policy,
            policy_targets,
            policy_target_index,
            properties,
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
        if let Some(property) = modal
            .properties
            .iter()
            .find(|property| property.editable() && property.error.is_some())
        {
            if let Some(Overlay::Pedal(modal)) = &mut self.overlay {
                modal.error = Some(format!(
                    "{}: {}",
                    property.schema.name,
                    property.error.as_deref().unwrap_or("invalid value")
                ));
            }
            return;
        }
        let mut candidate = editor.clone();
        let mut result = if device_id == &modal.original_device_id {
            Ok(())
        } else {
            candidate.set_instance_device(&modal.instance_id, device_id)
        }
        .and_then(|_| candidate.set_instance_enabled(&modal.instance_id, modal.enabled))
        .and_then(|_| {
            candidate.set_instance_state_policy(&modal.instance_id, modal.state_policy())
        });
        if result.is_ok() {
            for property in modal
                .properties
                .iter()
                .filter(|property| property.editable() && property.changed())
            {
                if let Err(error) = candidate.set_instance_control_value(
                    &modal.instance_id,
                    &property.schema.id,
                    property.value.clone(),
                ) {
                    result = Err(error);
                    break;
                }
            }
        }
        match result {
            Ok(()) => {
                let lifecycle = modal
                    .properties
                    .iter()
                    .filter(|property| property.editable() && property.changed())
                    .flat_map(|property| {
                        [
                            property
                                .schema
                                .requires_state_reset
                                .then_some("state reset"),
                            property.schema.requires_remount.then_some("remount"),
                            property.schema.requires_recompile.then_some("recompile"),
                        ]
                    })
                    .flatten()
                    .collect::<BTreeSet<_>>();
                self.editor = Some(candidate);
                self.overlay = None;
                self.status = if lifecycle.is_empty() {
                    format!("Updated {} · draft not mounted", modal.instance_id)
                } else {
                    format!(
                        "Updated {} · requires {} · draft not mounted",
                        modal.instance_id,
                        lifecycle.into_iter().collect::<Vec<_>>().join(", ")
                    )
                };
            }
            Err(error) => {
                if let Some(Overlay::Pedal(modal)) = &mut self.overlay {
                    modal.error = Some(error.to_string());
                }
            }
        }
    }

}
