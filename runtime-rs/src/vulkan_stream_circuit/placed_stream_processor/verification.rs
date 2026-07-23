impl VulkanResidentInProcessPlacedStreamProcessor {
    pub fn model_package(&self) -> &VulkanResidentInProcessPlacedModelPackage {
        &self.model
    }

    pub fn speculative_decoder_count(&self) -> usize {
        self.speculative_decoders.len()
    }

    fn synchronize_speculative_decoders_after_target_tick(
        &self,
        devices: &BTreeMap<String, Rc<VulkanComputeDevice>>,
        input_token_id: u32,
        stream_tick: u64,
    ) -> Result<(), VulkanResidentInProcessPlacedRuntimeError> {
        if self.speculative_decoders.is_empty() {
            return Ok(());
        }
        for decoder in &self.speculative_decoders {
            let draft_device = devices.get(&decoder.device_id).ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                    device_id: decoder.device_id.clone(),
                }
            })?;
            decoder.run_state_step(
                draft_device.as_ref(),
                input_token_id,
                stream_tick,
                VulkanDraftHiddenSource::PendingTarget,
            )?;
            decoder.commit_target_hidden()?;
        }
        Ok(())
    }

    fn ensure_verification_state_transactions(
        &self,
        devices: &BTreeMap<String, Rc<VulkanComputeDevice>>,
        transaction_width: usize,
    ) -> Result<(), VulkanResidentInProcessPlacedRuntimeError> {
        let transactions_are_sufficient = self
            .verification_state_transactions
            .borrow()
            .as_ref()
            .is_some_and(|transactions| {
                transactions
                    .iter()
                    .all(|transaction| transaction.cycle_width >= transaction_width)
            });
        let batch_execution_is_sufficient = self
            .component_batch_execution
            .borrow()
            .as_ref()
            .is_some_and(|runner| runner.lane_capacity >= transaction_width);
        let input_embedding_is_sufficient = self
            .verification_input_embedding
            .borrow()
            .as_ref()
            .is_some_and(|runner| runner.batch_capacity >= transaction_width);
        if transactions_are_sufficient
            && batch_execution_is_sufficient
            && input_embedding_is_sufficient
        {
            return Ok(());
        }
        if !transactions_are_sufficient {
            let transactions = create_placed_state_transactions(
                &self.device_slices,
                transaction_width,
                &|device_id| {
                    devices
                        .get(device_id)
                        .map(|device| device.as_ref())
                        .ok_or_else(|| {
                            VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                                device_id: device_id.to_string(),
                            }
                        })
                },
            )?;
            *self.verification_state_transactions.borrow_mut() = Some(transactions);
            // Recorded batch sequences contain copies into the transaction's snapshot buffers.
            *self.verification_input_embedding.borrow_mut() = None;
            *self.batched_output_projection.borrow_mut() = None;
            *self.component_batch_execution.borrow_mut() = None;
        }
        if self
            .component_batch_execution
            .borrow()
            .as_ref()
            .is_none_or(|runner| runner.lane_capacity < transaction_width)
        {
            *self.verification_input_embedding.borrow_mut() = None;
            *self.batched_output_projection.borrow_mut() = None;
            let runner = VulkanResidentPlacedComponentBatchRunner::new(
                devices,
                &self.device_slices,
                transaction_width,
                VulkanComponentBatchExecutionMode::IndependentCandidates,
                &self.model.distributed_execution_plan,
                &self.model.distributed_parameter_buffers,
            )?;
            *self.component_batch_execution.borrow_mut() = Some(runner);
        }
        if self
            .verification_input_embedding
            .borrow()
            .as_ref()
            .is_none_or(|runner| runner.batch_capacity < transaction_width)
        {
            let first_device_index =
                *self
                    .linear_pipeline_device_indices()?
                    .first()
                    .ok_or_else(|| {
                        VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                            "placed verification pipeline is empty".to_string(),
                        ))
                    })?;
            let input_device = devices.get(&self.model.input_device_id).ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                    device_id: self.model.input_device_id.clone(),
                }
            })?;
            let embedding_weight = self
                .model
                .input_transducer_parameter_buffers
                .parameter_buffer(&self.model.input_transducer_spec.parameter_tensor)
                .ok_or_else(|| {
                    VulkanResidentInProcessPlacedRuntimeError::InputTransducer(
                        VulkanResidentInputEmbeddingTransducerRunnerError::MissingTransducerParameterBuffer {
                            tensor: self.model.input_transducer_spec.parameter_tensor.clone(),
                        },
                    )
                })?;
            let batch_execution = self.component_batch_execution.borrow();
            let input_signal = batch_execution
                .as_ref()
                .expect("verification component batch was initialized")
                .slice(first_device_index)?
                .signal_buffer(&VulkanComponentBatchSignalKey::ModelInput(
                    self.model.input_transducer_spec.output_signal_id.clone(),
                ))?;
            let input_embedding = VulkanResidentBatchedInputEmbeddingRunner::new(
                input_device,
                transaction_width,
                embedding_weight,
                &input_signal.buffer,
                &self.model.input_transducer_batch_spirv_words,
                &self.model.input_transducer_spec,
            )?;
            drop(batch_execution);
            *self.verification_input_embedding.borrow_mut() = Some(input_embedding);
        }
        Ok(())
    }

    fn capture_verification_baseline(
        &self,
    ) -> Result<(), VulkanResidentInProcessPlacedRuntimeError> {
        let transactions = self.verification_state_transactions.borrow();
        let transactions = transactions.as_ref().ok_or_else(|| {
            VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                "verification state transaction is not mounted".to_string(),
            ))
        })?;
        for (transaction, slice) in transactions.iter().zip(&self.device_slices) {
            transaction
                .capture_baseline(&slice.mounted.buffers)
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        }
        Ok(())
    }

    fn restore_verification_baseline(
        &self,
    ) -> Result<(), VulkanResidentInProcessPlacedRuntimeError> {
        let transactions = self.verification_state_transactions.borrow();
        let transactions = transactions.as_ref().ok_or_else(|| {
            VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                "verification state transaction is not mounted".to_string(),
            ))
        })?;
        for (transaction, slice) in transactions.iter().zip(&self.device_slices) {
            transaction
                .restore_baseline(&slice.mounted.buffers)
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        }
        Ok(())
    }

    fn commit_verification_prefix(
        &self,
        processed_tick_count: usize,
    ) -> Result<(), VulkanResidentInProcessPlacedRuntimeError> {
        let transactions = self.verification_state_transactions.borrow();
        let transactions = transactions.as_ref().ok_or_else(|| {
            VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                "verification state transaction is not mounted".to_string(),
            ))
        })?;
        for (transaction, slice) in transactions.iter().zip(&self.device_slices) {
            transaction
                .commit_prefix(&slice.mounted.buffers, processed_tick_count)
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        }
        Ok(())
    }

}
