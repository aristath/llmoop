pub struct VulkanResidentInProcessPlacedPromptSession {
    pub next_stream_tick: u64,
    pub completed_prompt_event_count: usize,
    pub generated_token_count: usize,
    pub output_token_count: usize,
    transport: VulkanInProcessPlacedCableTransport,
}

impl VulkanResidentInProcessPlacedPromptSession {
    pub fn new(start_stream_tick: u64) -> Self {
        Self {
            next_stream_tick: start_stream_tick,
            completed_prompt_event_count: 0,
            generated_token_count: 0,
            output_token_count: 0,
            transport: VulkanInProcessPlacedCableTransport::new(),
        }
    }

    pub fn transport_direct_cable_binding_count(&self) -> usize {
        self.transport.direct_cable_binding_count()
    }

    pub fn transport_stats(&self) -> VulkanPlacedCableTransportStats {
        self.transport.stats()
    }

    fn complete_prompt_event(
        &mut self,
        start_stream_tick: u64,
        run: VulkanResidentInProcessPlacedPromptEventRun,
    ) -> Result<
        VulkanResidentInProcessPlacedPromptSessionRun,
        VulkanResidentInProcessPlacedRuntimeError,
    > {
        let tick_count = run.tick_count;
        let tick_delta = u64::try_from(tick_count)
            .map_err(|_| VulkanResidentInProcessPlacedRuntimeError::StreamTickOverflow)?;
        let next_stream_tick = start_stream_tick
            .checked_add(tick_delta)
            .ok_or(VulkanResidentInProcessPlacedRuntimeError::StreamTickOverflow)?;
        let prompt_event_index = self.completed_prompt_event_count;

        self.next_stream_tick = next_stream_tick;
        self.completed_prompt_event_count = self.completed_prompt_event_count.saturating_add(1);
        self.generated_token_count = self
            .generated_token_count
            .saturating_add(run.generated_token_ids.len());
        self.output_token_count = self
            .output_token_count
            .saturating_add(run.output_token_ids.len());

        Ok(VulkanResidentInProcessPlacedPromptSessionRun {
            prompt_event_index,
            start_stream_tick,
            next_stream_tick,
            tick_count,
            run,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentInProcessPlacedPromptSessionRun {
    pub prompt_event_index: usize,
    pub start_stream_tick: u64,
    pub next_stream_tick: u64,
    pub tick_count: usize,
    pub run: VulkanResidentInProcessPlacedPromptEventRun,
}

