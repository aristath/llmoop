use std::collections::{HashMap, VecDeque};

use crate::backend::{BackendError, DeviceBackend};
use crate::types::{
    ControlCommand, DeviceDispatchRun, DeviceDispatchTick, DeviceMemoryPlan, DeviceOutputEvent,
    DispatchStatus, ForkPolicy, ForkRequest, HostPortsManifest, InputSignal,
    InstalledProcessorManifest, MemoryRegion, MemoryRegionKind, MemorySharing,
    PermanentCircuitManifest, PromptInjection, PublicOutputSignal, RandomPolicy, StreamId,
    StreamTemplate, TokenId,
};

#[derive(Clone, Debug, Default)]
struct ContractStream {
    pending_external: VecDeque<InputSignal>,
    pending_feedback: VecDeque<InputSignal>,
    remaining_outputs: u32,
    input_counter: u64,
    public_counter: u64,
    feedback_counter: u64,
}

impl ContractStream {
    fn has_work(&self) -> bool {
        !self.pending_external.is_empty() || !self.pending_feedback.is_empty()
    }

    fn next_input(&mut self) -> Option<InputSignal> {
        self.pending_external
            .pop_front()
            .or_else(|| self.pending_feedback.pop_front())
    }

    fn next_contract_token(input_token: TokenId) -> TokenId {
        input_token + 1
    }
}

#[derive(Clone, Debug)]
pub struct ContractDeviceBackend {
    device_id: String,
    streams: HashMap<StreamId, ContractStream>,
    active_queue: VecDeque<StreamId>,
    output_queue: Vec<DeviceOutputEvent>,
    dispatch_tick: u64,
}

impl ContractDeviceBackend {
    pub const BACKEND_ID: &'static str = "contract_device_backend";

    pub fn new(device_id: impl Into<String>) -> Self {
        Self {
            device_id: device_id.into(),
            streams: HashMap::new(),
            active_queue: VecDeque::new(),
            output_queue: Vec::new(),
            dispatch_tick: 0,
        }
    }

    fn stream_mut(&mut self, stream_id: &str) -> Result<&mut ContractStream, BackendError> {
        self.streams
            .get_mut(stream_id)
            .ok_or_else(|| BackendError::UnknownStream(stream_id.to_string()))
    }

    fn schedule(&mut self, stream_id: &str) {
        if !self.active_queue.iter().any(|active| active == stream_id) {
            self.active_queue.push_back(stream_id.to_string());
        }
    }
}

impl DeviceBackend for ContractDeviceBackend {
    type Error = BackendError;

    fn backend_id(&self) -> &str {
        Self::BACKEND_ID
    }

    fn device_id(&self) -> &str {
        &self.device_id
    }

    fn has_stream(&self, stream_id: &str) -> bool {
        self.streams.contains_key(stream_id)
    }

    fn create_stream(&mut self, stream_id: &str) -> Result<(), Self::Error> {
        if self.has_stream(stream_id) {
            return Err(BackendError::DuplicateStream(stream_id.to_string()));
        }
        self.streams
            .insert(stream_id.to_string(), ContractStream::default());
        Ok(())
    }

    fn inject_prompt(&mut self, injection: PromptInjection) -> Result<(), Self::Error> {
        let stream_id = injection.stream_id.clone();
        let stream = self.stream_mut(&stream_id)?;
        for token_id in injection.prompt_ids {
            let signal = InputSignal::external(
                format!("input_{}", stream.input_counter),
                token_id,
                injection.origin.clone(),
            );
            stream.input_counter += 1;
            stream.pending_external.push_back(signal);
        }
        stream.remaining_outputs = stream
            .remaining_outputs
            .saturating_add(injection.max_new_tokens);
        self.schedule(&stream_id);
        Ok(())
    }

    fn inject_token(&mut self, stream_id: &str, signal: InputSignal) -> Result<(), Self::Error> {
        self.stream_mut(stream_id)?
            .pending_external
            .push_back(signal);
        self.schedule(stream_id);
        Ok(())
    }

    fn control(&mut self, stream_id: &str, command: ControlCommand) -> Result<(), Self::Error> {
        let stream = self.stream_mut(stream_id)?;
        match command {
            ControlCommand::Continue {
                additional_public_outputs,
                ..
            } => {
                stream.remaining_outputs = stream
                    .remaining_outputs
                    .saturating_add(additional_public_outputs);
            }
            ControlCommand::Interrupt { .. } => {
                stream.pending_feedback.clear();
                stream.remaining_outputs = 0;
            }
            ControlCommand::StopAfterCurrent { .. } => {
                if let Some(current) = stream.pending_feedback.pop_front() {
                    stream.pending_feedback.clear();
                    stream.pending_feedback.push_back(current);
                }
                stream.remaining_outputs = 0;
            }
            ControlCommand::ResetState { .. } => {
                *stream = ContractStream::default();
            }
            ControlCommand::ReseedRandom { .. } => {}
        }
        if stream.has_work() {
            self.schedule(stream_id);
        }
        Ok(())
    }

    fn fork_stream(&mut self, request: ForkRequest) -> Result<(), Self::Error> {
        if self.has_stream(&request.child_stream_id) {
            return Err(BackendError::DuplicateStream(request.child_stream_id));
        }
        let child = match request.state_policy {
            ForkPolicy::Clone => self
                .streams
                .get(&request.parent_stream_id)
                .ok_or_else(|| BackendError::UnknownStream(request.parent_stream_id.clone()))?
                .clone(),
            ForkPolicy::Fresh => ContractStream::default(),
        };
        let child_has_work = child.has_work();
        self.streams.insert(request.child_stream_id.clone(), child);
        if request.random_policy == RandomPolicy::Fresh {
            // Contract backend has no random state yet; the policy is still part of the boundary.
        }
        if child_has_work {
            self.schedule(&request.child_stream_id);
        }
        Ok(())
    }

    fn dispatch(&mut self, max_ticks: u32) -> Result<DeviceDispatchRun, Self::Error> {
        let mut ticks = Vec::new();
        let mut outputs = Vec::new();

        while !self.active_queue.is_empty() && ticks.len() < max_ticks as usize {
            let stream_id = self
                .active_queue
                .pop_front()
                .expect("queue checked as non-empty");
            let dispatch_tick = self.dispatch_tick;
            let device_id = self.device_id.clone();

            let mut output = None;
            let mut reschedule = false;
            let Some(input) = ({
                let stream = self.stream_mut(&stream_id)?;
                let input = stream.next_input();
                if let Some(input) = &input {
                    if stream.remaining_outputs > 0 && stream.pending_external.is_empty() {
                        let output_token = ContractStream::next_contract_token(input.token_id);
                        let public = PublicOutputSignal::token(
                            format!("public_{}", stream.public_counter),
                            output_token,
                            dispatch_tick,
                        );
                        stream.public_counter += 1;
                        stream.remaining_outputs -= 1;
                        let feedback = InputSignal::feedback(
                            format!("feedback_{}", stream.feedback_counter),
                            output_token,
                        );
                        stream.feedback_counter += 1;
                        stream.pending_feedback.push_back(feedback);
                        output = Some(DeviceOutputEvent {
                            device_id: device_id.clone(),
                            stream_id: stream_id.clone(),
                            output: public,
                            dispatch_tick,
                        });
                    }
                    reschedule = stream.has_work();
                }
                input
            }) else {
                continue;
            };

            self.dispatch_tick += 1;
            ticks.push(DeviceDispatchTick {
                device_id,
                dispatch_tick,
                stream_id: stream_id.clone(),
                input,
                status: "processed".to_string(),
            });

            if let Some(event) = output {
                self.output_queue.push(event.clone());
                outputs.push(event);
            }
            if reschedule {
                self.schedule(&stream_id);
            }
        }

        let status = if self.active_queue.is_empty() {
            DispatchStatus::Idle
        } else {
            DispatchStatus::BudgetExhausted
        };

        Ok(DeviceDispatchRun {
            device_id: self.device_id.clone(),
            ticks,
            outputs,
            status,
            active_streams: self.active_queue.iter().cloned().collect(),
        })
    }

    fn drain_outputs(&mut self) -> Result<Vec<DeviceOutputEvent>, Self::Error> {
        Ok(std::mem::take(&mut self.output_queue))
    }

    fn describe(&self) -> InstalledProcessorManifest {
        InstalledProcessorManifest {
            install_id: "contract_installed_processor".to_string(),
            backend: self.backend_id().to_string(),
            permanent_circuit: PermanentCircuitManifest {
                pedal_count: 0,
                input_signal: "token".to_string(),
                output_signal: "token".to_string(),
            },
            host_ports: HostPortsManifest {
                inputs: vec![
                    "external_input".to_string(),
                    "control".to_string(),
                    "random_input".to_string(),
                ],
                outputs: vec!["public_output".to_string(), "events".to_string()],
                private_feedback: "device_owned_insert_loop".to_string(),
            },
            stream_template: StreamTemplate {
                id: "stream_template".to_string(),
                state_allocations: Vec::new(),
            },
            memory_plan: DeviceMemoryPlan {
                regions: vec![
                    MemoryRegion {
                        id: "programs".to_string(),
                        kind: MemoryRegionKind::SpirvProgram,
                        sharing: MemorySharing::SharedByAllStreams,
                        bytes: None,
                    },
                    MemoryRegion {
                        id: "stream_transient_state".to_string(),
                        kind: MemoryRegionKind::StreamTransientState,
                        sharing: MemorySharing::PerStream,
                        bytes: None,
                    },
                    MemoryRegion {
                        id: "input_queue".to_string(),
                        kind: MemoryRegionKind::InputQueue,
                        sharing: MemorySharing::HostVisibleQueue,
                        bytes: None,
                    },
                    MemoryRegion {
                        id: "output_queue".to_string(),
                        kind: MemoryRegionKind::OutputQueue,
                        sharing: MemorySharing::HostVisibleQueue,
                        bytes: None,
                    },
                ],
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contract_backend_owns_feedback_loop() {
        let mut backend = ContractDeviceBackend::new("device_0");
        backend.create_stream("s0").unwrap();
        backend
            .inject_prompt(PromptInjection::new("s0", vec![1], 2))
            .unwrap();

        let run = backend.dispatch(16).unwrap();

        assert_eq!(run.status, DispatchStatus::Idle);
        assert_eq!(run.ticks.len(), 3);
        assert_eq!(run.outputs.len(), 2);
        assert_eq!(run.outputs[0].output.id, "public_0");
        assert_eq!(run.outputs[1].output.id, "public_1");
        assert_eq!(run.outputs[0].output.token_id, 2);
        assert_eq!(run.outputs[1].output.token_id, 3);
        assert!(run.active_streams.is_empty());
        assert_eq!(backend.drain_outputs().unwrap().len(), 2);
    }

    #[test]
    fn contract_backend_round_robins_streams() {
        let mut backend = ContractDeviceBackend::new("device_0");
        backend.create_stream("a").unwrap();
        backend.create_stream("b").unwrap();
        backend
            .inject_prompt(PromptInjection::new("a", vec![1], 1))
            .unwrap();
        backend
            .inject_prompt(PromptInjection::new("b", vec![10], 1))
            .unwrap();

        let run = backend.dispatch(16).unwrap();
        let order: Vec<_> = run
            .ticks
            .iter()
            .map(|tick| tick.stream_id.as_str())
            .collect();

        assert_eq!(order, vec!["a", "b", "a", "b"]);
        assert_eq!(run.outputs.len(), 2);
        assert_eq!(run.outputs[0].stream_id, "a");
        assert_eq!(run.outputs[1].stream_id, "b");
    }

    #[test]
    fn manifest_names_the_future_vulkan_boundary() {
        let backend = ContractDeviceBackend::new("device_0");
        let manifest = backend.describe();

        assert_eq!(
            manifest.host_ports.private_feedback,
            "device_owned_insert_loop"
        );
        assert!(
            manifest
                .host_ports
                .inputs
                .contains(&"external_input".to_string())
        );
        assert!(
            manifest
                .host_ports
                .outputs
                .contains(&"public_output".to_string())
        );
        assert!(
            manifest
                .memory_plan
                .regions
                .iter()
                .any(|region| region.kind == MemoryRegionKind::SpirvProgram)
        );
        assert!(
            manifest
                .memory_plan
                .regions
                .iter()
                .any(|region| region.kind == MemoryRegionKind::StreamTransientState)
        );
    }
}
