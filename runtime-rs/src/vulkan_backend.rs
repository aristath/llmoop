use std::collections::{HashMap, VecDeque};
use std::error::Error;
use std::fmt::{Display, Formatter};

use crate::backend::{BackendError, DeviceBackend};
use crate::types::{
    ControlCommand, DeviceDispatchRun, DeviceDispatchTick, DeviceMemoryPlan, DeviceOutputEvent,
    DispatchStatus, ForkPolicy, ForkRequest, HostPortsManifest, InputSignal,
    InstalledProcessorManifest, MemoryRegion, MemoryRegionKind, MemorySharing,
    PermanentCircuitManifest, PromptInjection, PublicOutputSignal, RandomPolicy, StreamId,
    StreamTemplate, TokenId,
};
use crate::vulkan::VULKAN_SPIRV_BACKEND_ID;
use crate::vulkan_compute::{VulkanComputeDevice, VulkanError, VulkanU32ShaderPedal};
use crate::vulkan_pedalboard::VulkanU32Pedalboard;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VulkanBackendError {
    Backend(BackendError),
    Vulkan(VulkanError),
    InvalidToken(TokenId),
    EmptyPedalOutput,
}

impl Display for VulkanBackendError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Backend(error) => Display::fmt(error, f),
            Self::Vulkan(error) => Display::fmt(error, f),
            Self::InvalidToken(token) => write!(f, "token {token} cannot be represented as u32"),
            Self::EmptyPedalOutput => f.write_str("Vulkan pedalboard returned no output token"),
        }
    }
}

impl Error for VulkanBackendError {}

impl From<BackendError> for VulkanBackendError {
    fn from(error: BackendError) -> Self {
        Self::Backend(error)
    }
}

impl From<VulkanError> for VulkanBackendError {
    fn from(error: VulkanError) -> Self {
        Self::Vulkan(error)
    }
}

#[derive(Clone, Debug, Default)]
struct VulkanU32Stream {
    pending_external: VecDeque<InputSignal>,
    pending_feedback: VecDeque<InputSignal>,
    remaining_outputs: u32,
    input_counter: u64,
    public_counter: u64,
    feedback_counter: u64,
}

impl VulkanU32Stream {
    fn has_work(&self) -> bool {
        !self.pending_external.is_empty() || !self.pending_feedback.is_empty()
    }

    fn next_input(&mut self) -> Option<InputSignal> {
        self.pending_external
            .pop_front()
            .or_else(|| self.pending_feedback.pop_front())
    }
}

pub struct VulkanU32Backend {
    device_id: String,
    device: VulkanComputeDevice,
    pedalboard: VulkanU32Pedalboard,
    streams: HashMap<StreamId, VulkanU32Stream>,
    active_queue: VecDeque<StreamId>,
    output_queue: Vec<DeviceOutputEvent>,
    dispatch_tick: u64,
}

impl VulkanU32Backend {
    pub fn new(
        device_id: impl Into<String>,
        pedalboard: VulkanU32Pedalboard,
    ) -> Result<Self, VulkanBackendError> {
        Ok(Self {
            device_id: device_id.into(),
            device: VulkanComputeDevice::new()?,
            pedalboard,
            streams: HashMap::new(),
            active_queue: VecDeque::new(),
            output_queue: Vec::new(),
            dispatch_tick: 0,
        })
    }

    pub fn from_pedals(
        device_id: impl Into<String>,
        pedals: Vec<VulkanU32ShaderPedal>,
    ) -> Result<Self, VulkanBackendError> {
        Self::new(device_id, VulkanU32Pedalboard::new(pedals))
    }

    pub fn device_name(&self) -> &str {
        self.device.device_name()
    }

    fn stream_mut(&mut self, stream_id: &str) -> Result<&mut VulkanU32Stream, BackendError> {
        self.streams
            .get_mut(stream_id)
            .ok_or_else(|| BackendError::UnknownStream(stream_id.to_string()))
    }

    fn schedule(&mut self, stream_id: &str) {
        if !self.active_queue.iter().any(|active| active == stream_id) {
            self.active_queue.push_back(stream_id.to_string());
        }
    }

    fn process_token(&self, token_id: TokenId) -> Result<TokenId, VulkanBackendError> {
        let token =
            u32::try_from(token_id).map_err(|_| VulkanBackendError::InvalidToken(token_id))?;
        let run = self.pedalboard.process(&self.device, &[token])?;
        let output = run
            .output
            .first()
            .copied()
            .ok_or(VulkanBackendError::EmptyPedalOutput)?;
        Ok(TokenId::from(output))
    }
}

impl DeviceBackend for VulkanU32Backend {
    type Error = VulkanBackendError;

    fn backend_id(&self) -> &str {
        VULKAN_SPIRV_BACKEND_ID
    }

    fn device_id(&self) -> &str {
        &self.device_id
    }

    fn has_stream(&self, stream_id: &str) -> bool {
        self.streams.contains_key(stream_id)
    }

    fn create_stream(&mut self, stream_id: &str) -> Result<(), Self::Error> {
        if self.has_stream(stream_id) {
            return Err(BackendError::DuplicateStream(stream_id.to_string()).into());
        }
        self.streams
            .insert(stream_id.to_string(), VulkanU32Stream::default());
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
                *stream = VulkanU32Stream::default();
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
            return Err(BackendError::DuplicateStream(request.child_stream_id).into());
        }
        let child = match request.state_policy {
            ForkPolicy::Clone => self
                .streams
                .get(&request.parent_stream_id)
                .ok_or_else(|| BackendError::UnknownStream(request.parent_stream_id.clone()))?
                .clone(),
            ForkPolicy::Fresh => VulkanU32Stream::default(),
        };
        let child_has_work = child.has_work();
        self.streams.insert(request.child_stream_id.clone(), child);
        if request.random_policy == RandomPolicy::Fresh {
            // This toy Vulkan backend has no random state yet; the policy remains part of the contract.
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

            let (input, can_emit) = {
                let stream = self.stream_mut(&stream_id)?;
                let input = stream.next_input();
                let can_emit = input.is_some()
                    && stream.remaining_outputs > 0
                    && stream.pending_external.is_empty();
                (input, can_emit)
            };
            let Some(input) = input else {
                continue;
            };

            let processed_token = self.process_token(input.token_id)?;
            let mut output = None;
            let reschedule = {
                let device_id = self.device_id.clone();
                let stream = self.stream_mut(&stream_id)?;
                if can_emit {
                    let public = PublicOutputSignal::token(
                        format!("public_{}", stream.public_counter),
                        processed_token,
                        dispatch_tick,
                    );
                    stream.public_counter += 1;
                    stream.remaining_outputs -= 1;
                    let feedback = InputSignal::feedback(
                        format!("feedback_{}", stream.feedback_counter),
                        processed_token,
                    );
                    stream.feedback_counter += 1;
                    stream.pending_feedback.push_back(feedback);
                    output = Some(DeviceOutputEvent {
                        device_id,
                        stream_id: stream_id.clone(),
                        output: public,
                        dispatch_tick,
                    });
                }
                stream.has_work()
            };

            self.dispatch_tick += 1;
            ticks.push(DeviceDispatchTick {
                device_id: self.device_id.clone(),
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
            install_id: "vulkan_u32_installed_processor".to_string(),
            backend: self.backend_id().to_string(),
            permanent_circuit: PermanentCircuitManifest {
                pedal_count: self.pedalboard.pedals().len(),
                input_signal: "u32_token".to_string(),
                output_signal: "u32_token".to_string(),
                source_model_dir: None,
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
                        id: "spirv_programs".to_string(),
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
    use crate::vulkan_compute::compile_test_shader_words;

    fn add_one_pedal(id: &str) -> Option<VulkanU32ShaderPedal> {
        let spirv_words = compile_test_shader_words()?;
        VulkanU32ShaderPedal::new(id, "u32_add_one", spirv_words, 64).ok()
    }

    fn backend_with_adders(count: usize) -> Option<VulkanU32Backend> {
        let mut pedals = Vec::new();
        for index in 0..count {
            pedals.push(add_one_pedal(&format!("add_one_{index}"))?);
        }
        match VulkanU32Backend::from_pedals("vulkan_device_0", pedals) {
            Ok(backend) => Some(backend),
            Err(error) => {
                eprintln!("skipping Vulkan backend smoke: {error}");
                None
            }
        }
    }

    #[test]
    fn vulkan_backend_owns_feedback_loop_with_gpu_pedalboard() {
        let Some(mut backend) = backend_with_adders(1) else {
            return;
        };
        backend.create_stream("s0").unwrap();
        backend
            .inject_prompt(PromptInjection::new("s0", vec![1], 2))
            .unwrap();

        let run = backend.dispatch(16).unwrap();

        assert_eq!(backend.backend_id(), VULKAN_SPIRV_BACKEND_ID);
        assert_eq!(run.status, DispatchStatus::Idle);
        assert_eq!(run.ticks.len(), 3);
        assert_eq!(run.outputs.len(), 2);
        assert_eq!(run.outputs[0].output.token_id, 2);
        assert_eq!(run.outputs[1].output.token_id, 3);
        assert_eq!(backend.drain_outputs().unwrap().len(), 2);
    }

    #[test]
    fn vulkan_backend_uses_series_pedals_for_token_transform() {
        let Some(mut backend) = backend_with_adders(2) else {
            return;
        };
        backend.create_stream("s0").unwrap();
        backend
            .inject_prompt(PromptInjection::new("s0", vec![1], 1))
            .unwrap();

        let run = backend.dispatch(16).unwrap();

        assert_eq!(run.status, DispatchStatus::Idle);
        assert_eq!(run.ticks.len(), 2);
        assert_eq!(run.outputs.len(), 1);
        assert_eq!(run.outputs[0].output.token_id, 3);
    }

    #[test]
    fn vulkan_backend_manifest_names_gpu_memory_regions() {
        let Some(backend) = backend_with_adders(1) else {
            return;
        };

        let manifest = backend.describe();

        assert_eq!(manifest.backend, VULKAN_SPIRV_BACKEND_ID);
        assert_eq!(manifest.permanent_circuit.pedal_count, 1);
        assert_eq!(
            manifest.host_ports.private_feedback,
            "device_owned_insert_loop"
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
