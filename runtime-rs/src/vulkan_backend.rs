use std::collections::{HashMap, VecDeque};
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::path::Path;

use crate::backend::{BackendError, DeviceBackend};
use crate::types::{
    ControlCommand, DeviceDispatchRun, DeviceDispatchTick, DeviceMemoryPlan, DeviceOutputEvent,
    DispatchStatus, ForkPolicy, ForkRequest, HostPortsManifest, InputSignal,
    InstalledProcessorManifest, MemoryRegion, MemoryRegionKind, MemorySharing,
    PermanentCircuitManifest, PromptInjection, PublicOutputSignal, RandomPolicy, StateAllocation,
    StreamId, StreamTemplate, TokenId,
};
use crate::vulkan::{
    VULKAN_SPIRV_BACKEND_ID, VulkanBackendArtifactManifest, VulkanBackendDescriptor,
};
use crate::vulkan_compute::{
    VulkanComputeDevice, VulkanError, VulkanPipelineCacheStats, VulkanU32ResidentBuffer,
    VulkanU32ResidentCopy, VulkanU32ResidentDispatch, VulkanU32ShaderPedal,
};
use crate::vulkan_pedalboard::VulkanU32Pedalboard;

const VULKAN_U32_TOKEN_PORT_CAPACITY: usize = 1;
const VULKAN_U32_STREAM_PORT_COUNT: usize = 3;
const VULKAN_U32_STREAM_ROUTE_COUNT: usize = 3;
const VULKAN_U32_STREAM_PORT_BYTES: usize =
    VULKAN_U32_STREAM_PORT_COUNT * std::mem::size_of::<u32>();

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VulkanBackendError {
    Backend(BackendError),
    Vulkan(VulkanError),
    InvalidDescriptor(String),
    InvalidToken(TokenId),
    EmptyPedalOutput,
}

impl Display for VulkanBackendError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Backend(error) => Display::fmt(error, f),
            Self::Vulkan(error) => Display::fmt(error, f),
            Self::InvalidDescriptor(message) => f.write_str(message),
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

struct VulkanU32StreamPorts {
    signal_frame: VulkanU32ResidentBuffer,
    public_output: VulkanU32ResidentBuffer,
    private_feedback: VulkanU32ResidentBuffer,
}

impl VulkanU32StreamPorts {
    fn new(device: &VulkanComputeDevice) -> Result<Self, VulkanError> {
        let ports = Self {
            signal_frame: device.create_u32_resident_buffer(VULKAN_U32_TOKEN_PORT_CAPACITY)?,
            public_output: device.create_u32_resident_buffer(VULKAN_U32_TOKEN_PORT_CAPACITY)?,
            private_feedback: device.create_u32_resident_buffer(VULKAN_U32_TOKEN_PORT_CAPACITY)?,
        };
        ports.clear()?;
        Ok(ports)
    }

    fn clone_from(device: &VulkanComputeDevice, source: &Self) -> Result<Self, VulkanError> {
        let ports = Self::new(device)?;
        ports.signal_frame.write(&source.signal_frame.read(1)?)?;
        ports.public_output.write(&source.public_output.read(1)?)?;
        ports
            .private_feedback
            .write(&source.private_feedback.read(1)?)?;
        Ok(ports)
    }

    fn clear(&self) -> Result<(), VulkanError> {
        self.signal_frame.write(&[0])?;
        self.public_output.write(&[0])?;
        self.private_feedback.write(&[0])?;
        Ok(())
    }
}

struct VulkanU32StreamAdvance {
    private_feedback_token: TokenId,
    public_token: Option<TokenId>,
}

struct VulkanU32QueuedAdvance {
    dispatch_tick: u64,
    input: InputSignal,
    public_output: Option<PublicOutputSignal>,
    has_more_work: bool,
}

struct VulkanU32StreamRun {
    advances: Vec<VulkanU32QueuedAdvance>,
    next_dispatch_tick: u64,
    has_more_work: bool,
}

enum VulkanU32StreamInput {
    External(InputSignal),
    PrivateFeedback(InputSignal),
}

impl VulkanU32StreamInput {
    fn into_signal(self) -> InputSignal {
        match self {
            Self::External(signal) | Self::PrivateFeedback(signal) => signal,
        }
    }
}

struct VulkanU32StreamRoutes {
    private_feedback_to_signal: VulkanU32ResidentCopy,
    signal_to_private_feedback: VulkanU32ResidentCopy,
    signal_to_public_output: VulkanU32ResidentCopy,
}

impl VulkanU32StreamRoutes {
    fn new(
        device: &VulkanComputeDevice,
        ports: &VulkanU32StreamPorts,
    ) -> Result<Self, VulkanError> {
        Ok(Self {
            private_feedback_to_signal: device.create_u32_resident_copy(
                &ports.private_feedback,
                &ports.signal_frame,
                VULKAN_U32_TOKEN_PORT_CAPACITY,
            )?,
            signal_to_private_feedback: device.create_u32_resident_copy(
                &ports.signal_frame,
                &ports.private_feedback,
                VULKAN_U32_TOKEN_PORT_CAPACITY,
            )?,
            signal_to_public_output: device.create_u32_resident_copy(
                &ports.signal_frame,
                &ports.public_output,
                VULKAN_U32_TOKEN_PORT_CAPACITY,
            )?,
        })
    }
}

struct VulkanU32MountedStream {
    signal_dispatches: Vec<VulkanU32ResidentDispatch>,
    routes: VulkanU32StreamRoutes,
    ports: VulkanU32StreamPorts,
}

impl VulkanU32MountedStream {
    fn new(
        device: &VulkanComputeDevice,
        pedalboard: &VulkanU32Pedalboard,
    ) -> Result<Self, VulkanError> {
        let ports = VulkanU32StreamPorts::new(device)?;
        let signal_dispatches =
            pedalboard.create_resident_dispatches(device, &ports.signal_frame, 1)?;
        let routes = VulkanU32StreamRoutes::new(device, &ports)?;
        Ok(Self {
            signal_dispatches,
            routes,
            ports,
        })
    }

    fn clone_from(
        device: &VulkanComputeDevice,
        pedalboard: &VulkanU32Pedalboard,
        source: &Self,
    ) -> Result<Self, VulkanError> {
        let ports = VulkanU32StreamPorts::clone_from(device, &source.ports)?;
        let signal_dispatches =
            pedalboard.create_resident_dispatches(device, &ports.signal_frame, 1)?;
        let routes = VulkanU32StreamRoutes::new(device, &ports)?;
        Ok(Self {
            signal_dispatches,
            routes,
            ports,
        })
    }

    fn reset(&self) -> Result<(), VulkanError> {
        self.ports.clear()
    }

    fn advance_external(
        &self,
        device: &VulkanComputeDevice,
        pedalboard: &VulkanU32Pedalboard,
        input_token: TokenId,
        emit_public: bool,
    ) -> Result<VulkanU32StreamAdvance, VulkanBackendError> {
        let token = u32::try_from(input_token)
            .map_err(|_| VulkanBackendError::InvalidToken(input_token))?;
        self.ports.signal_frame.write(&[token])?;
        self.advance_signal_frame(device, pedalboard, emit_public)
    }

    fn advance_private_feedback(
        &self,
        device: &VulkanComputeDevice,
        pedalboard: &VulkanU32Pedalboard,
        emit_public: bool,
    ) -> Result<VulkanU32StreamAdvance, VulkanBackendError> {
        device.run_u32_resident_copy(
            &self.routes.private_feedback_to_signal,
            VULKAN_U32_TOKEN_PORT_CAPACITY,
        )?;
        self.advance_signal_frame(device, pedalboard, emit_public)
    }

    fn advance_signal_frame(
        &self,
        device: &VulkanComputeDevice,
        pedalboard: &VulkanU32Pedalboard,
        emit_public: bool,
    ) -> Result<VulkanU32StreamAdvance, VulkanBackendError> {
        pedalboard.dispatch_bound_resident_in_place(device, &self.signal_dispatches, 1)?;
        let output = self
            .ports
            .signal_frame
            .read(1)?
            .first()
            .copied()
            .ok_or(VulkanBackendError::EmptyPedalOutput)?;
        device.run_u32_resident_copy(
            &self.routes.signal_to_private_feedback,
            VULKAN_U32_TOKEN_PORT_CAPACITY,
        )?;
        if emit_public {
            device.run_u32_resident_copy(
                &self.routes.signal_to_public_output,
                VULKAN_U32_TOKEN_PORT_CAPACITY,
            )?;
        }

        let output_token = TokenId::from(output);
        Ok(VulkanU32StreamAdvance {
            private_feedback_token: output_token,
            public_token: emit_public.then_some(output_token),
        })
    }
}

struct VulkanU32Stream {
    mounted: VulkanU32MountedStream,
    pending_external: VecDeque<InputSignal>,
    private_feedback_signal: Option<InputSignal>,
    remaining_outputs: u32,
    input_counter: u64,
    public_counter: u64,
    feedback_counter: u64,
}

impl VulkanU32Stream {
    fn new(
        device: &VulkanComputeDevice,
        pedalboard: &VulkanU32Pedalboard,
    ) -> Result<Self, VulkanError> {
        Ok(Self {
            mounted: VulkanU32MountedStream::new(device, pedalboard)?,
            pending_external: VecDeque::new(),
            private_feedback_signal: None,
            remaining_outputs: 0,
            input_counter: 0,
            public_counter: 0,
            feedback_counter: 0,
        })
    }

    fn fork_clone(
        &self,
        device: &VulkanComputeDevice,
        pedalboard: &VulkanU32Pedalboard,
    ) -> Result<Self, VulkanError> {
        Ok(Self {
            mounted: VulkanU32MountedStream::clone_from(device, pedalboard, &self.mounted)?,
            pending_external: self.pending_external.clone(),
            private_feedback_signal: self.private_feedback_signal.clone(),
            remaining_outputs: self.remaining_outputs,
            input_counter: self.input_counter,
            public_counter: self.public_counter,
            feedback_counter: self.feedback_counter,
        })
    }

    fn reset_state(&mut self) -> Result<(), VulkanError> {
        self.pending_external.clear();
        self.private_feedback_signal = None;
        self.remaining_outputs = 0;
        self.input_counter = 0;
        self.public_counter = 0;
        self.feedback_counter = 0;
        self.mounted.reset()
    }

    fn has_work(&self) -> bool {
        !self.pending_external.is_empty() || self.private_feedback_signal.is_some()
    }

    fn next_input(&mut self) -> Option<VulkanU32StreamInput> {
        self.pending_external
            .pop_front()
            .map(VulkanU32StreamInput::External)
            .or_else(|| {
                self.private_feedback_signal
                    .take()
                    .map(VulkanU32StreamInput::PrivateFeedback)
            })
    }

    fn advance_queued_once(
        &mut self,
        device: &VulkanComputeDevice,
        pedalboard: &VulkanU32Pedalboard,
        dispatch_tick: u64,
    ) -> Result<Option<VulkanU32QueuedAdvance>, VulkanBackendError> {
        let input = self.next_input();
        let can_emit =
            input.is_some() && self.remaining_outputs > 0 && self.pending_external.is_empty();
        let Some(input) = input else {
            return Ok(None);
        };

        let advance = match &input {
            VulkanU32StreamInput::External(signal) => {
                self.mounted
                    .advance_external(device, pedalboard, signal.token_id, can_emit)?
            }
            VulkanU32StreamInput::PrivateFeedback(_) => self
                .mounted
                .advance_private_feedback(device, pedalboard, can_emit)?,
        };
        let public_output = advance.public_token.map(|public_token| {
            let public = PublicOutputSignal::token(
                format!("public_{}", self.public_counter),
                public_token,
                dispatch_tick,
            );
            self.public_counter += 1;
            self.remaining_outputs -= 1;

            let feedback = InputSignal::feedback(
                format!("feedback_{}", self.feedback_counter),
                advance.private_feedback_token,
            );
            self.feedback_counter += 1;
            self.private_feedback_signal = Some(feedback);

            public
        });

        Ok(Some(VulkanU32QueuedAdvance {
            dispatch_tick,
            input: input.into_signal(),
            public_output,
            has_more_work: self.has_work(),
        }))
    }

    fn advance_for_budget(
        &mut self,
        device: &VulkanComputeDevice,
        pedalboard: &VulkanU32Pedalboard,
        start_dispatch_tick: u64,
        max_ticks: usize,
    ) -> Result<VulkanU32StreamRun, VulkanBackendError> {
        let mut advances = Vec::new();
        let mut dispatch_tick = start_dispatch_tick;

        while advances.len() < max_ticks {
            let Some(advance) = self.advance_queued_once(device, pedalboard, dispatch_tick)? else {
                break;
            };
            dispatch_tick += 1;
            let has_more_work = advance.has_more_work;
            advances.push(advance);
            if !has_more_work {
                break;
            }
        }

        Ok(VulkanU32StreamRun {
            has_more_work: self.has_work(),
            next_dispatch_tick: dispatch_tick,
            advances,
        })
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
        let device = VulkanComputeDevice::new()?;
        pedalboard.install(&device)?;
        Ok(Self {
            device_id: device_id.into(),
            device,
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

    pub fn from_descriptor(
        descriptor: VulkanBackendDescriptor,
    ) -> Result<Self, VulkanBackendError> {
        if descriptor.backend_id != VULKAN_SPIRV_BACKEND_ID {
            return Err(VulkanBackendError::InvalidDescriptor(format!(
                "unsupported Vulkan backend descriptor {:?}",
                descriptor.backend_id
            )));
        }
        let mut pedals = Vec::with_capacity(descriptor.programs.len());
        for program in descriptor.programs {
            pedals.push(VulkanU32ShaderPedal::from_program(program)?);
        }
        Self::from_pedals(descriptor.device_id, pedals)
    }

    pub fn from_artifact_manifest(
        manifest: VulkanBackendArtifactManifest,
        artifact_root: impl AsRef<Path>,
    ) -> Result<Self, VulkanBackendError> {
        let descriptor = manifest.resolve(artifact_root).map_err(|error| {
            VulkanBackendError::InvalidDescriptor(format!(
                "failed to resolve Vulkan backend artifact manifest: {error}"
            ))
        })?;
        Self::from_descriptor(descriptor)
    }

    pub fn from_artifact_manifest_file(
        manifest_path: impl AsRef<Path>,
        artifact_root: impl AsRef<Path>,
    ) -> Result<Self, VulkanBackendError> {
        let manifest =
            VulkanBackendArtifactManifest::from_json_file(manifest_path).map_err(|error| {
                VulkanBackendError::InvalidDescriptor(format!(
                    "failed to read Vulkan backend artifact manifest: {error}"
                ))
            })?;
        Self::from_artifact_manifest(manifest, artifact_root)
    }

    pub fn device_name(&self) -> &str {
        self.device.device_name()
    }

    pub fn pipeline_cache_stats(&self) -> VulkanPipelineCacheStats {
        self.device.pipeline_cache_stats()
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
}

impl Drop for VulkanU32Backend {
    fn drop(&mut self) {
        self.streams.clear();
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
        self.streams.insert(
            stream_id.to_string(),
            VulkanU32Stream::new(&self.device, &self.pedalboard)?,
        );
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
                stream.private_feedback_signal = None;
                stream.remaining_outputs = 0;
            }
            ControlCommand::StopAfterCurrent { .. } => {
                stream.remaining_outputs = 0;
            }
            ControlCommand::ResetState { .. } => {
                stream.reset_state()?;
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
                .fork_clone(&self.device, &self.pedalboard)?,
            ForkPolicy::Fresh => VulkanU32Stream::new(&self.device, &self.pedalboard)?,
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
            let remaining_tick_budget = max_ticks as usize - ticks.len();

            let run = {
                let device = &self.device;
                let pedalboard = &self.pedalboard;
                let stream = self
                    .streams
                    .get_mut(&stream_id)
                    .ok_or_else(|| BackendError::UnknownStream(stream_id.clone()))?;
                stream.advance_for_budget(
                    device,
                    pedalboard,
                    self.dispatch_tick,
                    remaining_tick_budget,
                )?
            };
            if run.advances.is_empty() {
                continue;
            }

            for advance in run.advances {
                let output = advance.public_output.map(|public| DeviceOutputEvent {
                    device_id: self.device_id.clone(),
                    stream_id: stream_id.clone(),
                    output: public,
                    dispatch_tick: advance.dispatch_tick,
                });

                ticks.push(DeviceDispatchTick {
                    device_id: self.device_id.clone(),
                    dispatch_tick: advance.dispatch_tick,
                    stream_id: stream_id.clone(),
                    input: advance.input,
                    status: "processed".to_string(),
                });

                if let Some(event) = output {
                    self.output_queue.push(event.clone());
                    outputs.push(event);
                }
            }

            self.dispatch_tick = run.next_dispatch_tick;
            if run.has_more_work {
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
                state_allocations: vec![
                    StateAllocation {
                        pedal_id: "mounted_stream".to_string(),
                        state_id: "signal_frame".to_string(),
                        state_type: "u32_resident_signal_port".to_string(),
                        static_shape: Some(vec![VULKAN_U32_TOKEN_PORT_CAPACITY]),
                        elements_per_token: Some(1),
                    },
                    StateAllocation {
                        pedal_id: "mounted_stream".to_string(),
                        state_id: "public_output".to_string(),
                        state_type: "u32_resident_public_output_port".to_string(),
                        static_shape: Some(vec![VULKAN_U32_TOKEN_PORT_CAPACITY]),
                        elements_per_token: Some(1),
                    },
                    StateAllocation {
                        pedal_id: "mounted_stream".to_string(),
                        state_id: "private_feedback".to_string(),
                        state_type: "u32_resident_feedback_port".to_string(),
                        static_shape: Some(vec![VULKAN_U32_TOKEN_PORT_CAPACITY]),
                        elements_per_token: Some(1),
                    },
                    StateAllocation {
                        pedal_id: "mounted_stream".to_string(),
                        state_id: "signal_dispatches".to_string(),
                        state_type: "vulkan_resident_dispatch_bindings".to_string(),
                        static_shape: Some(vec![self.pedalboard.pedals().len()]),
                        elements_per_token: None,
                    },
                    StateAllocation {
                        pedal_id: "mounted_stream".to_string(),
                        state_id: "port_routes".to_string(),
                        state_type: "vulkan_resident_copy_bindings".to_string(),
                        static_shape: Some(vec![VULKAN_U32_STREAM_ROUTE_COUNT]),
                        elements_per_token: None,
                    },
                ],
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
                        bytes: Some(VULKAN_U32_STREAM_PORT_BYTES),
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
    use crate::vulkan::{
        SpirvPedalProgram, SpirvPedalProgramRef, VulkanBackendArtifactManifest,
        VulkanBackendDescriptor, write_spirv_words,
    };
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

    fn backend_descriptor_with_adders(count: usize) -> Option<VulkanBackendDescriptor> {
        let mut descriptor = VulkanBackendDescriptor::empty("vulkan_device_0");
        for index in 0..count {
            let spirv_words = compile_test_shader_words()?;
            let program =
                SpirvPedalProgram::new(format!("add_one_{index}"), "u32_add_one", spirv_words);
            descriptor = descriptor.with_program(program);
        }
        Some(descriptor)
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
    fn vulkan_stream_ports_retain_signal_public_and_private_feedback() {
        let Some(mut backend) = backend_with_adders(1) else {
            return;
        };
        backend.create_stream("s0").unwrap();
        backend
            .inject_prompt(PromptInjection::new("s0", vec![1], 2))
            .unwrap();

        let run = backend.dispatch(16).unwrap();
        let stream = backend.streams.get("s0").unwrap();

        assert_eq!(run.status, DispatchStatus::Idle);
        assert_eq!(stream.mounted.ports.signal_frame.read(1).unwrap(), vec![4]);
        assert_eq!(stream.mounted.ports.public_output.read(1).unwrap(), vec![3]);
        assert_eq!(
            stream.mounted.ports.private_feedback.read(1).unwrap(),
            vec![4]
        );
    }

    #[test]
    fn mounted_stream_advance_owns_public_and_private_routing() {
        let Some(backend) = backend_with_adders(1) else {
            return;
        };
        let mounted = match VulkanU32MountedStream::new(&backend.device, &backend.pedalboard) {
            Ok(mounted) => mounted,
            Err(error) => {
                eprintln!("skipping mounted stream smoke: {error}");
                return;
            }
        };
        assert_eq!(mounted.signal_dispatches.len(), 1);

        let private_advance = mounted
            .advance_external(&backend.device, &backend.pedalboard, 1, false)
            .unwrap();

        assert_eq!(private_advance.private_feedback_token, 2);
        assert_eq!(private_advance.public_token, None);
        assert_eq!(mounted.ports.private_feedback.read(1).unwrap(), vec![2]);
        assert_eq!(mounted.ports.public_output.read(1).unwrap(), vec![0]);

        let public_advance = mounted
            .advance_external(&backend.device, &backend.pedalboard, 2, true)
            .unwrap();

        assert_eq!(public_advance.private_feedback_token, 3);
        assert_eq!(public_advance.public_token, Some(3));
        assert_eq!(mounted.ports.private_feedback.read(1).unwrap(), vec![3]);
        assert_eq!(mounted.ports.public_output.read(1).unwrap(), vec![3]);

        let feedback_advance = mounted
            .advance_private_feedback(&backend.device, &backend.pedalboard, false)
            .unwrap();

        assert_eq!(feedback_advance.private_feedback_token, 4);
        assert_eq!(feedback_advance.public_token, None);
        assert_eq!(mounted.ports.signal_frame.read(1).unwrap(), vec![4]);
        assert_eq!(mounted.ports.private_feedback.read(1).unwrap(), vec![4]);
        assert_eq!(mounted.ports.public_output.read(1).unwrap(), vec![3]);
    }

    #[test]
    fn stream_advance_keeps_feedback_in_mounted_insert_port() {
        let Some(backend) = backend_with_adders(1) else {
            return;
        };
        let mut stream = match VulkanU32Stream::new(&backend.device, &backend.pedalboard) {
            Ok(stream) => stream,
            Err(error) => {
                eprintln!("skipping stream advance smoke: {error}");
                return;
            }
        };
        stream
            .pending_external
            .push_back(InputSignal::external("input_0", 1, "test"));
        stream.remaining_outputs = 1;

        let advance = stream
            .advance_queued_once(&backend.device, &backend.pedalboard, 7)
            .unwrap()
            .unwrap();

        assert_eq!(advance.dispatch_tick, 7);
        assert_eq!(advance.input.token_id, 1);
        assert_eq!(advance.public_output.unwrap().token_id, 2);
        assert_eq!(stream.public_counter, 1);
        assert_eq!(stream.feedback_counter, 1);
        assert_eq!(stream.remaining_outputs, 0);
        assert_eq!(stream.private_feedback_signal.as_ref().unwrap().token_id, 2);
        assert!(advance.has_more_work);
        assert_eq!(stream.mounted.ports.public_output.read(1).unwrap(), vec![2]);
        assert_eq!(
            stream.mounted.ports.private_feedback.read(1).unwrap(),
            vec![2]
        );

        let closing = stream
            .advance_queued_once(&backend.device, &backend.pedalboard, 8)
            .unwrap()
            .unwrap();

        assert_eq!(closing.input.route, "insert_in");
        assert_eq!(closing.input.token_id, 2);
        assert!(closing.public_output.is_none());
        assert!(!closing.has_more_work);
        assert!(stream.private_feedback_signal.is_none());
        assert_eq!(stream.mounted.ports.signal_frame.read(1).unwrap(), vec![3]);
        assert_eq!(
            stream.mounted.ports.private_feedback.read(1).unwrap(),
            vec![3]
        );
    }

    #[test]
    fn stream_run_advances_feedback_loop_until_budget_or_idle() {
        let Some(backend) = backend_with_adders(1) else {
            return;
        };
        let mut stream = match VulkanU32Stream::new(&backend.device, &backend.pedalboard) {
            Ok(stream) => stream,
            Err(error) => {
                eprintln!("skipping stream run smoke: {error}");
                return;
            }
        };
        stream
            .pending_external
            .push_back(InputSignal::external("input_0", 1, "test"));
        stream.remaining_outputs = 3;

        let run = stream
            .advance_for_budget(&backend.device, &backend.pedalboard, 9, 2)
            .unwrap();

        assert_eq!(run.advances.len(), 2);
        assert_eq!(run.advances[0].dispatch_tick, 9);
        assert_eq!(run.advances[0].public_output.as_ref().unwrap().token_id, 2);
        assert_eq!(run.advances[1].dispatch_tick, 10);
        assert_eq!(run.advances[1].public_output.as_ref().unwrap().token_id, 3);
        assert_eq!(run.next_dispatch_tick, 11);
        assert!(run.has_more_work);
        assert_eq!(stream.remaining_outputs, 1);
        assert_eq!(stream.private_feedback_signal.as_ref().unwrap().token_id, 3);
    }

    #[test]
    fn vulkan_clone_fork_copies_resident_ports_without_sharing_them() {
        let Some(mut backend) = backend_with_adders(1) else {
            return;
        };
        backend.create_stream("parent").unwrap();
        backend
            .inject_prompt(PromptInjection::new("parent", vec![1], 1))
            .unwrap();
        backend.dispatch(1).unwrap();
        backend
            .fork_stream(ForkRequest {
                parent_stream_id: "parent".to_string(),
                child_stream_id: "child".to_string(),
                state_policy: ForkPolicy::Clone,
                random_policy: RandomPolicy::Clone,
                random_seed: None,
            })
            .unwrap();

        let parent_before_reset = backend.streams.get("parent").unwrap();
        let child_before_reset = backend.streams.get("child").unwrap();
        assert_eq!(
            parent_before_reset
                .mounted
                .ports
                .private_feedback
                .read(1)
                .unwrap(),
            vec![2]
        );
        assert_eq!(
            child_before_reset
                .mounted
                .ports
                .private_feedback
                .read(1)
                .unwrap(),
            vec![2]
        );

        backend
            .control(
                "parent",
                ControlCommand::ResetState {
                    reason: "test reset".to_string(),
                },
            )
            .unwrap();

        let parent_after_reset = backend.streams.get("parent").unwrap();
        let child_after_reset = backend.streams.get("child").unwrap();
        assert_eq!(
            parent_after_reset
                .mounted
                .ports
                .private_feedback
                .read(1)
                .unwrap(),
            vec![0]
        );
        assert_eq!(
            child_after_reset
                .mounted
                .ports
                .private_feedback
                .read(1)
                .unwrap(),
            vec![2]
        );
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
    fn vulkan_backend_lets_stream_consume_bounded_dispatch_budget() {
        let Some(mut backend) = backend_with_adders(1) else {
            return;
        };
        backend.create_stream("s0").unwrap();
        backend
            .inject_prompt(PromptInjection::new("s0", vec![1], 3))
            .unwrap();

        let first = backend.dispatch(2).unwrap();
        let second = backend.dispatch(8).unwrap();

        assert_eq!(first.status, DispatchStatus::BudgetExhausted);
        assert_eq!(first.ticks.len(), 2);
        assert_eq!(first.outputs.len(), 2);
        assert_eq!(first.outputs[0].output.token_id, 2);
        assert_eq!(first.outputs[1].output.token_id, 3);
        assert_eq!(first.active_streams, vec!["s0".to_string()]);

        assert_eq!(second.status, DispatchStatus::Idle);
        assert_eq!(second.ticks.len(), 2);
        assert_eq!(second.outputs.len(), 1);
        assert_eq!(second.outputs[0].output.token_id, 4);
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
        assert_eq!(manifest.stream_template.state_allocations.len(), 5);
        assert!(
            manifest
                .stream_template
                .state_allocations
                .iter()
                .any(|allocation| allocation.state_id == "private_feedback")
        );
        assert!(
            manifest
                .stream_template
                .state_allocations
                .iter()
                .any(|allocation| allocation.state_id == "signal_dispatches"
                    && allocation.static_shape == Some(vec![1]))
        );
        assert!(
            manifest
                .stream_template
                .state_allocations
                .iter()
                .any(|allocation| allocation.state_id == "port_routes"
                    && allocation.static_shape == Some(vec![VULKAN_U32_STREAM_ROUTE_COUNT]))
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
        assert!(
            manifest
                .memory_plan
                .regions
                .iter()
                .any(|region| region.id == "stream_transient_state"
                    && region.bytes == Some(VULKAN_U32_STREAM_PORT_BYTES))
        );
    }

    #[test]
    fn vulkan_backend_can_be_installed_from_descriptor() {
        let Some(descriptor) = backend_descriptor_with_adders(2) else {
            return;
        };
        let mut backend = match VulkanU32Backend::from_descriptor(descriptor) {
            Ok(backend) => backend,
            Err(error) => {
                eprintln!("skipping Vulkan backend descriptor smoke: {error}");
                return;
            }
        };
        assert_eq!(
            backend.pipeline_cache_stats(),
            VulkanPipelineCacheStats {
                u32_storage_pipelines: 1,
                hits: 1,
                misses: 1
            }
        );
        backend.create_stream("s0").unwrap();
        backend
            .inject_prompt(PromptInjection::new("s0", vec![1], 1))
            .unwrap();

        let run = backend.dispatch(16).unwrap();

        assert_eq!(backend.backend_id(), VULKAN_SPIRV_BACKEND_ID);
        assert_eq!(run.status, DispatchStatus::Idle);
        assert_eq!(run.outputs.len(), 1);
        assert_eq!(run.outputs[0].output.token_id, 3);
        assert_eq!(backend.describe().permanent_circuit.pedal_count, 2);
    }

    #[test]
    fn vulkan_backend_can_be_installed_from_artifact_manifest() {
        let Some(spirv_words) = compile_test_shader_words() else {
            return;
        };
        let root = std::env::temp_dir().join(format!(
            "llmoop-vulkan-backend-artifacts-{}",
            std::process::id()
        ));
        let shader_dir = root.join("shaders");
        std::fs::create_dir_all(&shader_dir).unwrap();
        write_spirv_words(shader_dir.join("add_one.spv"), &spirv_words).unwrap();

        let manifest = VulkanBackendArtifactManifest::empty("vulkan_device_0").with_program(
            SpirvPedalProgramRef::new("add_one_0", "u32_add_one", "shaders/add_one.spv"),
        );
        let mut backend = match VulkanU32Backend::from_artifact_manifest(manifest, &root) {
            Ok(backend) => backend,
            Err(error) => {
                eprintln!("skipping Vulkan backend artifact manifest smoke: {error}");
                let _ = std::fs::remove_dir_all(root);
                return;
            }
        };
        backend.create_stream("s0").unwrap();
        backend
            .inject_prompt(PromptInjection::new("s0", vec![41], 1))
            .unwrap();

        let run = backend.dispatch(16).unwrap();

        assert_eq!(run.status, DispatchStatus::Idle);
        assert_eq!(run.outputs.len(), 1);
        assert_eq!(run.outputs[0].output.token_id, 42);
        assert_eq!(backend.describe().permanent_circuit.pedal_count, 1);

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn vulkan_backend_can_be_installed_from_bundled_artifact_manifest_file() {
        let artifact_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("artifacts")
            .join("add_one");
        let manifest_path = artifact_root.join("backend.json");
        let mut backend =
            match VulkanU32Backend::from_artifact_manifest_file(&manifest_path, &artifact_root) {
                Ok(backend) => backend,
                Err(error) => {
                    eprintln!("skipping bundled Vulkan artifact smoke: {error}");
                    return;
                }
            };
        backend.create_stream("s0").unwrap();
        backend
            .inject_prompt(PromptInjection::new("s0", vec![6], 1))
            .unwrap();

        let run = backend.dispatch(16).unwrap();

        assert_eq!(run.status, DispatchStatus::Idle);
        assert_eq!(run.outputs.len(), 1);
        assert_eq!(run.outputs[0].output.token_id, 7);
    }

    #[test]
    fn vulkan_backend_rejects_wrong_descriptor_backend_id() {
        let descriptor = VulkanBackendDescriptor {
            backend_id: "not_vulkan".to_string(),
            device_id: "device_0".to_string(),
            queue_family: None,
            programs: Vec::new(),
        };

        match VulkanU32Backend::from_descriptor(descriptor) {
            Err(VulkanBackendError::InvalidDescriptor(_)) => {}
            Err(error) => panic!("expected invalid descriptor error, got {error}"),
            Ok(_) => panic!("expected descriptor installation to fail"),
        }
    }
}
