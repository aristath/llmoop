use std::error::Error;
use std::fmt::{Display, Formatter};

use crate::types::{
    ControlCommand, DeviceDispatchRun, DeviceOutputEvent, ForkRequest, InputSignal,
    InstalledProcessorManifest, PromptInjection, StreamId,
};

pub trait DeviceBackend {
    type Error: Error + Send + Sync + 'static;

    fn backend_id(&self) -> &str;
    fn device_id(&self) -> &str;
    fn has_stream(&self, stream_id: &str) -> bool;
    fn create_stream(&mut self, stream_id: &str) -> Result<(), Self::Error>;
    fn inject_prompt(&mut self, injection: PromptInjection) -> Result<(), Self::Error>;
    fn inject_token(&mut self, stream_id: &str, signal: InputSignal) -> Result<(), Self::Error>;
    fn control(&mut self, stream_id: &str, command: ControlCommand) -> Result<(), Self::Error>;
    fn fork_stream(&mut self, request: ForkRequest) -> Result<(), Self::Error>;
    fn dispatch(&mut self, max_ticks: u32) -> Result<DeviceDispatchRun, Self::Error>;
    fn drain_outputs(&mut self) -> Result<Vec<DeviceOutputEvent>, Self::Error>;
    fn describe(&self) -> InstalledProcessorManifest;
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BackendError {
    DuplicateStream(StreamId),
    UnknownStream(StreamId),
}

impl Display for BackendError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DuplicateStream(stream_id) => write!(f, "stream {stream_id:?} already exists"),
            Self::UnknownStream(stream_id) => write!(f, "unknown stream {stream_id:?}"),
        }
    }
}

impl Error for BackendError {}
