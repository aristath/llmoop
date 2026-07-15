use serde::{Deserialize, Serialize};

pub type DeviceId = String;
pub type StreamId = String;
pub type TokenId = i64;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct InputSignal {
    pub id: String,
    pub token_id: TokenId,
    pub origin: String,
    pub route: String,
}

impl InputSignal {
    pub fn external(id: impl Into<String>, token_id: TokenId, origin: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            token_id,
            origin: origin.into(),
            route: "input".to_string(),
        }
    }

    pub fn feedback(id: impl Into<String>, token_id: TokenId) -> Self {
        Self {
            id: id.into(),
            token_id,
            origin: "insert_out".to_string(),
            route: "insert_in".to_string(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PromptInjection {
    pub stream_id: StreamId,
    pub prompt_ids: Vec<TokenId>,
    pub max_new_tokens: u32,
    pub eos_token_id: Option<TokenId>,
    pub origin: String,
}

impl PromptInjection {
    pub fn new(
        stream_id: impl Into<String>,
        prompt_ids: Vec<TokenId>,
        max_new_tokens: u32,
    ) -> Self {
        Self {
            stream_id: stream_id.into(),
            prompt_ids,
            max_new_tokens,
            eos_token_id: None,
            origin: "external_input".to_string(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ControlCommand {
    Continue {
        additional_public_outputs: u32,
        reason: String,
    },
    Interrupt {
        reason: String,
    },
    StopAfterCurrent {
        reason: String,
    },
    ResetState {
        reason: String,
    },
    ReseedRandom {
        seed: i64,
        source_id: Option<String>,
        reason: String,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ForkPolicy {
    Clone,
    Fresh,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RandomPolicy {
    Clone,
    Fresh,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ForkRequest {
    pub parent_stream_id: StreamId,
    pub child_stream_id: StreamId,
    pub state_policy: ForkPolicy,
    pub random_policy: RandomPolicy,
    pub random_seed: Option<i64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PublicOutputSignal {
    pub id: String,
    pub token_id: TokenId,
    pub source_tick: u64,
    pub source_model_tick: u64,
    pub sampler: serde_json::Value,
    pub route: String,
}

impl PublicOutputSignal {
    pub fn token(id: impl Into<String>, token_id: TokenId, source_tick: u64) -> Self {
        Self {
            id: id.into(),
            token_id,
            source_tick,
            source_model_tick: source_tick,
            sampler: serde_json::json!({
                "sampler_id": "contract_sampler",
                "implementation": "contract_only"
            }),
            route: "public_output".to_string(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DeviceOutputEvent {
    pub device_id: DeviceId,
    pub stream_id: StreamId,
    pub output: PublicOutputSignal,
    pub dispatch_tick: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceDispatchTick {
    pub device_id: DeviceId,
    pub dispatch_tick: u64,
    pub stream_id: StreamId,
    pub input: InputSignal,
    pub status: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DispatchStatus {
    Idle,
    BudgetExhausted,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DeviceDispatchRun {
    pub device_id: DeviceId,
    pub ticks: Vec<DeviceDispatchTick>,
    pub outputs: Vec<DeviceOutputEvent>,
    pub status: DispatchStatus,
    pub active_streams: Vec<StreamId>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StateAllocation {
    pub pedal_id: String,
    pub state_id: String,
    pub state_type: String,
    pub static_shape: Option<Vec<usize>>,
    pub elements_per_token: Option<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StreamTemplate {
    pub id: String,
    pub state_allocations: Vec<StateAllocation>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PermanentCircuitManifest {
    pub pedal_count: usize,
    pub input_signal: String,
    pub output_signal: String,
    pub source_model_dir: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostPortsManifest {
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub private_feedback: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryRegionKind {
    PermanentWeights,
    PermanentParameters,
    SpirvProgram,
    StreamTransientState,
    InputQueue,
    OutputQueue,
    EventQueue,
    RandomQueue,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemorySharing {
    SharedByAllStreams,
    PerStream,
    HostVisibleQueue,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryRegion {
    pub id: String,
    pub kind: MemoryRegionKind,
    pub sharing: MemorySharing,
    pub bytes: Option<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceMemoryPlan {
    pub regions: Vec<MemoryRegion>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct InstalledProcessorManifest {
    pub install_id: String,
    pub backend: String,
    pub permanent_circuit: PermanentCircuitManifest,
    pub host_ports: HostPortsManifest,
    pub stream_template: StreamTemplate,
    pub memory_plan: DeviceMemoryPlan,
}
