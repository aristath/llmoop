#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VulkanResidentTokenEngineRunBudget {
    pub max_scheduler_turns: usize,
    pub max_runtime_cycles_per_turn: usize,
    pub ticks_per_runtime: usize,
}

impl VulkanResidentTokenEngineRunBudget {
    pub fn new(
        max_scheduler_turns: usize,
        max_runtime_cycles_per_turn: usize,
        ticks_per_runtime: usize,
    ) -> Self {
        Self {
            max_scheduler_turns,
            max_runtime_cycles_per_turn,
            ticks_per_runtime,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VulkanResidentTokenEngineRunStopCondition {
    Idle,
    SchedulerTurnBudget,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentTokenEngineRun {
    pub max_scheduler_turns: usize,
    pub max_runtime_cycles_per_turn: usize,
    pub ticks_per_runtime: usize,
    pub stop_condition: VulkanResidentTokenEngineRunStopCondition,
    pub scheduler_runs: Vec<VulkanResidentTokenRuntimeSchedulerRun>,
    pub output_events: Vec<VulkanResidentTokenRuntimeSchedulerOutputEvent>,
    pub runtime_cycle_count: usize,
    pub start_snapshot: VulkanResidentTokenEngineSnapshot,
    pub end_snapshot: VulkanResidentTokenEngineSnapshot,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentTokenEngineSubmittedInputRun {
    pub stream_id: String,
    pub input_event_id: String,
    pub queued_input_event: VulkanResidentTokenRuntimeQueuedInputEvent,
    pub run: VulkanResidentTokenEngineRun,
    pub output_events: Vec<VulkanResidentTokenRuntimeSchedulerOutputEvent>,
    pub generated_token_ids: Vec<u32>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentTokenEngineSubmittedTextRun {
    pub stream_id: String,
    pub input_event_id: String,
    pub input_text: String,
    pub encoded_token_ids: Vec<u32>,
    pub generated_text: String,
    pub submitted_tokens: VulkanResidentTokenEngineSubmittedInputRun,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentTokenEngineLiveTextTurnRun {
    pub stream_id: String,
    pub input_event_id: String,
    pub queued_input_event: VulkanResidentTokenEngineQueuedTextInputEvent,
    pub cycles: Vec<VulkanResidentTokenEngineTextCycleRun>,
    pub output_events: Vec<VulkanResidentTokenEngineTextOutputEvent>,
    pub generated_token_ids: Vec<u32>,
    pub generated_text: String,
    pub output_text: String,
    pub stop_condition: VulkanResidentTokenEngineRunStopCondition,
    pub runtime_cycle_count: usize,
}

impl VulkanResidentTokenEngineLiveTextTurnRun {
    pub fn scheduler_turn_count(&self) -> usize {
        self.cycles.len()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentTokenEngineTextInputRequest {
    pub stream_id: String,
    pub input_event_id: String,
    pub input_text: String,
    pub max_public_tokens: usize,
    pub origin: String,
}

impl VulkanResidentTokenEngineTextInputRequest {
    pub fn new(
        stream_id: impl Into<String>,
        input_event_id: impl Into<String>,
        input_text: impl Into<String>,
        max_public_tokens: usize,
    ) -> Self {
        Self {
            stream_id: stream_id.into(),
            input_event_id: input_event_id.into(),
            input_text: input_text.into(),
            max_public_tokens,
            origin: "host".to_string(),
        }
    }

    pub fn with_origin(mut self, origin: impl Into<String>) -> Self {
        self.origin = origin.into();
        self
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentTokenEngineLiveTextBatchRun {
    pub queued_input_events: Vec<VulkanResidentTokenEngineQueuedTextInputEvent>,
    pub cycles: Vec<VulkanResidentTokenEngineTextCycleRun>,
    pub output_events: Vec<VulkanResidentTokenEngineTextOutputEvent>,
    pub generated_token_ids: Vec<u32>,
    pub generated_text: String,
    pub stop_condition: VulkanResidentTokenEngineRunStopCondition,
    pub runtime_cycle_count: usize,
}

impl VulkanResidentTokenEngineLiveTextBatchRun {
    pub fn scheduler_turn_count(&self) -> usize {
        self.cycles.len()
    }

    pub fn generated_token_ids_for(&self, stream_id: &str, input_event_id: &str) -> Vec<u32> {
        self.output_events
            .iter()
            .filter(|event| event.stream_id == stream_id && event.input_event_id == input_event_id)
            .map(|event| event.token_id)
            .collect()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentTokenEngineQueuedTextInputEvent {
    pub stream_id: String,
    pub input_event_id: String,
    pub input_text: String,
    pub encoded_token_ids: Vec<u32>,
    pub queued_input_event: VulkanResidentTokenRuntimeQueuedInputEvent,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentTokenEngineTextCycleRun {
    pub scheduler_run: VulkanResidentTokenRuntimeSchedulerRun,
    pub output_events: Vec<VulkanResidentTokenEngineTextOutputEvent>,
    pub generated_token_ids: Vec<u32>,
    pub generated_text: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentTokenEngineTextOutputEvent {
    pub stream_id: String,
    pub input_event_id: String,
    pub output_index: usize,
    pub token_id: u32,
    pub text: String,
    pub source_stream_tick: u64,
}

pub trait VulkanResidentTokenTextCodec {
    fn encode_text(&self, text: &str) -> Result<Vec<u32>, VulkanResidentTokenTextCodecError>;

    fn decode_tokens(&self, token_ids: &[u32])
    -> Result<String, VulkanResidentTokenTextCodecError>;
}

#[cfg(feature = "tokenizers")]
pub struct VulkanResidentHfTokenizerTextCodec {
    tokenizer: tokenizers::Tokenizer,
    add_special_tokens: bool,
    skip_special_tokens: bool,
}

#[cfg(feature = "tokenizers")]
pub struct VulkanResidentHfTokenDecodeStream<'a> {
    stream: tokenizers::DecodeStream<
        'a,
        tokenizers::models::ModelWrapper,
        tokenizers::normalizers::NormalizerWrapper,
        tokenizers::pre_tokenizers::PreTokenizerWrapper,
        tokenizers::processors::PostProcessorWrapper,
        tokenizers::decoders::DecoderWrapper,
    >,
}

#[cfg(feature = "tokenizers")]
impl VulkanResidentHfTokenDecodeStream<'_> {
    pub fn step(
        &mut self,
        token_id: u32,
    ) -> Result<Option<String>, VulkanResidentTokenTextCodecError> {
        self.stream.step(token_id).map_err(|error| {
            VulkanResidentTokenTextCodecError::new(format!(
                "failed to incrementally decode token id {token_id}: {error}"
            ))
        })
    }
}

#[cfg(feature = "tokenizers")]
impl VulkanResidentHfTokenizerTextCodec {
    pub fn from_model_dir(
        model_dir: impl AsRef<Path>,
    ) -> Result<Self, VulkanResidentTokenTextCodecError> {
        Self::from_file(model_dir.as_ref().join("tokenizer.json"))
    }

    pub fn from_file(
        tokenizer_path: impl AsRef<Path>,
    ) -> Result<Self, VulkanResidentTokenTextCodecError> {
        let tokenizer_path = tokenizer_path.as_ref();
        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path).map_err(|error| {
            VulkanResidentTokenTextCodecError::new(format!(
                "failed to load tokenizer file {:?}: {error}",
                tokenizer_path
            ))
        })?;
        Ok(Self {
            tokenizer,
            add_special_tokens: true,
            skip_special_tokens: true,
        })
    }

    pub fn with_add_special_tokens(mut self, add_special_tokens: bool) -> Self {
        self.add_special_tokens = add_special_tokens;
        self
    }

    pub fn with_skip_special_tokens(mut self, skip_special_tokens: bool) -> Self {
        self.skip_special_tokens = skip_special_tokens;
        self
    }

    pub fn add_special_tokens(&self) -> bool {
        self.add_special_tokens
    }

    pub fn skip_special_tokens(&self) -> bool {
        self.skip_special_tokens
    }

    pub fn decode_stream(&self) -> VulkanResidentHfTokenDecodeStream<'_> {
        VulkanResidentHfTokenDecodeStream {
            stream: self.tokenizer.decode_stream(self.skip_special_tokens),
        }
    }
}

#[cfg(feature = "tokenizers")]
impl VulkanResidentTokenTextCodec for VulkanResidentHfTokenizerTextCodec {
    fn encode_text(&self, text: &str) -> Result<Vec<u32>, VulkanResidentTokenTextCodecError> {
        let encoding = self
            .tokenizer
            .encode(text, self.add_special_tokens)
            .map_err(|error| {
                VulkanResidentTokenTextCodecError::new(format!(
                    "failed to encode text with tokenizer: {error}"
                ))
            })?;
        Ok(encoding.get_ids().to_vec())
    }

    fn decode_tokens(
        &self,
        token_ids: &[u32],
    ) -> Result<String, VulkanResidentTokenTextCodecError> {
        self.tokenizer
            .decode(token_ids, self.skip_special_tokens)
            .map_err(|error| {
                VulkanResidentTokenTextCodecError::new(format!(
                    "failed to decode token ids with tokenizer: {error}"
                ))
            })
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VulkanResidentTokenIdTextCodec;

impl VulkanResidentTokenTextCodec for VulkanResidentTokenIdTextCodec {
    fn encode_text(&self, text: &str) -> Result<Vec<u32>, VulkanResidentTokenTextCodecError> {
        let mut tokens = Vec::new();
        for fragment in text
            .split(|character: char| character == ',' || character.is_whitespace())
            .filter(|fragment| !fragment.is_empty())
        {
            tokens.push(fragment.parse::<u32>().map_err(|error| {
                VulkanResidentTokenTextCodecError::new(format!(
                    "invalid numeric token fragment {fragment:?}: {error}"
                ))
            })?);
        }

        if tokens.is_empty() {
            return Err(VulkanResidentTokenTextCodecError::new(
                "numeric token text must contain at least one token id",
            ));
        }

        Ok(tokens)
    }

    fn decode_tokens(
        &self,
        token_ids: &[u32],
    ) -> Result<String, VulkanResidentTokenTextCodecError> {
        Ok(token_ids
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(" "))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentTokenTextCodecError {
    message: String,
}

impl VulkanResidentTokenTextCodecError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl Display for VulkanResidentTokenTextCodecError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl Error for VulkanResidentTokenTextCodecError {}

#[derive(Debug)]
pub enum VulkanResidentTokenEngineError {
    Device(VulkanError),
    Build(VulkanResidentTokenModelPackageError),
    Scheduler(VulkanResidentTokenRuntimeSchedulerError),
    TextCodec(VulkanResidentTokenTextCodecError),
    DuplicateModel(String),
    UnknownModel(String),
    RunCycleCountOverflow,
    RunStalled,
}

impl Display for VulkanResidentTokenEngineError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Device(error) => Display::fmt(error, f),
            Self::Build(error) => Display::fmt(error, f),
            Self::Scheduler(error) => Display::fmt(error, f),
            Self::TextCodec(error) => Display::fmt(error, f),
            Self::DuplicateModel(model_id) => {
                write!(
                    f,
                    "resident token engine model {model_id:?} is already loaded"
                )
            }
            Self::UnknownModel(model_id) => {
                write!(f, "resident token engine model {model_id:?} is not loaded")
            }
            Self::RunCycleCountOverflow => {
                f.write_str("resident token engine run cycle count overflowed")
            }
            Self::RunStalled => f.write_str(
                "resident token engine run stalled while scheduler still had active runtimes",
            ),
        }
    }
}

impl Error for VulkanResidentTokenEngineError {}

impl From<VulkanError> for VulkanResidentTokenEngineError {
    fn from(error: VulkanError) -> Self {
        Self::Device(error)
    }
}

impl From<VulkanResidentTokenModelPackageError> for VulkanResidentTokenEngineError {
    fn from(error: VulkanResidentTokenModelPackageError) -> Self {
        Self::Build(error)
    }
}

impl From<VulkanResidentTokenRuntimeSchedulerError> for VulkanResidentTokenEngineError {
    fn from(error: VulkanResidentTokenRuntimeSchedulerError) -> Self {
        Self::Scheduler(error)
    }
}

impl From<VulkanResidentTokenTextCodecError> for VulkanResidentTokenEngineError {
    fn from(error: VulkanResidentTokenTextCodecError) -> Self {
        Self::TextCodec(error)
    }
}

