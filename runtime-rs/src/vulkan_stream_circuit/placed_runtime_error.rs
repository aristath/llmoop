#[derive(Debug)]
pub enum VulkanResidentInProcessPlacedRuntimeError {
    ZeroTickBudget,
    EmptyPromptEvent,
    PromptStreamBusy,
    MissingPrivateFeedback,
    MissingFusedSamplerRun,
    MissingBoundDevice { device_id: String },
    StreamTickOverflow,
    FeedbackDepthOverflow,
    IncompleteTick(VulkanMountedPlacedResidentInProcessStreamTickRunStatus),
    Package(VulkanResidentTokenModelPackageError),
    BoundDispatchPlan(VulkanBoundDispatchPlanError),
    ResidentDispatch(VulkanMountedPlacedResidentKernelDispatchError),
    Schedule(VulkanError),
    Tick(VulkanMountedPlacedResidentInProcessStreamTickError),
    InputTransducer(VulkanResidentInputEmbeddingTransducerRunnerError),
    OutputTransducer(VulkanResidentOutputTransducerRunnerError),
    Sampler(VulkanResidentSamplerRunnerError),
    FeedbackEdge(VulkanError),
    BackendLoop(VulkanError),
}

impl Display for VulkanResidentInProcessPlacedRuntimeError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroTickBudget => {
                f.write_str("placed feedback loop tick budget must not be zero")
            }
            Self::EmptyPromptEvent => f.write_str("placed prompt event must not be empty"),
            Self::PromptStreamBusy => {
                f.write_str("placed prompt stream already has active or queued input")
            }
            Self::MissingPrivateFeedback => {
                f.write_str("placed feedback loop is missing private feedback")
            }
            Self::MissingFusedSamplerRun => {
                f.write_str("placed token tick completed without its fused sampler result")
            }
            Self::MissingBoundDevice { device_id } => write!(
                f,
                "placed model package has no runtime device bound for logical device {device_id:?}"
            ),
            Self::StreamTickOverflow => f.write_str("placed feedback loop tick overflowed"),
            Self::FeedbackDepthOverflow => f.write_str("placed feedback depth overflowed"),
            Self::IncompleteTick(status) => write!(
                f,
                "placed stream tick did not complete before output sampling: {status:?}"
            ),
            Self::Package(error) => Display::fmt(error, f),
            Self::BoundDispatchPlan(error) => Display::fmt(error, f),
            Self::ResidentDispatch(error) => Display::fmt(error, f),
            Self::Schedule(error) => Display::fmt(error, f),
            Self::Tick(error) => Display::fmt(error, f),
            Self::InputTransducer(error) => Display::fmt(error, f),
            Self::OutputTransducer(error) => Display::fmt(error, f),
            Self::Sampler(error) => Display::fmt(error, f),
            Self::FeedbackEdge(error) => Display::fmt(error, f),
            Self::BackendLoop(error) => Display::fmt(error, f),
        }
    }
}

impl Error for VulkanResidentInProcessPlacedRuntimeError {}

impl From<VulkanResidentTokenModelPackageError> for VulkanResidentInProcessPlacedRuntimeError {
    fn from(error: VulkanResidentTokenModelPackageError) -> Self {
        Self::Package(error)
    }
}

impl From<VulkanBoundDispatchPlanError> for VulkanResidentInProcessPlacedRuntimeError {
    fn from(error: VulkanBoundDispatchPlanError) -> Self {
        Self::BoundDispatchPlan(error)
    }
}

impl From<VulkanMountedPlacedResidentInProcessStreamTickError>
    for VulkanResidentInProcessPlacedRuntimeError
{
    fn from(error: VulkanMountedPlacedResidentInProcessStreamTickError) -> Self {
        Self::Tick(error)
    }
}

impl From<VulkanResidentInputEmbeddingTransducerRunnerError>
    for VulkanResidentInProcessPlacedRuntimeError
{
    fn from(error: VulkanResidentInputEmbeddingTransducerRunnerError) -> Self {
        Self::InputTransducer(error)
    }
}

impl From<VulkanResidentOutputTransducerRunnerError> for VulkanResidentInProcessPlacedRuntimeError {
    fn from(error: VulkanResidentOutputTransducerRunnerError) -> Self {
        Self::OutputTransducer(error)
    }
}

impl From<VulkanResidentSamplerRunnerError> for VulkanResidentInProcessPlacedRuntimeError {
    fn from(error: VulkanResidentSamplerRunnerError) -> Self {
        Self::Sampler(error)
    }
}

