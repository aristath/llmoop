#[derive(Debug)]
pub enum VulkanMountedPlacedStreamTickError {
    BoundDispatchPlan(VulkanBoundDispatchPlanError),
}

impl Display for VulkanMountedPlacedStreamTickError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BoundDispatchPlan(error) => Display::fmt(error, f),
        }
    }
}

impl Error for VulkanMountedPlacedStreamTickError {}

impl From<VulkanBoundDispatchPlanError> for VulkanMountedPlacedStreamTickError {
    fn from(error: VulkanBoundDispatchPlanError) -> Self {
        Self::BoundDispatchPlan(error)
    }
}

#[derive(Debug)]
pub enum VulkanMountedPlacedStreamTickTransportError {
    DeviceMismatch {
        plan_device_id: String,
        mounted_device_id: String,
    },
    Transport(VulkanPlacedEdgeTransportError),
}

impl Display for VulkanMountedPlacedStreamTickTransportError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DeviceMismatch {
                plan_device_id,
                mounted_device_id,
            } => write!(
                f,
                "stream tick plan for device {plan_device_id:?} cannot advance mounted device {mounted_device_id:?}"
            ),
            Self::Transport(error) => Display::fmt(error, f),
        }
    }
}

impl Error for VulkanMountedPlacedStreamTickTransportError {}
