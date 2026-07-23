#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentPlanError(pub String);

impl Display for VulkanResidentPlanError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for VulkanResidentPlanError {}

fn optional_mul(
    elements: Option<usize>,
    bytes_per_element: Option<usize>,
) -> Result<Option<usize>, VulkanResidentPlanError> {
    match (elements, bytes_per_element) {
        (Some(elements), Some(bytes_per_element)) => Ok(Some(checked_mul(
            elements,
            bytes_per_element,
            "resident byte count",
        )?)),
        _ => Ok(None),
    }
}

fn optional_add(
    left: Option<usize>,
    right: Option<usize>,
    label: &str,
) -> Result<Option<usize>, VulkanResidentPlanError> {
    match (left, right) {
        (Some(left), Some(right)) => Ok(Some(checked_add(left, right, label)?)),
        _ => Ok(None),
    }
}

fn optional_state_contribution_bytes(
    elements: Option<usize>,
    bytes_per_element: Option<usize>,
) -> Result<Option<usize>, VulkanResidentPlanError> {
    match elements {
        Some(elements) => optional_mul(Some(elements), bytes_per_element),
        None => Ok(Some(0)),
    }
}

fn stream_state_byte_capacity(
    state: &VulkanResidentStateBuffer,
    dynamic_state_capacity_activations: usize,
) -> Result<usize, VulkanError> {
    let static_bytes = state.static_bytes.unwrap_or(0);
    let dynamic_bytes = match state.bytes_per_activation {
        Some(bytes_per_activation) => {
            if dynamic_state_capacity_activations == 0 {
                return Err(VulkanError(format!(
                    "{}.{} requires non-zero dynamic state capacity",
                    state.component_id, state.state_id
                )));
            }
            let state_capacity = state
                .max_dynamic_activations
                .map(|limit| limit.min(dynamic_state_capacity_activations))
                .unwrap_or(dynamic_state_capacity_activations);
            bytes_per_activation
                .checked_mul(state_capacity)
                .ok_or_else(|| {
                    VulkanError(format!(
                        "{}.{} dynamic state byte capacity overflowed",
                        state.component_id, state.state_id
                    ))
                })?
        }
        None => 0,
    };
    let total = static_bytes.checked_add(dynamic_bytes).ok_or_else(|| {
        VulkanError(format!(
            "{}.{} state byte capacity overflowed",
            state.component_id, state.state_id
        ))
    })?;
    if total == 0 {
        return Err(VulkanError(format!(
            "{}.{} has unknown or zero byte capacity",
            state.component_id, state.state_id
        )));
    }
    Ok(total)
}

fn checked_add_bytes(left: usize, right: usize, label: &str) -> Result<usize, VulkanError> {
    left.checked_add(right)
        .ok_or_else(|| VulkanError(format!("{label} overflowed")))
}

fn product(shape: &[usize]) -> Option<usize> {
    shape
        .iter()
        .try_fold(1usize, |total, value| total.checked_mul(*value))
}

fn checked_add(left: usize, right: usize, label: &str) -> Result<usize, VulkanResidentPlanError> {
    left.checked_add(right)
        .ok_or_else(|| VulkanResidentPlanError(format!("{label} overflowed")))
}

fn checked_mul(left: usize, right: usize, label: &str) -> Result<usize, VulkanResidentPlanError> {
    left.checked_mul(right)
        .ok_or_else(|| VulkanResidentPlanError(format!("{label} overflowed")))
}

#[cfg(test)]
mod tests;
