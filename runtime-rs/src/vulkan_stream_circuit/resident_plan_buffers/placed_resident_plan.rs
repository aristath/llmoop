#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanPlacedStreamCircuitResidentPlan {
    pub backend_id: String,
    pub device_id: String,
    pub hosted_pedal_ids: Vec<String>,
    pub signal_element_bytes: Option<usize>,
    pub local_cables: Vec<PedalCablePlacement>,
    pub incoming_cables: Vec<PedalCablePlacement>,
    pub outgoing_cables: Vec<PedalCablePlacement>,
    pub resident_plan: VulkanStreamCircuitResidentPlan,
}

impl VulkanPlacedStreamCircuitResidentPlan {
    pub fn from_resource_plan_for_device(
        resource_plan: &StreamCircuitResourcePlan,
        placement_plan: &StreamCircuitPlacementPlan,
        device_id: impl Into<String>,
        tensor_index: Option<&TensorIndex>,
        activation_element_bytes: Option<usize>,
    ) -> Result<Self, VulkanResidentPlanError> {
        let device_id = device_id.into();
        if device_id.is_empty() {
            return Err(VulkanResidentPlanError(
                "Vulkan placed resident plan device_id must not be empty".to_string(),
            ));
        }
        let hosted_pedal_ids = placement_plan
            .pedals
            .iter()
            .filter(|pedal| pedal.device_id == device_id)
            .map(|pedal| pedal.pedal_id.clone())
            .collect::<Vec<_>>();
        let hosted_pedal_set = hosted_pedal_ids.iter().cloned().collect::<BTreeSet<_>>();
        let resident_plan = VulkanStreamCircuitResidentPlan::from_resource_plan_with_hosted_pedals(
            resource_plan,
            Some(&hosted_pedal_set),
            tensor_index,
            activation_element_bytes,
        )?;
        let local_cables = placement_plan
            .cables
            .iter()
            .filter(|cable| {
                cable.connection.is_forward()
                    && cable.source_device_id == device_id
                    && cable.destination_device_id == device_id
            })
            .cloned()
            .collect();
        let incoming_cables = placement_plan
            .cables
            .iter()
            .filter(|cable| {
                cable.connection.is_forward()
                    && cable.source_device_id != device_id
                    && cable.destination_device_id == device_id
            })
            .cloned()
            .collect();
        let outgoing_cables = placement_plan
            .cables
            .iter()
            .filter(|cable| {
                cable.connection.is_forward()
                    && cable.source_device_id == device_id
                    && cable.destination_device_id != device_id
            })
            .cloned()
            .collect();

        Ok(Self {
            backend_id: VULKAN_STREAM_CIRCUIT_BACKEND_ID.to_string(),
            device_id,
            hosted_pedal_ids,
            signal_element_bytes: activation_element_bytes,
            local_cables,
            incoming_cables,
            outgoing_cables,
            resident_plan,
        })
    }

    pub fn hosts_pedal(&self, pedal_id: &str) -> bool {
        self.hosted_pedal_ids
            .iter()
            .any(|hosted| hosted == pedal_id)
    }
}
