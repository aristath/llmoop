#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanPlacedStreamCircuitResidentPlan {
    pub backend_id: String,
    pub device_id: String,
    pub hosted_component_ids: Vec<String>,
    pub signal_element_bytes: Option<usize>,
    pub local_edges: Vec<ComponentEdgePlacement>,
    pub incoming_edges: Vec<ComponentEdgePlacement>,
    pub outgoing_edges: Vec<ComponentEdgePlacement>,
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
        let hosted_component_ids = placement_plan
            .components
            .iter()
            .filter(|component| component.device_id == device_id)
            .map(|component| component.component_id.clone())
            .collect::<Vec<_>>();
        let hosted_component_set = hosted_component_ids.iter().cloned().collect::<BTreeSet<_>>();
        let resident_plan = VulkanStreamCircuitResidentPlan::from_resource_plan_with_hosted_components(
            resource_plan,
            Some(&hosted_component_set),
            tensor_index,
            activation_element_bytes,
        )?;
        let local_edges = placement_plan
            .edges
            .iter()
            .filter(|edge| {
                edge.connection.is_forward()
                    && edge.source_device_id == device_id
                    && edge.destination_device_id == device_id
            })
            .cloned()
            .collect();
        let incoming_edges = placement_plan
            .edges
            .iter()
            .filter(|edge| {
                edge.connection.is_forward()
                    && edge.source_device_id != device_id
                    && edge.destination_device_id == device_id
            })
            .cloned()
            .collect();
        let outgoing_edges = placement_plan
            .edges
            .iter()
            .filter(|edge| {
                edge.connection.is_forward()
                    && edge.source_device_id == device_id
                    && edge.destination_device_id != device_id
            })
            .cloned()
            .collect();

        Ok(Self {
            backend_id: VULKAN_STREAM_CIRCUIT_BACKEND_ID.to_string(),
            device_id,
            hosted_component_ids,
            signal_element_bytes: activation_element_bytes,
            local_edges,
            incoming_edges,
            outgoing_edges,
            resident_plan,
        })
    }

    pub fn hosts_component(&self, component_id: &str) -> bool {
        self.hosted_component_ids
            .iter()
            .any(|hosted| hosted == component_id)
    }
}
