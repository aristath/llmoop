#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanDistributedParameterExclusionPlan {
    pub devices: Vec<VulkanDistributedDeviceParameterExclusions>,
    pub device_count: usize,
    pub unique_tensor_count: usize,
    pub excluded_full_allocation_count: usize,
    pub excluded_full_byte_capacity: usize,
}

impl VulkanDistributedParameterExclusionPlan {
    pub fn from_execution_and_prepared_plans(
        execution_plan: &VulkanDistributedExecutionPlan,
        prepared_plans: &[(&str, &VulkanPreparedDispatchPlan)],
        tensor_index: &TensorIndex,
    ) -> Result<Self, VulkanDistributedPlanError> {
        let mut distributed_dispatch_tensors =
            BTreeMap::<VulkanDistributedDispatchKey, BTreeSet<String>>::new();
        let mut target_tensors = BTreeSet::<(String, String)>::new();
        for dispatch in &execution_plan.dispatches {
            let key = VulkanDistributedDispatchKey::from_distributed(dispatch);
            let tensors = dispatch
                .shards
                .iter()
                .flat_map(|shard| shard.parameters.iter())
                .map(|fragment| fragment.tensor.clone())
                .collect::<BTreeSet<_>>();
            if tensors.is_empty() {
                return Err(VulkanDistributedPlanError(format!(
                    "distributed dispatch {}.{} has no parameter tensors",
                    dispatch.pedal_id, dispatch.node_id
                )));
            }
            if distributed_dispatch_tensors
                .insert(key, tensors.clone())
                .is_some()
            {
                return Err(VulkanDistributedPlanError(format!(
                    "distributed execution plan repeats dispatch {}.{} at index {} on {:?}",
                    dispatch.pedal_id,
                    dispatch.node_id,
                    dispatch.dispatch_index,
                    dispatch.owner_device_id
                )));
            }
            target_tensors.extend(
                tensors
                    .into_iter()
                    .map(|tensor| (dispatch.owner_device_id.clone(), tensor)),
            );
        }

        let mut prepared_device_ids = BTreeSet::new();
        let mut matched_dispatches = BTreeSet::new();
        for (device_id, prepared_plan) in prepared_plans {
            if !prepared_device_ids.insert(*device_id) {
                return Err(VulkanDistributedPlanError(format!(
                    "distributed parameter exclusion repeats prepared plan for device {device_id:?}"
                )));
            }
            for dispatch in &prepared_plan.dispatches {
                let key = VulkanDistributedDispatchKey::from_prepared(device_id, dispatch);
                let parameter_tensors = dispatch
                    .descriptors
                    .iter()
                    .filter_map(|descriptor| match &descriptor.resource {
                        VulkanDescriptorResourceAddress::PermanentParameter { tensor, .. } => {
                            Some(tensor.clone())
                        }
                        _ => None,
                    })
                    .collect::<BTreeSet<_>>();
                if let Some(distributed_tensors) = distributed_dispatch_tensors.get(&key) {
                    if &parameter_tensors != distributed_tensors {
                        return Err(VulkanDistributedPlanError(format!(
                            "distributed dispatch {}.{} parameter tensors changed between preparation and physical lowering",
                            dispatch.pedal_id, dispatch.node_id
                        )));
                    }
                    matched_dispatches.insert(key);
                } else if let Some(tensor) = parameter_tensors.iter().find(|tensor| {
                    target_tensors.contains(&((*device_id).to_string(), (*tensor).clone()))
                }) {
                    return Err(VulkanDistributedPlanError(format!(
                        "cannot exclude distributed tensor {tensor:?} on {device_id:?}; canonical dispatch {}.{} still uses it",
                        dispatch.pedal_id, dispatch.node_id
                    )));
                }
            }
        }
        if let Some(missing) = distributed_dispatch_tensors
            .keys()
            .find(|key| !matched_dispatches.contains(*key))
        {
            return Err(VulkanDistributedPlanError(format!(
                "distributed dispatch {}.{} at index {} on {:?} is absent from prepared plans",
                missing.pedal_id, missing.node_id, missing.dispatch_index, missing.owner_device_id
            )));
        }

        let mut tensors_by_device = BTreeMap::<String, Vec<String>>::new();
        let mut excluded_full_byte_capacity = 0usize;
        for (device_id, tensor) in &target_tensors {
            let byte_count = tensor_index
                .tensors
                .get(tensor)
                .and_then(|metadata| metadata.byte_count)
                .ok_or_else(|| {
                    VulkanDistributedPlanError(format!(
                        "distributed exclusion tensor {tensor:?} has no byte count"
                    ))
                })?;
            excluded_full_byte_capacity = excluded_full_byte_capacity
                .checked_add(byte_count)
                .ok_or_else(|| {
                    VulkanDistributedPlanError(
                        "distributed exclusion byte capacity overflowed".to_string(),
                    )
                })?;
            tensors_by_device
                .entry(device_id.clone())
                .or_default()
                .push(tensor.clone());
        }
        let devices = tensors_by_device
            .into_iter()
            .map(|(device_id, tensors)| {
                let total_byte_capacity = tensors.iter().try_fold(0usize, |total, tensor| {
                    total
                        .checked_add(
                            tensor_index.tensors[tensor]
                                .byte_count
                                .expect("validated distributed exclusion byte count"),
                        )
                        .ok_or_else(|| {
                            VulkanDistributedPlanError(
                                "distributed device exclusion byte capacity overflowed".to_string(),
                            )
                        })
                })?;
                Ok(VulkanDistributedDeviceParameterExclusions {
                    device_id,
                    tensors,
                    total_byte_capacity,
                })
            })
            .collect::<Result<Vec<_>, VulkanDistributedPlanError>>()?;
        let unique_tensor_count = target_tensors
            .iter()
            .map(|(_, tensor)| tensor.as_str())
            .collect::<BTreeSet<_>>()
            .len();

        Ok(Self {
            device_count: devices.len(),
            devices,
            unique_tensor_count,
            excluded_full_allocation_count: target_tensors.len(),
            excluded_full_byte_capacity,
        })
    }

    pub fn tensors_for_device(&self, device_id: &str) -> BTreeSet<String> {
        self.devices
            .iter()
            .find(|device| device.device_id == device_id)
            .map(|device| device.tensors.iter().cloned().collect())
            .unwrap_or_default()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanDistributedDeviceParameterExclusions {
    pub device_id: String,
    pub tensors: Vec<String>,
    pub total_byte_capacity: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct VulkanDistributedDispatchKey {
    owner_device_id: String,
    dispatch_index: usize,
    pedal_id: String,
    node_id: String,
}

impl VulkanDistributedDispatchKey {
    fn from_distributed(dispatch: &VulkanDistributedDispatchPlan) -> Self {
        Self {
            owner_device_id: dispatch.owner_device_id.clone(),
            dispatch_index: dispatch.dispatch_index,
            pedal_id: dispatch.pedal_id.clone(),
            node_id: dispatch.node_id.clone(),
        }
    }

    fn from_prepared(owner_device_id: &str, dispatch: &VulkanPreparedDispatch) -> Self {
        Self {
            owner_device_id: owner_device_id.to_string(),
            dispatch_index: dispatch.dispatch_index,
            pedal_id: dispatch.pedal_id.clone(),
            node_id: dispatch.node_id.clone(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanDistributedParameterLoadReport {
    pub tensor_count: usize,
    pub source_file_count: usize,
    pub allocation_count: usize,
    pub write_count: usize,
    pub total_bytes_read: usize,
    pub total_bytes_written: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanDistributedParameterLoadError(pub String);

impl Display for VulkanDistributedParameterLoadError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for VulkanDistributedParameterLoadError {}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct VulkanDistributedParameterAllocationKey {
    device_id: String,
    tensor: String,
    byte_offset: usize,
    byte_count: usize,
}

impl From<&VulkanDistributedParameterAllocation> for VulkanDistributedParameterAllocationKey {
    fn from(allocation: &VulkanDistributedParameterAllocation) -> Self {
        Self {
            device_id: allocation.device_id.clone(),
            tensor: allocation.tensor.clone(),
            byte_offset: allocation.byte_offset,
            byte_count: allocation.byte_count,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanDistributedPlanError(pub String);

impl Display for VulkanDistributedPlanError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for VulkanDistributedPlanError {}

