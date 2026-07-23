#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanPermanentParameterBufferPlan {
    pub backend_id: String,
    pub device_id: String,
    pub parameters: Vec<VulkanPermanentParameterBuffer>,
    pub parameter_count: usize,
    pub total_byte_capacity: Option<usize>,
    pub unresolved_tensors: Vec<String>,
}

impl VulkanPermanentParameterBufferPlan {
    pub fn from_placed_resident_plan(
        placed_resident_plan: &VulkanPlacedStreamCircuitResidentPlan,
    ) -> Result<Self, VulkanPermanentParameterBufferPlanError> {
        Self::from_placed_resident_plan_excluding_tensors(placed_resident_plan, &BTreeSet::new())
    }

    pub fn from_placed_resident_plan_excluding_tensors(
        placed_resident_plan: &VulkanPlacedStreamCircuitResidentPlan,
        excluded_tensors: &BTreeSet<String>,
    ) -> Result<Self, VulkanPermanentParameterBufferPlanError> {
        let mut parameters = Vec::with_capacity(
            placed_resident_plan
                .resident_plan
                .permanent_parameters
                .len(),
        );
        let mut tensor_ids = BTreeSet::new();
        let mut found_excluded_tensors = BTreeSet::new();
        let mut total_byte_capacity = Some(0usize);
        let mut unresolved_tensors = Vec::new();

        for parameter in &placed_resident_plan.resident_plan.permanent_parameters {
            if !tensor_ids.insert(parameter.tensor.clone()) {
                return Err(VulkanPermanentParameterBufferPlanError(format!(
                    "{} permanent parameter tensor {:?} appears more than once",
                    placed_resident_plan.device_id, parameter.tensor
                )));
            }
            if excluded_tensors.contains(&parameter.tensor) {
                found_excluded_tensors.insert(parameter.tensor.clone());
                continue;
            }

            match (total_byte_capacity, parameter.byte_count) {
                (Some(total), Some(bytes)) => {
                    total_byte_capacity = Some(add_parameter_bytes(
                        total,
                        bytes,
                        "permanent parameter buffer plan",
                    )?);
                }
                _ => {
                    total_byte_capacity = None;
                    unresolved_tensors.push(parameter.tensor.clone());
                }
            }

            parameters.push(VulkanPermanentParameterBuffer {
                buffer_index: parameters.len(),
                tensor: parameter.tensor.clone(),
                dtype: parameter.dtype.clone(),
                shape: parameter.shape.clone(),
                byte_capacity: parameter.byte_count,
                use_count: parameter.use_count,
            });
        }

        if let Some(tensor) = excluded_tensors.difference(&found_excluded_tensors).next() {
            return Err(VulkanPermanentParameterBufferPlanError(format!(
                "{} cannot exclude unavailable permanent parameter tensor {tensor:?}",
                placed_resident_plan.device_id
            )));
        }

        let parameter_count = parameters.len();
        Ok(Self {
            backend_id: VULKAN_STREAM_CIRCUIT_BACKEND_ID.to_string(),
            device_id: placed_resident_plan.device_id.clone(),
            parameters,
            parameter_count,
            total_byte_capacity,
            unresolved_tensors,
        })
    }

    pub fn from_transducer_parameters(
        device_id: impl Into<String>,
        resource_plan: &StreamCircuitResourcePlan,
        tensor_index: Option<&TensorIndex>,
    ) -> Result<Self, VulkanPermanentParameterBufferPlanError> {
        Self::from_transducer_parameters_matching(device_id, resource_plan, tensor_index, |_| true)
    }

    pub fn from_transducer_parameters_for(
        device_id: impl Into<String>,
        resource_plan: &StreamCircuitResourcePlan,
        tensor_index: Option<&TensorIndex>,
        transducer_id: &str,
    ) -> Result<Self, VulkanPermanentParameterBufferPlanError> {
        Self::from_transducer_parameters_matching(
            device_id,
            resource_plan,
            tensor_index,
            |parameter| {
                parameter
                    .uses
                    .iter()
                    .any(|parameter_use| parameter_use.circuit_id == transducer_id)
            },
        )
    }

    fn from_transducer_parameters_matching<F>(
        device_id: impl Into<String>,
        resource_plan: &StreamCircuitResourcePlan,
        tensor_index: Option<&TensorIndex>,
        mut include: F,
    ) -> Result<Self, VulkanPermanentParameterBufferPlanError>
    where
        F: FnMut(&PlannedParameterResource) -> bool,
    {
        let device_id = device_id.into();
        let mut parameters = Vec::with_capacity(resource_plan.transducer_parameters.len());
        let mut tensor_ids = BTreeSet::new();
        let mut total_byte_capacity = Some(0usize);
        let mut unresolved_tensors = Vec::new();

        for parameter in &resource_plan.transducer_parameters {
            if !include(parameter) {
                continue;
            }
            if !tensor_ids.insert(parameter.tensor.clone()) {
                return Err(VulkanPermanentParameterBufferPlanError(format!(
                    "{device_id} transducer parameter tensor {:?} appears more than once",
                    parameter.tensor
                )));
            }

            let metadata = tensor_index.and_then(|index| index.tensors.get(&parameter.tensor));
            let byte_count = metadata.and_then(|metadata| metadata.byte_count);
            match (total_byte_capacity, byte_count) {
                (Some(total), Some(bytes)) => {
                    total_byte_capacity = Some(add_parameter_bytes(
                        total,
                        bytes,
                        "transducer parameter buffer plan",
                    )?);
                }
                _ => {
                    total_byte_capacity = None;
                    unresolved_tensors.push(parameter.tensor.clone());
                }
            }

            parameters.push(VulkanPermanentParameterBuffer {
                buffer_index: parameters.len(),
                tensor: parameter.tensor.clone(),
                dtype: metadata.map(|metadata| metadata.dtype.clone()),
                shape: metadata.map(|metadata| metadata.shape.clone()),
                byte_capacity: byte_count,
                use_count: parameter.uses.len(),
            });
        }

        let parameter_count = parameters.len();
        Ok(Self {
            backend_id: VULKAN_STREAM_CIRCUIT_BACKEND_ID.to_string(),
            device_id,
            parameters,
            parameter_count,
            total_byte_capacity,
            unresolved_tensors,
        })
    }

    pub fn allocate_buffers(
        &self,
        device: &VulkanComputeDevice,
    ) -> Result<VulkanPermanentParameterBuffers, VulkanError> {
        let mut buffers = Vec::with_capacity(self.parameters.len());
        let mut total_byte_capacity = 0usize;

        for parameter in &self.parameters {
            let byte_capacity = parameter.byte_capacity.ok_or_else(|| {
                VulkanError(format!(
                    "{} permanent parameter {:?} has unknown byte capacity",
                    self.device_id, parameter.tensor
                ))
            })?;
            total_byte_capacity = checked_add_bytes(
                total_byte_capacity,
                byte_capacity,
                "permanent parameter buffer allocation",
            )?;
            buffers.push(VulkanPermanentParameterBufferAllocation {
                parameter: parameter.clone(),
                byte_capacity,
                buffer: device.create_resident_buffer(byte_capacity)?,
            });
        }

        Ok(VulkanPermanentParameterBuffers {
            plan: self.clone(),
            buffers,
            total_byte_capacity,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanPermanentParameterBuffer {
    pub buffer_index: usize,
    pub tensor: String,
    pub dtype: Option<String>,
    pub shape: Option<Vec<usize>>,
    pub byte_capacity: Option<usize>,
    pub use_count: usize,
}

pub struct VulkanPermanentParameterBuffers {
    pub plan: VulkanPermanentParameterBufferPlan,
    pub buffers: Vec<VulkanPermanentParameterBufferAllocation>,
    pub total_byte_capacity: usize,
}

impl VulkanPermanentParameterBuffers {
    pub fn parameter_buffer(
        &self,
        tensor: &str,
    ) -> Option<&VulkanPermanentParameterBufferAllocation> {
        self.buffers
            .iter()
            .find(|buffer| buffer.parameter.tensor == tensor)
    }

    pub fn load_parameter_from_tensor_index(
        &self,
        tensor_index: &TensorIndex,
        tensor: &str,
    ) -> Result<VulkanPermanentParameterLoadRecord, VulkanPermanentParameterLoadError> {
        let allocation = self.parameter_buffer(tensor).ok_or_else(|| {
            VulkanPermanentParameterLoadError(format!(
                "mounted parameter buffer for tensor {tensor:?} is missing"
            ))
        })?;
        load_parameter_allocation_from_tensor_index(allocation, tensor_index)
    }

    pub fn load_from_tensor_index(
        &self,
        tensor_index: &TensorIndex,
    ) -> Result<VulkanPermanentParameterLoadReport, VulkanPermanentParameterLoadError> {
        let mut records = Vec::with_capacity(self.buffers.len());
        let mut total_bytes_loaded = 0usize;
        let mut source_files = BTreeSet::new();

        for allocation in &self.buffers {
            let record = load_parameter_allocation_from_tensor_index(allocation, tensor_index)?;
            total_bytes_loaded = total_bytes_loaded
                .checked_add(record.byte_count)
                .ok_or_else(|| {
                    VulkanPermanentParameterLoadError(
                        "permanent parameter loaded byte count overflowed".to_string(),
                    )
                })?;
            source_files.insert(record.source_file.clone());
            records.push(record);
        }

        Ok(VulkanPermanentParameterLoadReport {
            parameter_count: self.buffers.len(),
            loaded_count: records.len(),
            total_bytes_loaded,
            source_file_count: source_files.len(),
            records,
        })
    }
}

pub struct VulkanPermanentParameterBufferAllocation {
    pub parameter: VulkanPermanentParameterBuffer,
    pub byte_capacity: usize,
    pub buffer: VulkanResidentBuffer,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanPermanentParameterLoadReport {
    pub parameter_count: usize,
    pub loaded_count: usize,
    pub total_bytes_loaded: usize,
    pub source_file_count: usize,
    pub records: Vec<VulkanPermanentParameterLoadRecord>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanPermanentParameterLoadRecord {
    pub tensor: String,
    pub buffer_index: usize,
    pub source_file: String,
    pub data_start: usize,
    pub data_end: usize,
    pub byte_count: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanPermanentParameterLoadError(pub String);

impl Display for VulkanPermanentParameterLoadError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for VulkanPermanentParameterLoadError {}

fn load_parameter_allocation_from_tensor_index(
    allocation: &VulkanPermanentParameterBufferAllocation,
    tensor_index: &TensorIndex,
) -> Result<VulkanPermanentParameterLoadRecord, VulkanPermanentParameterLoadError> {
    let tensor = &allocation.parameter.tensor;
    let storage = TensorStorage::from_index(tensor_index, tensor).map_err(|error| {
        VulkanPermanentParameterLoadError(format!(
            "failed to resolve mounted parameter tensor {tensor:?}: {error}"
        ))
    })?;
    if storage.byte_count != allocation.byte_capacity {
        return Err(VulkanPermanentParameterLoadError(format!(
            "tensor {tensor:?} byte count {} does not match mounted buffer capacity {}",
            storage.byte_count, allocation.byte_capacity
        )));
    }
    let bytes = storage
        .read_all()
        .map_err(|error| VulkanPermanentParameterLoadError(error.to_string()))?;
    allocation.buffer.write_bytes(&bytes)?;

    Ok(VulkanPermanentParameterLoadRecord {
        tensor: tensor.clone(),
        buffer_index: allocation.parameter.buffer_index,
        source_file: storage.source_file.to_string_lossy().into_owned(),
        data_start: storage.data_start,
        data_end: storage.data_end,
        byte_count: storage.byte_count,
    })
}

impl From<VulkanError> for VulkanPermanentParameterLoadError {
    fn from(error: VulkanError) -> Self {
        Self(error.to_string())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanPermanentParameterBufferPlanError(pub String);

impl Display for VulkanPermanentParameterBufferPlanError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for VulkanPermanentParameterBufferPlanError {}

fn add_parameter_bytes(
    total: usize,
    bytes: usize,
    label: &str,
) -> Result<usize, VulkanPermanentParameterBufferPlanError> {
    total
        .checked_add(bytes)
        .ok_or_else(|| VulkanPermanentParameterBufferPlanError(format!("{label} overflowed")))
}
