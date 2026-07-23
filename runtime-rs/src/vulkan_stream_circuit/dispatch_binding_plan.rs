#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanPreparedDispatchPlan {
    pub backend_id: String,
    pub reusable_family_count: usize,
    pub dispatches: Vec<VulkanPreparedDispatch>,
    pub total_descriptor_count: usize,
}

impl VulkanPreparedDispatchPlan {
    pub fn from_plans(
        dispatch_plan: &VulkanKernelDispatchPlan,
        reusable_plan: &VulkanReusableKernelPlan,
        descriptor_plan: &VulkanDescriptorResourcePlan,
        manifest: &VulkanReusableKernelArtifactManifest,
    ) -> Result<Self, VulkanPreparedDispatchPlanError> {
        let link_plan = reusable_plan.link_artifacts(manifest);
        if !link_plan.is_fully_linked() {
            return Err(VulkanPreparedDispatchPlanError::Link(Box::new(link_plan)));
        }

        let descriptor_by_dispatch: BTreeMap<usize, &VulkanDispatchDescriptorResourcePlan> =
            descriptor_plan
                .dispatches
                .iter()
                .map(|dispatch| (dispatch.dispatch_index, dispatch))
                .collect();
        let mut family_by_dispatch = BTreeMap::new();
        for family in &reusable_plan.families {
            for command_ref in &family.command_refs {
                family_by_dispatch.insert(command_ref.dispatch_index, family);
            }
        }
        let artifact_by_family: BTreeMap<_, _> = manifest
            .artifacts
            .iter()
            .map(|artifact| (artifact.family_id.as_str(), artifact))
            .collect();

        let mut dispatches = Vec::with_capacity(dispatch_plan.commands.len());
        for command in &dispatch_plan.commands {
            let descriptor_dispatch = descriptor_by_dispatch.get(&command.dispatch_index).ok_or(
                VulkanPreparedDispatchPlanError::MissingDescriptorResources {
                    dispatch_index: command.dispatch_index,
                },
            )?;
            let family = family_by_dispatch.get(&command.dispatch_index).ok_or(
                VulkanPreparedDispatchPlanError::MissingReusableFamily {
                    dispatch_index: command.dispatch_index,
                },
            )?;
            let artifact = artifact_by_family
                .get(family.family_id.as_str())
                .ok_or_else(|| VulkanPreparedDispatchPlanError::MissingLinkedArtifact {
                    family_id: family.family_id.clone(),
                })?;

            dispatches.push(VulkanPreparedDispatch {
                dispatch_index: command.dispatch_index,
                kernel_id: command.kernel_id.clone(),
                pedal_id: command.pedal_id.clone(),
                circuit_id: command.circuit_id.clone(),
                node_index: command.node_index,
                node_id: command.node_id.clone(),
                op: command.op.clone(),
                reusable_family_id: family.family_id.clone(),
                artifact_path: artifact.path.clone(),
                entry_point: artifact.entry_point.clone(),
                local_size_x: artifact.local_size_x,
                descriptors: descriptor_dispatch.descriptors.clone(),
                push_constants: command.push_constants.clone(),
                uses_stream_tick: command.uses_stream_tick,
            });
        }
        let total_descriptor_count = dispatches
            .iter()
            .map(|dispatch| dispatch.descriptors.len())
            .sum();

        Ok(Self {
            backend_id: VULKAN_STREAM_CIRCUIT_BACKEND_ID.to_string(),
            reusable_family_count: reusable_plan.total_family_count(),
            dispatches,
            total_descriptor_count,
        })
    }

    pub fn dispatch(&self, pedal_id: &str, node_id: &str) -> Option<&VulkanPreparedDispatch> {
        self.dispatches
            .iter()
            .find(|dispatch| dispatch.pedal_id == pedal_id && dispatch.node_id == node_id)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanPreparedDispatch {
    pub dispatch_index: usize,
    pub kernel_id: String,
    pub pedal_id: String,
    pub circuit_id: String,
    pub node_index: usize,
    pub node_id: String,
    pub op: String,
    pub reusable_family_id: String,
    pub artifact_path: String,
    pub entry_point: String,
    pub local_size_x: u32,
    pub descriptors: Vec<VulkanResolvedDescriptorBinding>,
    pub push_constants: Vec<VulkanKernelScalarBinding>,
    pub uses_stream_tick: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VulkanPreparedDispatchPlanError {
    DescriptorResource(VulkanDescriptorResourcePlanError),
    Link(Box<VulkanLinkedReusableKernelPlan>),
    MissingDescriptorResources { dispatch_index: usize },
    MissingReusableFamily { dispatch_index: usize },
    MissingLinkedArtifact { family_id: String },
}

impl Display for VulkanPreparedDispatchPlanError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DescriptorResource(error) => Display::fmt(error, f),
            Self::Link(plan) => write!(
                f,
                "reusable Vulkan kernels are not fully linked: {} missing families, {} incompatible families",
                plan.missing_family_count, plan.incompatible_family_count
            ),
            Self::MissingDescriptorResources { dispatch_index } => write!(
                f,
                "dispatch {dispatch_index} has no resolved descriptor resources"
            ),
            Self::MissingReusableFamily { dispatch_index } => {
                write!(f, "dispatch {dispatch_index} has no reusable kernel family")
            }
            Self::MissingLinkedArtifact { family_id } => {
                write!(f, "reusable family {family_id:?} has no linked artifact")
            }
        }
    }
}

impl Error for VulkanPreparedDispatchPlanError {}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanBoundDispatchPlan {
    pub backend_id: String,
    pub dispatches: Vec<VulkanBoundDispatch>,
    pub total_descriptor_count: usize,
    pub boundary_descriptor_count: usize,
    pub permanent_parameter_descriptor_count: usize,
    pub stream_state_descriptor_count: usize,
    pub activation_slot_descriptor_count: usize,
}

impl VulkanBoundDispatchPlan {
    pub fn from_prepared_plan(
        prepared_plan: &VulkanPreparedDispatchPlan,
        buffers: &VulkanStreamCircuitStreamBuffers,
    ) -> Result<Self, VulkanBoundDispatchPlanError> {
        let mut boundary_descriptor_count = 0usize;
        let mut permanent_parameter_descriptor_count = 0usize;
        let mut stream_state_descriptor_count = 0usize;
        let mut activation_slot_descriptor_count = 0usize;
        let mut dispatches = Vec::with_capacity(prepared_plan.dispatches.len());

        for prepared in &prepared_plan.dispatches {
            let mut descriptors = Vec::with_capacity(prepared.descriptors.len());
            for descriptor in &prepared.descriptors {
                let target =
                    VulkanBoundDescriptorTarget::from_resource(prepared, descriptor, buffers)?;
                match target {
                    VulkanBoundDescriptorTarget::BoundaryInput { .. }
                    | VulkanBoundDescriptorTarget::BoundaryOutput { .. } => {
                        boundary_descriptor_count += 1;
                    }
                    VulkanBoundDescriptorTarget::PermanentParameter { .. } => {
                        permanent_parameter_descriptor_count += 1;
                    }
                    VulkanBoundDescriptorTarget::StreamStateBuffer { .. }
                    | VulkanBoundDescriptorTarget::StreamStateView { .. } => {
                        stream_state_descriptor_count += 1;
                    }
                    VulkanBoundDescriptorTarget::ActivationSlot { .. } => {
                        activation_slot_descriptor_count += 1;
                    }
                }
                descriptors.push(VulkanBoundDescriptor {
                    binding: descriptor.binding,
                    usage: descriptor.usage.clone(),
                    name: descriptor.name.clone(),
                    target,
                });
            }

            dispatches.push(VulkanBoundDispatch {
                dispatch_index: prepared.dispatch_index,
                kernel_id: prepared.kernel_id.clone(),
                pedal_id: prepared.pedal_id.clone(),
                circuit_id: prepared.circuit_id.clone(),
                node_index: prepared.node_index,
                node_id: prepared.node_id.clone(),
                op: prepared.op.clone(),
                reusable_family_id: prepared.reusable_family_id.clone(),
                artifact_path: prepared.artifact_path.clone(),
                entry_point: prepared.entry_point.clone(),
                local_size_x: prepared.local_size_x,
                descriptors,
                push_constants: prepared.push_constants.clone(),
                uses_stream_tick: prepared.uses_stream_tick,
            });
        }

        let total_descriptor_count = boundary_descriptor_count
            + permanent_parameter_descriptor_count
            + stream_state_descriptor_count
            + activation_slot_descriptor_count;
        Ok(Self {
            backend_id: prepared_plan.backend_id.clone(),
            dispatches,
            total_descriptor_count,
            boundary_descriptor_count,
            permanent_parameter_descriptor_count,
            stream_state_descriptor_count,
            activation_slot_descriptor_count,
        })
    }

    pub fn dispatch(&self, pedal_id: &str, node_id: &str) -> Option<&VulkanBoundDispatch> {
        self.dispatches
            .iter()
            .find(|dispatch| dispatch.pedal_id == pedal_id && dispatch.node_id == node_id)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanPlacedBoundDispatchPlan {
    pub backend_id: String,
    pub device_id: String,
    pub dispatches: Vec<VulkanPlacedBoundDispatch>,
    pub total_descriptor_count: usize,
    pub resident_descriptor_count: usize,
    pub model_boundary_descriptor_count: usize,
    pub local_cable_descriptor_count: usize,
    pub incoming_cable_descriptor_count: usize,
    pub outgoing_cable_descriptor_count: usize,
}

impl VulkanPlacedBoundDispatchPlan {
    pub fn from_bound_plan(
        bound_plan: &VulkanBoundDispatchPlan,
        placed_resident_plan: &VulkanPlacedStreamCircuitResidentPlan,
    ) -> Self {
        let mut resident_descriptor_count = 0usize;
        let mut model_boundary_descriptor_count = 0usize;
        let mut local_cable_descriptor_count = 0usize;
        let mut incoming_cable_descriptor_count = 0usize;
        let mut outgoing_cable_descriptor_count = 0usize;
        let mut dispatches = Vec::with_capacity(bound_plan.dispatches.len());

        for dispatch in &bound_plan.dispatches {
            let mut descriptors = Vec::with_capacity(dispatch.descriptors.len());
            for descriptor in &dispatch.descriptors {
                let target = VulkanPlacedBoundDescriptorTarget::from_bound_target(
                    &dispatch.pedal_id,
                    &descriptor.target,
                    placed_resident_plan,
                );
                match target {
                    VulkanPlacedBoundDescriptorTarget::Resident { .. } => {
                        resident_descriptor_count += 1;
                    }
                    VulkanPlacedBoundDescriptorTarget::ModelInput { .. }
                    | VulkanPlacedBoundDescriptorTarget::ModelOutput { .. } => {
                        model_boundary_descriptor_count += 1;
                    }
                    VulkanPlacedBoundDescriptorTarget::LocalCableInput { .. }
                    | VulkanPlacedBoundDescriptorTarget::LocalCableOutput { .. } => {
                        local_cable_descriptor_count += 1;
                    }
                    VulkanPlacedBoundDescriptorTarget::IncomingCable { .. } => {
                        incoming_cable_descriptor_count += 1;
                    }
                    VulkanPlacedBoundDescriptorTarget::OutgoingCable { .. } => {
                        outgoing_cable_descriptor_count += 1;
                    }
                }
                descriptors.push(VulkanPlacedBoundDescriptor {
                    binding: descriptor.binding,
                    usage: descriptor.usage.clone(),
                    name: descriptor.name.clone(),
                    target,
                });
            }

            dispatches.push(VulkanPlacedBoundDispatch {
                dispatch_index: dispatch.dispatch_index,
                kernel_id: dispatch.kernel_id.clone(),
                pedal_id: dispatch.pedal_id.clone(),
                circuit_id: dispatch.circuit_id.clone(),
                node_index: dispatch.node_index,
                node_id: dispatch.node_id.clone(),
                op: dispatch.op.clone(),
                reusable_family_id: dispatch.reusable_family_id.clone(),
                artifact_path: dispatch.artifact_path.clone(),
                entry_point: dispatch.entry_point.clone(),
                local_size_x: dispatch.local_size_x,
                descriptors,
                push_constants: dispatch.push_constants.clone(),
                uses_stream_tick: dispatch.uses_stream_tick,
            });
        }

        let total_descriptor_count = resident_descriptor_count
            + model_boundary_descriptor_count
            + local_cable_descriptor_count
            + incoming_cable_descriptor_count
            + outgoing_cable_descriptor_count;

        Self {
            backend_id: bound_plan.backend_id.clone(),
            device_id: placed_resident_plan.device_id.clone(),
            dispatches,
            total_descriptor_count,
            resident_descriptor_count,
            model_boundary_descriptor_count,
            local_cable_descriptor_count,
            incoming_cable_descriptor_count,
            outgoing_cable_descriptor_count,
        }
    }

    pub fn dispatch(&self, pedal_id: &str, node_id: &str) -> Option<&VulkanPlacedBoundDispatch> {
        self.dispatches
            .iter()
            .find(|dispatch| dispatch.pedal_id == pedal_id && dispatch.node_id == node_id)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanMountedPlacedBoundDispatchPlan {
    pub backend_id: String,
    pub device_id: String,
    pub dispatches: Vec<VulkanMountedPlacedBoundDispatch>,
    pub total_descriptor_count: usize,
    pub resident_descriptor_count: usize,
    pub model_boundary_descriptor_count: usize,
    pub local_cable_descriptor_count: usize,
    pub cable_endpoint_descriptor_count: usize,
    pub incoming_cable_descriptor_count: usize,
    pub outgoing_cable_descriptor_count: usize,
}

impl VulkanMountedPlacedBoundDispatchPlan {
    pub fn from_placed_bound_plan(
        placed_bound_plan: &VulkanPlacedBoundDispatchPlan,
        cable_io: &VulkanPlacedCableIoBuffers,
    ) -> Result<Self, VulkanBoundDispatchPlanError> {
        if placed_bound_plan.device_id != cable_io.plan.device_id {
            return Err(VulkanBoundDispatchPlanError::CableIoDeviceMismatch {
                plan_device_id: placed_bound_plan.device_id.clone(),
                cable_io_device_id: cable_io.plan.device_id.clone(),
            });
        }

        let mut resident_descriptor_count = 0usize;
        let mut model_boundary_descriptor_count = 0usize;
        let mut local_cable_descriptor_count = 0usize;
        let mut cable_endpoint_descriptor_count = 0usize;
        let mut incoming_cable_descriptor_count = 0usize;
        let mut outgoing_cable_descriptor_count = 0usize;
        let mut dispatches = Vec::with_capacity(placed_bound_plan.dispatches.len());

        for dispatch in &placed_bound_plan.dispatches {
            let mut descriptors = Vec::with_capacity(dispatch.descriptors.len());
            for descriptor in &dispatch.descriptors {
                let target = VulkanMountedPlacedBoundDescriptorTarget::from_placed_target(
                    dispatch, descriptor, cable_io,
                )?;
                match target {
                    VulkanMountedPlacedBoundDescriptorTarget::Resident { .. } => {
                        resident_descriptor_count += 1;
                    }
                    VulkanMountedPlacedBoundDescriptorTarget::ModelInput { .. }
                    | VulkanMountedPlacedBoundDescriptorTarget::ModelOutput { .. } => {
                        model_boundary_descriptor_count += 1;
                    }
                    VulkanMountedPlacedBoundDescriptorTarget::LocalCableInputBuffer { .. }
                    | VulkanMountedPlacedBoundDescriptorTarget::LocalCableOutputBuffer { .. } => {
                        local_cable_descriptor_count += 1;
                    }
                    VulkanMountedPlacedBoundDescriptorTarget::IncomingCableBuffer { .. } => {
                        incoming_cable_descriptor_count += 1;
                        cable_endpoint_descriptor_count += 1;
                    }
                    VulkanMountedPlacedBoundDescriptorTarget::OutgoingCableBuffer { .. } => {
                        outgoing_cable_descriptor_count += 1;
                        cable_endpoint_descriptor_count += 1;
                    }
                }
                descriptors.push(VulkanMountedPlacedBoundDescriptor {
                    binding: descriptor.binding,
                    usage: descriptor.usage.clone(),
                    name: descriptor.name.clone(),
                    target,
                });
            }

            dispatches.push(VulkanMountedPlacedBoundDispatch {
                dispatch_index: dispatch.dispatch_index,
                kernel_id: dispatch.kernel_id.clone(),
                pedal_id: dispatch.pedal_id.clone(),
                circuit_id: dispatch.circuit_id.clone(),
                node_index: dispatch.node_index,
                node_id: dispatch.node_id.clone(),
                op: dispatch.op.clone(),
                reusable_family_id: dispatch.reusable_family_id.clone(),
                artifact_path: dispatch.artifact_path.clone(),
                entry_point: dispatch.entry_point.clone(),
                local_size_x: dispatch.local_size_x,
                descriptors,
                push_constants: dispatch.push_constants.clone(),
                uses_stream_tick: dispatch.uses_stream_tick,
            });
        }

        let total_descriptor_count = resident_descriptor_count
            + model_boundary_descriptor_count
            + local_cable_descriptor_count
            + cable_endpoint_descriptor_count;

        Ok(Self {
            backend_id: placed_bound_plan.backend_id.clone(),
            device_id: placed_bound_plan.device_id.clone(),
            dispatches,
            total_descriptor_count,
            resident_descriptor_count,
            model_boundary_descriptor_count,
            local_cable_descriptor_count,
            cable_endpoint_descriptor_count,
            incoming_cable_descriptor_count,
            outgoing_cable_descriptor_count,
        })
    }

    pub fn dispatch(
        &self,
        pedal_id: &str,
        node_id: &str,
    ) -> Option<&VulkanMountedPlacedBoundDispatch> {
        self.dispatches
            .iter()
            .find(|dispatch| dispatch.pedal_id == pedal_id && dispatch.node_id == node_id)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanMountedPlacedBoundDispatch {
    pub dispatch_index: usize,
    pub kernel_id: String,
    pub pedal_id: String,
    pub circuit_id: String,
    pub node_index: usize,
    pub node_id: String,
    pub op: String,
    pub reusable_family_id: String,
    pub artifact_path: String,
    pub entry_point: String,
    pub local_size_x: u32,
    pub descriptors: Vec<VulkanMountedPlacedBoundDescriptor>,
    pub push_constants: Vec<VulkanKernelScalarBinding>,
    pub uses_stream_tick: bool,
}

fn vulkan_dispatch_semantic_label(
    dispatch: &VulkanMountedPlacedBoundDispatch,
    execution_detail: Option<&str>,
) -> String {
    let mut label = format!(
        "kernel={} pedal={} node={} op={}",
        dispatch.kernel_id, dispatch.pedal_id, dispatch.node_id, dispatch.op
    );
    if let Some(detail) = execution_detail {
        label.push(' ');
        label.push_str(detail);
    }
    label
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanMountedPlacedBoundDescriptor {
    pub binding: usize,
    pub usage: VulkanKernelDescriptorUsage,
    pub name: String,
    pub target: VulkanMountedPlacedBoundDescriptorTarget,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VulkanMountedPlacedBoundDescriptorTarget {
    Resident {
        target: VulkanBoundDescriptorTarget,
    },
    ModelInput {
        signal_id: String,
    },
    ModelOutput {
        signal_id: String,
    },
    LocalCableInputBuffer {
        cable: VulkanPlacedLocalCableBufferBinding,
    },
    LocalCableOutputBuffer {
        cable: VulkanPlacedLocalCableBufferBinding,
    },
    IncomingCableBuffer {
        endpoint: VulkanPlacedCableEndpointBufferBinding,
    },
    OutgoingCableBuffer {
        endpoint: VulkanPlacedCableEndpointBufferBinding,
    },
}

impl VulkanMountedPlacedBoundDescriptorTarget {
    fn from_placed_target(
        dispatch: &VulkanPlacedBoundDispatch,
        descriptor: &VulkanPlacedBoundDescriptor,
        cable_io: &VulkanPlacedCableIoBuffers,
    ) -> Result<Self, VulkanBoundDispatchPlanError> {
        match &descriptor.target {
            VulkanPlacedBoundDescriptorTarget::Resident { target } => Ok(Self::Resident {
                target: target.clone(),
            }),
            VulkanPlacedBoundDescriptorTarget::ModelInput { signal_id } => Ok(Self::ModelInput {
                signal_id: signal_id.clone(),
            }),
            VulkanPlacedBoundDescriptorTarget::ModelOutput { signal_id } => Ok(Self::ModelOutput {
                signal_id: signal_id.clone(),
            }),
            VulkanPlacedBoundDescriptorTarget::LocalCableInput { cable } => {
                Ok(Self::LocalCableInputBuffer {
                    cable: bind_local_cable_buffer(
                        dispatch,
                        descriptor,
                        cable.cable_index,
                        cable_io,
                    )?,
                })
            }
            VulkanPlacedBoundDescriptorTarget::LocalCableOutput { cable } => {
                Ok(Self::LocalCableOutputBuffer {
                    cable: bind_local_cable_buffer(
                        dispatch,
                        descriptor,
                        cable.cable_index,
                        cable_io,
                    )?,
                })
            }
            VulkanPlacedBoundDescriptorTarget::IncomingCable { cable } => {
                Ok(Self::IncomingCableBuffer {
                    endpoint: bind_cable_endpoint_buffer(
                        dispatch,
                        descriptor,
                        VulkanPlacedCableDirection::Incoming,
                        cable.cable_index,
                        cable_io,
                    )?,
                })
            }
            VulkanPlacedBoundDescriptorTarget::OutgoingCable { cable } => {
                Ok(Self::OutgoingCableBuffer {
                    endpoint: bind_cable_endpoint_buffer(
                        dispatch,
                        descriptor,
                        VulkanPlacedCableDirection::Outgoing,
                        cable.cable_index,
                        cable_io,
                    )?,
                })
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanMountedPlacedResidentKernelDispatchReadinessPlan {
    pub backend_id: String,
    pub device_id: String,
    pub dispatches: Vec<VulkanMountedPlacedResidentKernelDispatchReadiness>,
    pub dispatch_count: usize,
    pub instantiable_count: usize,
    pub blocked_count: usize,
    pub missing_loaded_artifact_count: usize,
    pub descriptor_binding_blocked_count: usize,
    pub push_constant_blocked_count: usize,
    pub instantiable_descriptor_count: usize,
}

impl VulkanMountedPlacedResidentKernelDispatchReadinessPlan {
    fn from_mounted_bound_plan(
        mounted: &VulkanMountedPlacedStreamCircuit,
        mounted_bound_plan: &VulkanMountedPlacedBoundDispatchPlan,
        loaded_manifest: &VulkanLoadedReusableKernelArtifactManifest,
    ) -> Self {
        let mut instantiable_count = 0usize;
        let mut blocked_count = 0usize;
        let mut missing_loaded_artifact_count = 0usize;
        let mut descriptor_binding_blocked_count = 0usize;
        let mut push_constant_blocked_count = 0usize;
        let mut instantiable_descriptor_count = 0usize;

        let dispatches = mounted_bound_plan
            .dispatches
            .iter()
            .map(|dispatch| {
                let status = mounted.resident_kernel_dispatch_readiness_for_bound_dispatch(
                    dispatch,
                    loaded_manifest,
                );
                match &status {
                    VulkanMountedPlacedResidentKernelDispatchStatus::Instantiable {
                        descriptor_count,
                        ..
                    } => {
                        instantiable_count += 1;
                        instantiable_descriptor_count += descriptor_count;
                    }
                    VulkanMountedPlacedResidentKernelDispatchStatus::Blocked { error } => {
                        blocked_count += 1;
                        match error {
                            VulkanMountedPlacedResidentKernelDispatchError::MissingLoadedArtifact {
                                ..
                            } => missing_loaded_artifact_count += 1,
                            VulkanMountedPlacedResidentKernelDispatchError::UnsupportedPushConstantScalar {
                                ..
                            }
                            | VulkanMountedPlacedResidentKernelDispatchError::UnsupportedPushConstantBinding {
                                ..
                            }
                            | VulkanMountedPlacedResidentKernelDispatchError::PushConstantByteCountOverflow => {
                                push_constant_blocked_count += 1;
                            }
                            _ => descriptor_binding_blocked_count += 1,
                        }
                    }
                }
                VulkanMountedPlacedResidentKernelDispatchReadiness {
                    dispatch_index: dispatch.dispatch_index,
                    kernel_id: dispatch.kernel_id.clone(),
                    pedal_id: dispatch.pedal_id.clone(),
                    node_id: dispatch.node_id.clone(),
                    op: dispatch.op.clone(),
                    reusable_family_id: dispatch.reusable_family_id.clone(),
                    status,
                }
            })
            .collect::<Vec<_>>();

        let dispatch_count = dispatches.len();
        Self {
            backend_id: mounted_bound_plan.backend_id.clone(),
            device_id: mounted_bound_plan.device_id.clone(),
            dispatches,
            dispatch_count,
            instantiable_count,
            blocked_count,
            missing_loaded_artifact_count,
            descriptor_binding_blocked_count,
            push_constant_blocked_count,
            instantiable_descriptor_count,
        }
    }

    pub fn dispatch(
        &self,
        pedal_id: &str,
        node_id: &str,
    ) -> Option<&VulkanMountedPlacedResidentKernelDispatchReadiness> {
        self.dispatches
            .iter()
            .find(|dispatch| dispatch.pedal_id == pedal_id && dispatch.node_id == node_id)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanMountedPlacedResidentKernelDispatchReadiness {
    pub dispatch_index: usize,
    pub kernel_id: String,
    pub pedal_id: String,
    pub node_id: String,
    pub op: String,
    pub reusable_family_id: String,
    pub status: VulkanMountedPlacedResidentKernelDispatchStatus,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VulkanMountedPlacedResidentKernelDispatchStatus {
    Instantiable {
        descriptor_count: usize,
        workgroup_count_x: u32,
        local_size_x: u32,
        push_constant_byte_count: u32,
    },
    Blocked {
        error: VulkanMountedPlacedResidentKernelDispatchError,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanPlacedLocalCableBufferBinding {
    pub buffer_index: usize,
    pub cable: VulkanPlacedLocalCable,
    pub byte_capacity: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanPlacedCableEndpointBufferBinding {
    pub buffer_index: usize,
    pub endpoint: VulkanPlacedCableEndpoint,
    pub byte_capacity: usize,
}

fn bind_local_cable_buffer(
    dispatch: &VulkanPlacedBoundDispatch,
    descriptor: &VulkanPlacedBoundDescriptor,
    cable_index: usize,
    cable_io: &VulkanPlacedCableIoBuffers,
) -> Result<VulkanPlacedLocalCableBufferBinding, VulkanBoundDispatchPlanError> {
    let (buffer_index, allocation) = cable_io.local_buffer(cable_index).ok_or({
        VulkanBoundDispatchPlanError::MissingLocalCableBuffer {
            dispatch_index: dispatch.dispatch_index,
            binding: descriptor.binding,
            cable_index,
        }
    })?;
    if allocation.cable.byte_capacity != Some(allocation.byte_capacity) {
        return Err(
            VulkanBoundDispatchPlanError::LocalCableByteCapacityMismatch {
                dispatch_index: dispatch.dispatch_index,
                binding: descriptor.binding,
                cable_index,
                cable_byte_capacity: allocation.cable.byte_capacity,
                mounted_byte_capacity: allocation.byte_capacity,
            },
        );
    }

    Ok(VulkanPlacedLocalCableBufferBinding {
        buffer_index,
        cable: allocation.cable.clone(),
        byte_capacity: allocation.byte_capacity,
    })
}

fn bind_cable_endpoint_buffer(
    dispatch: &VulkanPlacedBoundDispatch,
    descriptor: &VulkanPlacedBoundDescriptor,
    direction: VulkanPlacedCableDirection,
    cable_index: usize,
    cable_io: &VulkanPlacedCableIoBuffers,
) -> Result<VulkanPlacedCableEndpointBufferBinding, VulkanBoundDispatchPlanError> {
    let (buffer_index, allocation) = cable_io.buffer(direction, cable_index).ok_or({
        VulkanBoundDispatchPlanError::MissingCableEndpointBuffer {
            dispatch_index: dispatch.dispatch_index,
            binding: descriptor.binding,
            direction,
            cable_index,
        }
    })?;
    if allocation.endpoint.byte_capacity != Some(allocation.byte_capacity) {
        return Err(
            VulkanBoundDispatchPlanError::CableEndpointByteCapacityMismatch {
                dispatch_index: dispatch.dispatch_index,
                binding: descriptor.binding,
                cable_index,
                endpoint_byte_capacity: allocation.endpoint.byte_capacity,
                mounted_byte_capacity: allocation.byte_capacity,
            },
        );
    }

    Ok(VulkanPlacedCableEndpointBufferBinding {
        buffer_index,
        endpoint: allocation.endpoint.clone(),
        byte_capacity: allocation.byte_capacity,
    })
}
