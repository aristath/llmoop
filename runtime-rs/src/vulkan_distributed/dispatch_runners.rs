pub struct VulkanDistributedDispatchRunners {
    pub dispatches: Vec<VulkanDistributedDispatchRunner>,
    pub dispatch_count: usize,
    pub shard_count: usize,
}

fn create_distributed_resident_dispatch(
    device: &VulkanComputeDevice,
    planned_dispatch: &VulkanDistributedDispatchPlan,
    planned_shard: &VulkanDistributedDispatchShard,
    parameter_buffers: &VulkanDistributedParameterBuffers,
    activation_buffers: &VulkanDistributedActivationBuffers,
    artifact: &VulkanLoadedReusableKernelArtifact,
) -> Result<VulkanResidentKernelDispatch, VulkanDistributedDispatchRunnerError> {
    let input = activation_buffers
        .activation_buffer(
            &planned_dispatch.owner_device_id,
            &planned_dispatch.input_activation.component_id,
            planned_dispatch.input_activation.slot,
            &planned_shard.device_id,
        )
        .ok_or_else(|| {
            VulkanDistributedDispatchRunnerError(format!(
                "distributed dispatch {}.{} has no input activation on {:?}",
                planned_dispatch.component_id, planned_dispatch.node_id, planned_shard.device_id
            ))
        })?;
    let output = activation_buffers
        .activation_buffer(
            &planned_dispatch.owner_device_id,
            &planned_dispatch.output_activation.component_id,
            planned_dispatch.output_activation.slot,
            &planned_shard.device_id,
        )
        .ok_or_else(|| {
            VulkanDistributedDispatchRunnerError(format!(
                "distributed dispatch {}.{} has no output activation on {:?}",
                planned_dispatch.component_id, planned_dispatch.node_id, planned_shard.device_id
            ))
        })?;
    let mut bindings = Vec::with_capacity(
        2 + planned_dispatch.auxiliary_input_activations.len() + planned_shard.parameters.len(),
    );
    bindings.push(
        VulkanResidentKernelBufferBinding::new(
            u32::try_from(planned_dispatch.input_activation.binding).map_err(|_| {
                VulkanDistributedDispatchRunnerError(
                    "distributed primary input binding exceeds u32".to_string(),
                )
            })?,
            input,
            planned_dispatch.input_byte_capacity,
        )
        .with_access(VulkanResidentKernelBufferAccess::Read),
    );
    for auxiliary in &planned_dispatch.auxiliary_input_activations {
        let buffer = activation_buffers
            .activation_buffer(
                &planned_dispatch.owner_device_id,
                &auxiliary.component_id,
                auxiliary.slot,
                &planned_shard.device_id,
            )
            .ok_or_else(|| {
                VulkanDistributedDispatchRunnerError(format!(
                    "distributed dispatch {}.{} has no auxiliary input {} on {:?}",
                    planned_dispatch.component_id,
                    planned_dispatch.node_id,
                    auxiliary.signal_id,
                    planned_shard.device_id
                ))
            })?;
        bindings.push(
            VulkanResidentKernelBufferBinding::new(
                u32::try_from(auxiliary.binding).map_err(|_| {
                    VulkanDistributedDispatchRunnerError(
                        "distributed auxiliary input binding exceeds u32".to_string(),
                    )
                })?,
                buffer,
                auxiliary.signal_byte_capacity,
            )
            .with_access(VulkanResidentKernelBufferAccess::Read),
        );
    }
    bindings.push(
        VulkanResidentKernelBufferBinding::new(
            u32::try_from(planned_dispatch.output_activation.binding).map_err(|_| {
                VulkanDistributedDispatchRunnerError(
                    "distributed output binding exceeds u32".to_string(),
                )
            })?,
            output,
            planned_shard.output_byte_count,
        )
        .with_byte_offset(planned_shard.output_byte_offset)
        .with_access(VulkanResidentKernelBufferAccess::Write),
    );
    for fragment in &planned_shard.parameters {
        let allocation = parameter_buffers
            .parameter_buffer(
                &planned_shard.device_id,
                &fragment.tensor,
                fragment.byte_offset,
                fragment.byte_count,
            )
            .ok_or_else(|| {
                VulkanDistributedDispatchRunnerError(format!(
                    "distributed dispatch {}.{} has no tensor {:?} range at byte {} with length {} on {:?}",
                    planned_dispatch.component_id,
                    planned_dispatch.node_id,
                    fragment.tensor,
                    fragment.byte_offset,
                    fragment.byte_count,
                    planned_shard.device_id
                ))
            })?;
        let binding = u32::try_from(fragment.binding).map_err(|_| {
            VulkanDistributedDispatchRunnerError(format!(
                "distributed descriptor binding {} exceeds u32",
                fragment.binding
            ))
        })?;
        bindings.push(
            VulkanResidentKernelBufferBinding::new(
                binding,
                &allocation.buffer,
                fragment.byte_count,
            )
            .with_access(VulkanResidentKernelBufferAccess::Read),
        );
    }
    device
        .create_resident_kernel_dispatch_2d_with_base_z(
            &artifact.words,
            &bindings,
            planned_shard.workgroup_count_x,
            1,
            planned_shard.base_workgroup_z,
            artifact.artifact.local_size_x,
            0,
            Some(format!(
                "component={} node={} distributed=device:{} rows={}..{} base_z={} distribution={:?}",
                planned_dispatch.component_id,
                planned_dispatch.node_id,
                planned_shard.device_id,
                planned_shard.row_start,
                planned_shard.row_start + planned_shard.row_count,
                planned_shard.base_workgroup_z,
                planned_dispatch.distribution,
            )),
        )
        .map_err(|error| {
            VulkanDistributedDispatchRunnerError(format!(
                "failed to create distributed dispatch {}.{} shard on {:?}: {error}",
                planned_dispatch.component_id, planned_dispatch.node_id, planned_shard.device_id
            ))
        })
}

impl VulkanDistributedDispatchRunners {
    pub fn create<'a, F, E>(
        execution_plan: &VulkanDistributedExecutionPlan,
        parameter_buffers: &VulkanDistributedParameterBuffers,
        activation_buffers: &VulkanDistributedActivationBuffers,
        loaded_manifest: &VulkanLoadedReusableKernelArtifactManifest,
        mut device_for: F,
    ) -> Result<Self, VulkanDistributedDispatchRunnerError>
    where
        F: FnMut(&str) -> Result<&'a VulkanComputeDevice, E>,
        E: Display,
    {
        let mut dispatches = Vec::with_capacity(execution_plan.dispatch_groups.len());
        let mut shard_count = 0usize;
        for planned_group in &execution_plan.dispatch_groups {
            let leader = planned_group.leader();
            let tail = planned_group.tail();
            let owner_device = device_for(&planned_group.owner_device_id).map_err(|error| {
                VulkanDistributedDispatchRunnerError(format!(
                    "failed to resolve distributed owner device {:?}: {error}",
                    planned_group.owner_device_id
                ))
            })?;
            let mut shards = Vec::with_capacity(leader.shards.len());
            for shard_index in 0..leader.shards.len() {
                let leader_shard = &leader.shards[shard_index];
                let device = device_for(&leader_shard.device_id).map_err(|error| {
                    VulkanDistributedDispatchRunnerError(format!(
                        "failed to resolve distributed shard device {:?}: {error}",
                        leader_shard.device_id
                    ))
                })?;
                let mut resident_dispatches = Vec::with_capacity(planned_group.dispatches.len());
                let mut planned_shards = Vec::with_capacity(planned_group.dispatches.len());
                for planned_dispatch in &planned_group.dispatches {
                    let planned_shard =
                        planned_dispatch.shards.get(shard_index).ok_or_else(|| {
                            VulkanDistributedDispatchRunnerError(format!(
                                "distributed group {}..{} has no shard {shard_index} for {}.{}",
                                leader.dispatch_index,
                                tail.dispatch_index,
                                planned_dispatch.component_id,
                                planned_dispatch.node_id
                            ))
                        })?;
                    if planned_shard.device_id != leader_shard.device_id {
                        return Err(VulkanDistributedDispatchRunnerError(format!(
                            "distributed group {}..{} changes shard {shard_index} device from {:?} to {:?}",
                            leader.dispatch_index,
                            tail.dispatch_index,
                            leader_shard.device_id,
                            planned_shard.device_id
                        )));
                    }
                    let artifact = loaded_manifest
                        .artifact(&planned_dispatch.reusable_family_id)
                        .ok_or_else(|| {
                            VulkanDistributedDispatchRunnerError(format!(
                                "distributed dispatch {}.{} is missing loaded family {:?}",
                                planned_dispatch.component_id,
                                planned_dispatch.node_id,
                                planned_dispatch.reusable_family_id
                            ))
                        })?;
                    resident_dispatches.push(create_distributed_resident_dispatch(
                        device,
                        planned_dispatch,
                        planned_shard,
                        parameter_buffers,
                        activation_buffers,
                        artifact,
                    )?);
                    planned_shards.push(planned_shard.clone());
                }
                let sequence = device.create_resident_kernel_sequence().map_err(|error| {
                    VulkanDistributedDispatchRunnerError(format!(
                        "failed to create distributed sequence {}..{} shard on {:?}: {error}",
                        leader.dispatch_index, tail.dispatch_index, leader_shard.device_id
                    ))
                })?;
                let steps = resident_dispatches
                    .iter()
                    .map(|dispatch| VulkanResidentKernelSequenceStep::new(dispatch, &[]))
                    .collect::<Vec<_>>();
                device
                    .record_resident_kernel_sequence(&sequence, &steps)
                    .map_err(|error| {
                        VulkanDistributedDispatchRunnerError(format!(
                            "failed to record distributed sequence {}..{} shard on {:?}: {error}",
                            leader.dispatch_index, tail.dispatch_index, leader_shard.device_id
                        ))
                    })?;
                shards.push(VulkanDistributedDispatchShardRunner {
                    device_id: leader_shard.device_id.clone(),
                    planned: planned_shards,
                    resident_dispatches,
                    sequence,
                });
                shard_count = shard_count
                    .checked_add(planned_group.dispatches.len())
                    .ok_or_else(|| {
                        VulkanDistributedDispatchRunnerError(
                            "distributed dispatch shard count overflowed".to_string(),
                        )
                    })?;
            }
            let mut helper_synchronization = Vec::with_capacity(
                leader
                    .shards
                    .iter()
                    .filter(|shard| shard.device_id != planned_group.owner_device_id)
                    .count(),
            );
            for planned_shard in &leader.shards {
                if planned_shard.device_id == planned_group.owner_device_id {
                    continue;
                }
                let helper_device = device_for(&planned_shard.device_id).map_err(|error| {
                    VulkanDistributedDispatchRunnerError(format!(
                        "failed to resolve distributed helper device {:?}: {error}",
                        planned_shard.device_id
                    ))
                })?;
                if !owner_device.supports_opaque_fd_timeline_semaphores()
                    || !helper_device.supports_opaque_fd_timeline_semaphores()
                {
                    return Err(VulkanDistributedDispatchRunnerError(format!(
                        "distributed dispatch {}.{} requires persistent opaque-file timeline semaphores on owner {:?} and helper {:?}",
                        leader.component_id,
                        leader.node_id,
                        planned_group.owner_device_id,
                        planned_shard.device_id
                    )));
                }
                let ready_source = owner_device
                    .create_opaque_fd_exportable_timeline_semaphore(0)
                    .map_err(VulkanDistributedDispatchRunnerError::from)?;
                let ready_wait = helper_device
                    .create_timeline_semaphore(0)
                    .map_err(VulkanDistributedDispatchRunnerError::from)?;
                helper_device
                    .import_timeline_semaphore_opaque_fd(
                        &ready_wait,
                        owner_device
                            .export_timeline_semaphore_opaque_fd(&ready_source)
                            .map_err(VulkanDistributedDispatchRunnerError::from)?,
                    )
                    .map_err(VulkanDistributedDispatchRunnerError::from)?;
                let done_source = helper_device
                    .create_opaque_fd_exportable_timeline_semaphore(0)
                    .map_err(VulkanDistributedDispatchRunnerError::from)?;
                let done_wait = owner_device
                    .create_timeline_semaphore(0)
                    .map_err(VulkanDistributedDispatchRunnerError::from)?;
                owner_device
                    .import_timeline_semaphore_opaque_fd(
                        &done_wait,
                        helper_device
                            .export_timeline_semaphore_opaque_fd(&done_source)
                            .map_err(VulkanDistributedDispatchRunnerError::from)?,
                    )
                    .map_err(VulkanDistributedDispatchRunnerError::from)?;
                helper_synchronization.push(VulkanDistributedDispatchHelperSynchronization {
                    device_id: planned_shard.device_id.clone(),
                    ready_source,
                    ready_wait,
                    done_source,
                    done_wait,
                });
            }
            dispatches.push(VulkanDistributedDispatchRunner {
                planned: planned_group.clone(),
                shards,
                helper_synchronization,
                dependency_clock: VulkanDistributedDependencyClock::new(),
            });
        }

        Ok(Self {
            dispatch_count: execution_plan.dispatches.len(),
            dispatches,
            shard_count,
        })
    }

    pub fn dispatch(
        &self,
        owner_device_id: &str,
        dispatch_index: usize,
    ) -> Option<&VulkanDistributedDispatchRunner> {
        self.dispatches.iter().find(|dispatch| {
            dispatch.planned.owner_device_id == owner_device_id
                && dispatch.planned.leader().dispatch_index == dispatch_index
        })
    }

    pub fn dispatch_group(
        &self,
        owner_device_id: &str,
        dispatch_index: usize,
    ) -> Option<&VulkanDistributedDispatchGroup> {
        self.dispatches
            .iter()
            .find(|runner| {
                runner.planned.owner_device_id == owner_device_id
                    && runner.planned.contains_dispatch(dispatch_index)
            })
            .map(|runner| &runner.planned)
    }

    pub fn leader_dispatch_index(
        &self,
        owner_device_id: &str,
        dispatch_index: usize,
    ) -> Option<usize> {
        self.dispatch_group(owner_device_id, dispatch_index)
            .map(|group| group.leader().dispatch_index)
    }

    pub fn reserve_dependency_value(
        &self,
        owner_device_id: &str,
        dispatch_index: usize,
    ) -> Result<u64, VulkanDistributedDispatchRunnerError> {
        let dispatch = self.dispatch(owner_device_id, dispatch_index).ok_or_else(|| {
            VulkanDistributedDispatchRunnerError(format!(
                "distributed runner has no dispatch {dispatch_index} owned by {owner_device_id:?}"
            ))
        })?;
        dispatch
            .dependency_clock
            .reserve(owner_device_id, dispatch_index)
    }

    pub fn advance_replayed_dependency_values(
        &self,
        count: usize,
    ) -> Result<(), VulkanDistributedDispatchRunnerError> {
        let count = u64::try_from(count).map_err(|_| {
            VulkanDistributedDispatchRunnerError(
                "distributed replay dependency count exceeds u64".to_string(),
            )
        })?;
        for dispatch in &self.dispatches {
            dispatch.dependency_clock.validate_advance(
                count,
                &dispatch.planned.owner_device_id,
                dispatch.planned.leader().dispatch_index,
            )?;
        }
        for dispatch in &self.dispatches {
            dispatch.dependency_clock.advance(count);
        }
        Ok(())
    }

    pub fn owner_ready_signal_points(
        &self,
        owner_device_id: &str,
        dispatch_index: usize,
        dependency_value: u64,
    ) -> Result<Vec<VulkanTimelineSemaphorePoint<'_>>, VulkanDistributedDispatchRunnerError> {
        let dispatch = self.dispatch(owner_device_id, dispatch_index).ok_or_else(|| {
            VulkanDistributedDispatchRunnerError(format!(
                "distributed runner has no dispatch {dispatch_index} owned by {owner_device_id:?}"
            ))
        })?;
        Ok(dispatch
            .helper_synchronization
            .iter()
            .map(|sync| VulkanTimelineSemaphorePoint::new(&sync.ready_source, dependency_value))
            .collect())
    }

    pub fn owner_completion_wait_points(
        &self,
        owner_device_id: &str,
        dispatch_index: usize,
        dependency_value: u64,
    ) -> Result<Vec<VulkanTimelineSemaphorePoint<'_>>, VulkanDistributedDispatchRunnerError> {
        let dispatch = self.dispatch(owner_device_id, dispatch_index).ok_or_else(|| {
            VulkanDistributedDispatchRunnerError(format!(
                "distributed runner has no dispatch {dispatch_index} owned by {owner_device_id:?}"
            ))
        })?;
        Ok(dispatch
            .helper_synchronization
            .iter()
            .map(|sync| VulkanTimelineSemaphorePoint::new(&sync.done_wait, dependency_value))
            .collect())
    }

    pub fn submit_dispatch_with_device_dependencies<'a, F, E>(
        &self,
        owner_device_id: &str,
        dispatch_index: usize,
        submission: VulkanDistributedDispatchSubmission,
        submission_batch: Option<&VulkanResidentQueueSubmissionBatch<'a>>,
        mut device_for: F,
    ) -> Result<VulkanDistributedDispatchRun, VulkanDistributedDispatchRunnerError>
    where
        F: FnMut(&str) -> Result<&'a VulkanComputeDevice, E>,
        E: Display,
    {
        let VulkanDistributedDispatchSubmission {
            dependency_value,
            consume_owner_ready_signal,
            prepare_owner_continuation,
            signal_completion,
        } = submission;
        let dispatch = self.dispatch(owner_device_id, dispatch_index).ok_or_else(|| {
            VulkanDistributedDispatchRunnerError(format!(
                "distributed runner has no dispatch {dispatch_index} owned by {owner_device_id:?}"
            ))
        })?;
        let resolved_shards = dispatch
            .shards
            .iter()
            .map(|shard| {
                device_for(&shard.device_id)
                    .map(|device| (shard, device))
                    .map_err(|error| {
                        VulkanDistributedDispatchRunnerError(format!(
                            "failed to resolve distributed shard device {:?}: {error}",
                            shard.device_id
                        ))
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut submitted: Vec<(&VulkanComputeDevice, &VulkanDistributedDispatchShardRunner)> =
            Vec::with_capacity(dispatch.shards.len());
        for (shard, device) in resolved_shards {
            let synchronization = dispatch
                .helper_synchronization
                .iter()
                .find(|sync| sync.device_id == shard.device_id);
            let wait_points = synchronization
                .filter(|_| consume_owner_ready_signal)
                .map(|sync| {
                    vec![VulkanTimelineSemaphorePoint::new(
                        &sync.ready_wait,
                        dependency_value,
                    )]
                })
                .unwrap_or_default();
            let signal_points = synchronization
                .filter(|_| prepare_owner_continuation)
                .map(|sync| {
                    vec![VulkanTimelineSemaphorePoint::new(
                        &sync.done_source,
                        dependency_value,
                    )]
                })
                .unwrap_or_default();
            let submission = if let Some(submission_batch) = submission_batch {
                submission_batch.enqueue_recorded_sequence(
                    device,
                    &shard.sequence,
                    &wait_points,
                    &signal_points,
                    signal_completion,
                )
            } else if signal_completion {
                device.submit_recorded_resident_kernel_sequence_with_timeline_semaphores(
                    &shard.sequence,
                    &wait_points,
                    &signal_points,
                )
            } else {
                device.submit_recorded_resident_kernel_sequence_unfenced_with_timeline_semaphores(
                    &shard.sequence,
                    &wait_points,
                    &signal_points,
                )
            };
            if let Err(error) = submission {
                for (submitted_device, submitted_shard) in &submitted {
                    let _ =
                        submitted_device.wait_resident_kernel_sequence(&submitted_shard.sequence);
                }
                return Err(VulkanDistributedDispatchRunnerError(format!(
                    "failed to submit distributed dispatch {}.{} shard on {:?}: {error}",
                    dispatch.planned.leader().component_id,
                    dispatch.planned.leader().node_id,
                    shard.device_id
                )));
            }
            submitted.push((device, shard));
        }

        Ok(VulkanDistributedDispatchRun {
            owner_device_id: owner_device_id.to_string(),
            dispatch_index,
            component_id: dispatch.planned.leader().component_id.clone(),
            node_id: dispatch.planned.tail().node_id.clone(),
            shard_count: dispatch.shards.len(),
        })
    }

    pub fn wait_dispatch<'a, F, E>(
        &self,
        owner_device_id: &str,
        dispatch_index: usize,
        mut device_for: F,
    ) -> Result<(), VulkanDistributedDispatchRunnerError>
    where
        F: FnMut(&str) -> Result<&'a VulkanComputeDevice, E>,
        E: Display,
    {
        let dispatch = self.dispatch(owner_device_id, dispatch_index).ok_or_else(|| {
            VulkanDistributedDispatchRunnerError(format!(
                "distributed runner has no dispatch {dispatch_index} owned by {owner_device_id:?}"
            ))
        })?;
        let resolved_shards = dispatch
            .shards
            .iter()
            .map(|shard| {
                device_for(&shard.device_id)
                    .map(|device| (shard, device))
                    .map_err(|error| {
                        VulkanDistributedDispatchRunnerError(format!(
                            "failed to resolve distributed shard device {:?}: {error}",
                            shard.device_id
                        ))
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut first_error = None;
        for (shard, device) in resolved_shards {
            if let Err(error) = device.wait_resident_kernel_sequence(&shard.sequence)
                && first_error.is_none()
            {
                first_error = Some(format!(
                    "failed waiting for distributed dispatch {}.{} shard on {:?}: {error}",
                    dispatch.planned.leader().component_id,
                    dispatch.planned.tail().node_id,
                    shard.device_id
                ));
            }
        }
        if let Some(error) = first_error {
            return Err(VulkanDistributedDispatchRunnerError(error));
        }
        Ok(())
    }

    pub fn run_dispatch<'a, F, E>(
        &self,
        owner_device_id: &str,
        dispatch_index: usize,
        mut device_for: F,
    ) -> Result<VulkanDistributedDispatchRun, VulkanDistributedDispatchRunnerError>
    where
        F: FnMut(&str) -> Result<&'a VulkanComputeDevice, E>,
        E: Display,
    {
        let dispatch = self
            .dispatch(owner_device_id, dispatch_index)
            .ok_or_else(|| {
                VulkanDistributedDispatchRunnerError(format!(
                    "distributed runner has no dispatch {dispatch_index} owned by {owner_device_id:?}"
                ))
            })?;
        let mut submitted: Vec<(&VulkanComputeDevice, &VulkanDistributedDispatchShardRunner)> =
            Vec::with_capacity(dispatch.shards.len());
        for shard in &dispatch.shards {
            let device = device_for(&shard.device_id).map_err(|error| {
                VulkanDistributedDispatchRunnerError(format!(
                    "failed to resolve distributed shard device {:?}: {error}",
                    shard.device_id
                ))
            })?;
            if let Err(error) = device.submit_recorded_resident_kernel_sequence(&shard.sequence) {
                for (submitted_device, submitted_shard) in &submitted {
                    let _ =
                        submitted_device.wait_resident_kernel_sequence(&submitted_shard.sequence);
                }
                return Err(VulkanDistributedDispatchRunnerError(format!(
                    "failed to submit distributed dispatch {}.{} shard on {:?}: {error}",
                    dispatch.planned.leader().component_id,
                    dispatch.planned.tail().node_id,
                    shard.device_id
                )));
            }
            submitted.push((device, shard));
        }
        let mut first_wait_error = None;
        for (device, shard) in &submitted {
            if let Err(error) = device.wait_resident_kernel_sequence(&shard.sequence)
                && first_wait_error.is_none()
            {
                first_wait_error = Some(format!(
                    "failed waiting for distributed dispatch {}.{} shard on {:?}: {error}",
                    dispatch.planned.leader().component_id,
                    dispatch.planned.tail().node_id,
                    shard.device_id
                ));
            }
        }
        if let Some(error) = first_wait_error {
            return Err(VulkanDistributedDispatchRunnerError(error));
        }
        Ok(VulkanDistributedDispatchRun {
            owner_device_id: owner_device_id.to_string(),
            dispatch_index,
            component_id: dispatch.planned.leader().component_id.clone(),
            node_id: dispatch.planned.tail().node_id.clone(),
            shard_count: dispatch.shards.len(),
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanDistributedDispatchRun {
    pub owner_device_id: String,
    pub dispatch_index: usize,
    pub component_id: String,
    pub node_id: String,
    pub shard_count: usize,
}

pub struct VulkanDistributedDispatchRunner {
    pub planned: VulkanDistributedDispatchGroup,
    pub shards: Vec<VulkanDistributedDispatchShardRunner>,
    helper_synchronization: Vec<VulkanDistributedDispatchHelperSynchronization>,
    dependency_clock: VulkanDistributedDependencyClock,
}

struct VulkanDistributedDependencyClock {
    next_value: Cell<u64>,
}

impl VulkanDistributedDependencyClock {
    fn new() -> Self {
        Self {
            next_value: Cell::new(1),
        }
    }

    fn reserve(
        &self,
        owner_device_id: &str,
        dispatch_index: usize,
    ) -> Result<u64, VulkanDistributedDispatchRunnerError> {
        let value = self.next_value.get();
        let next = value.checked_add(1).ok_or_else(|| {
            VulkanDistributedDispatchRunnerError(format!(
                "distributed dispatch {dispatch_index} owned by {owner_device_id:?} exhausted its timeline semaphore values"
            ))
        })?;
        self.next_value.set(next);
        Ok(value)
    }

    fn validate_advance(
        &self,
        count: u64,
        owner_device_id: &str,
        dispatch_index: usize,
    ) -> Result<(), VulkanDistributedDispatchRunnerError> {
        self.next_value.get().checked_add(count).ok_or_else(|| {
            VulkanDistributedDispatchRunnerError(format!(
                "distributed dispatch {dispatch_index} owned by {owner_device_id:?} exhausts its timeline semaphore values during replay"
            ))
        })?;
        Ok(())
    }

    fn advance(&self, count: u64) {
        self.next_value.set(
            self.next_value
                .get()
                .checked_add(count)
                .expect("distributed replay dependency advance was validated"),
        );
    }
}

pub struct VulkanDistributedDispatchShardRunner {
    pub device_id: String,
    pub planned: Vec<VulkanDistributedDispatchShard>,
    pub resident_dispatches: Vec<VulkanResidentKernelDispatch>,
    pub sequence: VulkanResidentKernelSequence,
}

struct VulkanDistributedDispatchHelperSynchronization {
    device_id: String,
    ready_source: VulkanTimelineSemaphore,
    ready_wait: VulkanTimelineSemaphore,
    done_source: VulkanTimelineSemaphore,
    done_wait: VulkanTimelineSemaphore,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanDistributedDispatchRunnerError(pub String);

impl Display for VulkanDistributedDispatchRunnerError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for VulkanDistributedDispatchRunnerError {}

impl From<VulkanError> for VulkanDistributedDispatchRunnerError {
    fn from(error: VulkanError) -> Self {
        Self(error.to_string())
    }
}

