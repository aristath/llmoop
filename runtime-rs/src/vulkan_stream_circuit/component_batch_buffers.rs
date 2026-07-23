#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum VulkanComponentBatchSignalKey {
    Activation { component_id: String, signal_id: String },
    ModelInput(String),
    ModelOutput(String),
    LocalEdge(usize),
    IncomingEdge(usize),
    OutgoingEdge(usize),
}

struct VulkanComponentBatchSignalBuffer {
    frame_byte_capacity: usize,
    buffer: Arc<VulkanResidentBuffer>,
    shared_device_buffers: BTreeMap<String, Arc<VulkanResidentBuffer>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct VulkanComponentBatchSignalLifetime {
    key: VulkanComponentBatchSignalKey,
    frame_byte_capacity: usize,
    host_visible: bool,
    first_dispatch: usize,
    last_dispatch: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct VulkanComponentBatchSignalBufferPlan {
    frame_byte_capacity: usize,
    host_visible: bool,
    last_dispatch: usize,
}

fn allocate_component_batch_signal_lifetimes(
    mut lifetimes: Vec<VulkanComponentBatchSignalLifetime>,
) -> (
    BTreeMap<VulkanComponentBatchSignalKey, usize>,
    Vec<VulkanComponentBatchSignalBufferPlan>,
) {
    lifetimes.sort_by(|left, right| {
        left.first_dispatch
            .cmp(&right.first_dispatch)
            .then_with(|| left.last_dispatch.cmp(&right.last_dispatch))
            .then_with(|| left.key.cmp(&right.key))
    });
    let mut signal_buffer_indices = BTreeMap::new();
    let mut buffers = Vec::<VulkanComponentBatchSignalBufferPlan>::new();
    for lifetime in lifetimes {
        let buffer_index = buffers
            .iter()
            .position(|buffer| {
                buffer.frame_byte_capacity == lifetime.frame_byte_capacity
                    && buffer.host_visible == lifetime.host_visible
                    && buffer.last_dispatch < lifetime.first_dispatch
            })
            .unwrap_or_else(|| {
                buffers.push(VulkanComponentBatchSignalBufferPlan {
                    frame_byte_capacity: lifetime.frame_byte_capacity,
                    host_visible: lifetime.host_visible,
                    last_dispatch: lifetime.last_dispatch,
                });
                buffers.len() - 1
            });
        buffers[buffer_index].last_dispatch = lifetime.last_dispatch;
        signal_buffer_indices.insert(lifetime.key, buffer_index);
    }
    (signal_buffer_indices, buffers)
}

fn component_batch_signal_buffer_plan(
    mounted: &VulkanMountedPlacedStreamCircuit,
    dispatches: &[VulkanMountedPlacedBoundDispatch],
) -> Result<
    (
        BTreeMap<VulkanComponentBatchSignalKey, usize>,
        Vec<VulkanComponentBatchSignalBufferPlan>,
    ),
    VulkanResidentInProcessPlacedRuntimeError,
> {
    let dispatch_count = dispatches.len();
    let mut lifetimes = BTreeMap::<VulkanComponentBatchSignalKey, (usize, bool, usize, usize)>::new();
    for (dispatch_index, dispatch) in dispatches.iter().enumerate() {
        for descriptor in &dispatch.descriptors {
            let Some((key, frame_byte_capacity)) =
                component_batch_signal_target_with_mounted(mounted, descriptor)?
            else {
                continue;
            };
            let host_visible = matches!(
                key,
                VulkanComponentBatchSignalKey::IncomingEdge(_)
                    | VulkanComponentBatchSignalKey::OutgoingEdge(_)
            );
            let external_source = matches!(
                key,
                VulkanComponentBatchSignalKey::ModelInput(_)
                    | VulkanComponentBatchSignalKey::IncomingEdge(_)
            );
            let external_sink = matches!(
                key,
                VulkanComponentBatchSignalKey::ModelOutput(_)
                    | VulkanComponentBatchSignalKey::OutgoingEdge(_)
            );
            let first_dispatch = if external_source { 0 } else { dispatch_index };
            let last_dispatch = if external_sink {
                dispatch_count
            } else {
                dispatch_index
            };
            match lifetimes.entry(key.clone()) {
                std::collections::btree_map::Entry::Vacant(entry) => {
                    entry.insert((
                        frame_byte_capacity,
                        host_visible,
                        first_dispatch,
                        last_dispatch,
                    ));
                }
                std::collections::btree_map::Entry::Occupied(mut entry) => {
                    let (existing_capacity, existing_visibility, first, last) = entry.get_mut();
                    if *existing_capacity != frame_byte_capacity
                        || *existing_visibility != host_visible
                    {
                        return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                            VulkanError(format!(
                                "component batch signal {key:?} has incompatible physical requirements"
                            )),
                        ));
                    }
                    *first = (*first).min(first_dispatch);
                    *last = (*last).max(last_dispatch);
                }
            }
        }
    }
    Ok(allocate_component_batch_signal_lifetimes(
        lifetimes
            .into_iter()
            .map(
                |(key, (frame_byte_capacity, host_visible, first_dispatch, last_dispatch))| {
                    VulkanComponentBatchSignalLifetime {
                        key,
                        frame_byte_capacity,
                        host_visible,
                        first_dispatch,
                        last_dispatch,
                    }
                },
            )
            .collect(),
    ))
}

