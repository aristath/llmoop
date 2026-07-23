fn descriptor_bindings_for_kernel(
    kernel: &VulkanKernelInterface,
) -> Vec<VulkanKernelDescriptorBinding> {
    let mut bindings = Vec::new();

    for input in &kernel.inputs {
        push_descriptor_binding(
            &mut bindings,
            VulkanKernelDescriptorUsage::InputSignal,
            input.signal_id.clone(),
            VulkanKernelDescriptorResource::Signal(input.clone()),
        );
    }
    for output in &kernel.outputs {
        push_descriptor_binding(
            &mut bindings,
            VulkanKernelDescriptorUsage::OutputSignal,
            output.signal_id.clone(),
            VulkanKernelDescriptorResource::Signal(output.clone()),
        );
    }
    for parameter in &kernel.parameters {
        push_descriptor_binding(
            &mut bindings,
            VulkanKernelDescriptorUsage::Parameter,
            parameter.param_id.clone(),
            VulkanKernelDescriptorResource::Parameter(parameter.clone()),
        );
    }
    for state in &kernel.state_reads {
        push_descriptor_binding(
            &mut bindings,
            VulkanKernelDescriptorUsage::StateRead,
            state.state_id.clone(),
            VulkanKernelDescriptorResource::State {
                pedal_id: state.pedal_id.clone(),
                binding: state.clone(),
            },
        );
    }
    for state in &kernel.state_writes {
        push_descriptor_binding(
            &mut bindings,
            VulkanKernelDescriptorUsage::StateWrite,
            state.state_id.clone(),
            VulkanKernelDescriptorResource::State {
                pedal_id: state.pedal_id.clone(),
                binding: state.clone(),
            },
        );
    }
    for state_view in &kernel.state_views {
        push_descriptor_binding(
            &mut bindings,
            VulkanKernelDescriptorUsage::StateView,
            state_view.signal_id.clone(),
            VulkanKernelDescriptorResource::Signal(state_view.clone()),
        );
    }

    bindings
}

fn push_descriptor_binding(
    bindings: &mut Vec<VulkanKernelDescriptorBinding>,
    usage: VulkanKernelDescriptorUsage,
    name: String,
    resource: VulkanKernelDescriptorResource,
) {
    bindings.push(VulkanKernelDescriptorBinding {
        binding: bindings.len(),
        usage,
        name,
        resource,
    });
}

fn parameter_binding_index(
    resource_plan: &StreamCircuitResourcePlan,
    resident_plan: &VulkanStreamCircuitResidentPlan,
    hosted_pedals: Option<&BTreeSet<String>>,
) -> Result<BTreeMap<(String, String), VulkanParameterBinding>, VulkanBindingPlanError> {
    let hosts_pedal = |pedal_id: &str| {
        hosted_pedals
            .map(|pedals| pedals.contains(pedal_id))
            .unwrap_or(true)
    };
    let resident_by_tensor: BTreeMap<_, _> = resident_plan
        .permanent_parameters
        .iter()
        .map(|parameter| (parameter.tensor.as_str(), parameter))
        .collect();
    let mut bindings = BTreeMap::new();

    for parameter in &resource_plan.parameters {
        let hosted_uses = parameter
            .uses
            .iter()
            .filter(|use_ref| hosts_pedal(&use_ref.pedal_id))
            .collect::<Vec<_>>();
        if hosted_uses.is_empty() {
            continue;
        }
        let resident = resident_by_tensor
            .get(parameter.tensor.as_str())
            .ok_or_else(|| {
                VulkanBindingPlanError(format!(
                    "resident plan has no permanent parameter for tensor {:?}",
                    parameter.tensor
                ))
            })?;
        for use_ref in hosted_uses {
            let key = (use_ref.pedal_id.clone(), use_ref.param_id.clone());
            let previous = bindings.insert(
                key.clone(),
                VulkanParameterBinding {
                    param_id: use_ref.param_id.clone(),
                    tensor: parameter.tensor.clone(),
                    byte_count: resident.byte_count,
                    shape: resident.shape.clone(),
                },
            );
            if previous.is_some() {
                return Err(VulkanBindingPlanError(format!(
                    "duplicate parameter binding for {}.{}",
                    key.0, key.1
                )));
            }
        }
    }

    Ok(bindings)
}

fn state_binding_index(
    resource_plan: &StreamCircuitResourcePlan,
    resident_plan: &VulkanStreamCircuitResidentPlan,
) -> Result<BTreeMap<(String, String), VulkanStateBinding>, VulkanBindingPlanError> {
    let mut bindings = BTreeMap::new();
    for state in &resident_plan.stream_state_buffers {
        let key = (state.pedal_id.clone(), state.state_id.clone());
        let previous = bindings.insert(
            key.clone(),
            VulkanStateBinding {
                pedal_id: state.pedal_id.clone(),
                state_id: state.state_id.clone(),
                state_type: state.state_type.clone(),
                static_bytes: state.static_bytes,
                bytes_per_activation: state.bytes_per_activation,
            },
        );
        if previous.is_some() {
            return Err(VulkanBindingPlanError(format!(
                "duplicate state binding for {}.{}",
                key.0, key.1
            )));
        }
    }

    let hosted_pedals = resident_plan
        .activation_banks
        .iter()
        .map(|bank| bank.pedal_id.as_str())
        .collect::<BTreeSet<_>>();
    let mut aliases = BTreeMap::new();
    for state in &resource_plan.state_allocations {
        if !hosted_pedals.contains(state.pedal_id.as_str()) {
            continue;
        }
        let target = (state.pedal_id.clone(), state.state_id.clone());
        let Some(source) = state
            .sharing
            .as_deref()
            .map(shared_state_source)
            .transpose()?
            .flatten()
        else {
            continue;
        };
        aliases.insert(target, source);
    }

    let planned = resource_plan
        .state_allocations
        .iter()
        .map(|state| ((state.pedal_id.as_str(), state.state_id.as_str()), state))
        .collect::<BTreeMap<_, _>>();

    for (target, initial_source) in &aliases {
        let mut source = initial_source.clone();
        let mut visited = BTreeSet::from([target.clone()]);
        while let Some(next) = aliases.get(&source) {
            if !visited.insert(source.clone()) {
                return Err(VulkanBindingPlanError(format!(
                    "shared state alias cycle reaches {}.{}",
                    source.0, source.1
                )));
            }
            source = next.clone();
        }
        if !visited.insert(source.clone()) {
            return Err(VulkanBindingPlanError(format!(
                "shared state alias cycle reaches {}.{}",
                source.0, source.1
            )));
        }

        let target_state = planned
            .get(&(target.0.as_str(), target.1.as_str()))
            .ok_or_else(|| {
                VulkanBindingPlanError(format!(
                    "shared state target {}.{} is not planned",
                    target.0, target.1
                ))
            })?;
        let source_state = planned
            .get(&(source.0.as_str(), source.1.as_str()))
            .ok_or_else(|| {
                VulkanBindingPlanError(format!(
                    "shared state {}.{} references unplanned source {}.{}",
                    target.0, target.1, source.0, source.1
                ))
            })?;
        let source_binding = bindings.get(&source).cloned().ok_or_else(|| {
            VulkanBindingPlanError(format!(
                "shared state {}.{} references non-resident source {}.{}",
                target.0, target.1, source.0, source.1
            ))
        })?;
        if target_state.state_type != source_state.state_type
            || target_state.shape != source_state.shape
            || target_state.elements_per_activation != source_state.elements_per_activation
            || target_state.element_bytes != source_state.element_bytes
        {
            return Err(VulkanBindingPlanError(format!(
                "shared state {}.{} is incompatible with source {}.{}",
                target.0, target.1, source.0, source.1
            )));
        }
        bindings.insert(target.clone(), source_binding);
    }

    Ok(bindings)
}

fn shared_state_source(sharing: &str) -> Result<Option<(String, String)>, VulkanBindingPlanError> {
    let Some(source) = sharing.strip_prefix("shared_from:") else {
        return Ok(None);
    };
    parse_state_source(source).map(Some)
}

fn parse_state_source(source: &str) -> Result<(String, String), VulkanBindingPlanError> {
    let Some((pedal_id, state_id)) = source.rsplit_once('.') else {
        return Err(VulkanBindingPlanError(format!(
            "shared state source {source:?} must be PEDAL_ID.STATE_ID"
        )));
    };
    if pedal_id.is_empty() || state_id.is_empty() {
        return Err(VulkanBindingPlanError(format!(
            "shared state source {source:?} must contain non-empty pedal and state ids"
        )));
    }
    Ok((pedal_id.to_string(), state_id.to_string()))
}

type VulkanActivationBindingIndex =
    BTreeMap<(String, String), (usize, Option<usize>, Option<usize>)>;

fn activation_binding_index(
    resident_plan: &VulkanStreamCircuitResidentPlan,
) -> Result<VulkanActivationBindingIndex, VulkanBindingPlanError> {
    let mut bindings = BTreeMap::new();
    for bank in &resident_plan.activation_banks {
        for slot in &bank.slots {
            for signal_id in &slot.signal_ids {
                let key = (bank.pedal_id.clone(), signal_id.clone());
                let previous =
                    bindings.insert(key.clone(), (slot.slot, slot.bytes, slot.max_elements));
                if previous.is_some() {
                    return Err(VulkanBindingPlanError(format!(
                        "duplicate activation binding for {}.{}",
                        key.0, key.1
                    )));
                }
            }
        }
    }
    Ok(bindings)
}

fn bind_circuit(
    circuit: &CircuitActivationPlan,
    parameter_bindings: &BTreeMap<(String, String), VulkanParameterBinding>,
    state_bindings: &BTreeMap<(String, String), VulkanStateBinding>,
    activation_bindings: &VulkanActivationBindingIndex,
) -> Result<VulkanCircuitBindingPlan, VulkanBindingPlanError> {
    let nodes = circuit
        .nodes
        .iter()
        .map(|node| {
            bind_node(
                circuit,
                node,
                parameter_bindings,
                state_bindings,
                activation_bindings,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(VulkanCircuitBindingPlan {
        pedal_id: circuit.pedal_id.clone(),
        circuit_id: circuit.circuit_id.clone(),
        input_ports: circuit.input_ports.clone(),
        output_ports: circuit.output_ports.clone(),
        nodes,
    })
}

fn bind_node(
    circuit: &CircuitActivationPlan,
    node: &PlannedNode,
    parameter_bindings: &BTreeMap<(String, String), VulkanParameterBinding>,
    state_bindings: &BTreeMap<(String, String), VulkanStateBinding>,
    activation_bindings: &VulkanActivationBindingIndex,
) -> Result<VulkanNodeBinding, VulkanBindingPlanError> {
    let inputs = node
        .inputs
        .iter()
        .map(|signal_id| {
            bind_signal(
                circuit,
                node,
                signal_id,
                state_bindings,
                activation_bindings,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let outputs = node
        .outputs
        .iter()
        .map(|signal_id| {
            bind_signal(
                circuit,
                node,
                signal_id,
                state_bindings,
                activation_bindings,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let parameters = node
        .params
        .iter()
        .map(|param_id| {
            parameter_bindings
                .get(&(circuit.pedal_id.clone(), param_id.clone()))
                .cloned()
                .ok_or_else(|| {
                    VulkanBindingPlanError(format!(
                        "{} node {} parameter {:?} is not bound",
                        circuit.pedal_id, node.id, param_id
                    ))
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let state_reads = bind_state_refs(circuit, node, &node.state_reads, state_bindings)?;
    let state_writes = bind_state_refs(circuit, node, &node.state_writes, state_bindings)?;

    Ok(VulkanNodeBinding {
        node_index: node.index,
        node_id: node.id.clone(),
        op: node.op.clone(),
        specialization: node.specialization.clone(),
        inputs,
        outputs,
        parameters,
        state_reads,
        state_writes,
    })
}

fn bind_state_refs(
    circuit: &CircuitActivationPlan,
    node: &PlannedNode,
    state_ids: &[String],
    state_bindings: &BTreeMap<(String, String), VulkanStateBinding>,
) -> Result<Vec<VulkanStateBinding>, VulkanBindingPlanError> {
    state_ids
        .iter()
        .map(|state_id| {
            state_bindings
                .get(&(circuit.pedal_id.clone(), state_id.clone()))
                .cloned()
                .ok_or_else(|| {
                    VulkanBindingPlanError(format!(
                        "{} node {} state {:?} is not bound",
                        circuit.pedal_id, node.id, state_id
                    ))
                })
        })
        .collect()
}

fn bind_signal(
    circuit: &CircuitActivationPlan,
    node: &PlannedNode,
    signal_id: &str,
    state_bindings: &BTreeMap<(String, String), VulkanStateBinding>,
    activation_bindings: &VulkanActivationBindingIndex,
) -> Result<VulkanSignalBinding, VulkanBindingPlanError> {
    let signal = circuit.signal(signal_id).ok_or_else(|| {
        VulkanBindingPlanError(format!(
            "{} node {} signal {:?} is not planned",
            circuit.pedal_id, node.id, signal_id
        ))
    })?;

    let resource = if signal.is_boundary_output {
        VulkanSignalResource::BoundaryOutput
    } else {
        match signal.storage {
            SignalStorage::Boundary => VulkanSignalResource::BoundaryInput,
            SignalStorage::State => {
                let state = state_bindings
                    .get(&(circuit.pedal_id.clone(), signal_id.to_string()))
                    .ok_or_else(|| {
                        VulkanBindingPlanError(format!(
                            "{} signal {:?} has no state buffer binding",
                            circuit.pedal_id, signal_id
                        ))
                    })?;
                VulkanSignalResource::StateBuffer {
                    pedal_id: state.pedal_id.clone(),
                    state_id: state.state_id.clone(),
                    static_bytes: state.static_bytes,
                    bytes_per_activation: state.bytes_per_activation,
                }
            }
            SignalStorage::Activation => {
                let (slot, bytes, max_elements) = activation_bindings
                    .get(&(circuit.pedal_id.clone(), signal_id.to_string()))
                    .ok_or_else(|| {
                        VulkanBindingPlanError(format!(
                            "{} signal {:?} has no activation slot binding",
                            circuit.pedal_id, signal_id
                        ))
                    })?;
                let signal_bytes =
                    activation_signal_byte_capacity(circuit, node, signal, *bytes, *max_elements)?;
                VulkanSignalResource::ActivationSlot {
                    pedal_id: circuit.pedal_id.clone(),
                    slot: *slot,
                    bytes: *bytes,
                    signal_bytes,
                }
            }
            SignalStorage::StateView => {
                let state_id = state_view_state_id(circuit, signal_id)?;
                let state = state_bindings
                    .get(&(circuit.pedal_id.clone(), state_id.clone()))
                    .ok_or_else(|| {
                        VulkanBindingPlanError(format!(
                            "{} state-view signal {:?} has no state buffer binding for {:?}",
                            circuit.pedal_id, signal_id, state_id
                        ))
                    })?;
                VulkanSignalResource::StateView {
                    pedal_id: state.pedal_id.clone(),
                    state_id: state.state_id.clone(),
                    static_bytes: state.static_bytes,
                    bytes_per_activation: state.bytes_per_activation,
                }
            }
        }
    };

    Ok(VulkanSignalBinding {
        signal_id: signal_id.to_string(),
        resource,
    })
}

fn activation_signal_byte_capacity(
    circuit: &CircuitActivationPlan,
    node: &PlannedNode,
    signal: &crate::stream_plan::PlannedSignal,
    slot_bytes: Option<usize>,
    slot_max_elements: Option<usize>,
) -> Result<Option<usize>, VulkanBindingPlanError> {
    let Some(signal_elements) = signal.shape.as_deref().and_then(product) else {
        return Ok(None);
    };
    let (Some(slot_bytes), Some(slot_max_elements)) = (slot_bytes, slot_max_elements) else {
        return Ok(None);
    };
    if slot_max_elements == 0 {
        if slot_bytes == 0 && signal_elements == 0 {
            return Ok(Some(0));
        }
        return Err(VulkanBindingPlanError(format!(
            "{} node {} activation signal {:?} has an invalid zero-element slot of {slot_bytes} bytes",
            circuit.pedal_id, node.id, signal.id
        )));
    }
    if slot_bytes % slot_max_elements != 0 {
        return Err(VulkanBindingPlanError(format!(
            "{} node {} activation signal {:?} slot capacity {slot_bytes} is not divisible by {slot_max_elements} elements",
            circuit.pedal_id, node.id, signal.id
        )));
    }
    let element_bytes = slot_bytes / slot_max_elements;
    let signal_bytes = signal_elements.checked_mul(element_bytes).ok_or_else(|| {
        VulkanBindingPlanError(format!(
            "{} node {} activation signal {:?} byte capacity overflowed",
            circuit.pedal_id, node.id, signal.id
        ))
    })?;
    if signal_bytes > slot_bytes {
        return Err(VulkanBindingPlanError(format!(
            "{} node {} activation signal {:?} requires {signal_bytes} bytes, exceeding slot capacity {slot_bytes}",
            circuit.pedal_id, node.id, signal.id
        )));
    }
    Ok(Some(signal_bytes))
}

fn state_view_state_id(
    circuit: &CircuitActivationPlan,
    signal_id: &str,
) -> Result<String, VulkanBindingPlanError> {
    let signal = circuit.signal(signal_id).ok_or_else(|| {
        VulkanBindingPlanError(format!(
            "{} signal {:?} is not planned",
            circuit.pedal_id, signal_id
        ))
    })?;
    let SignalProducer::Node { node_id } = &signal.producer else {
        return Err(VulkanBindingPlanError(format!(
            "{} state-view signal {:?} is not produced by a node",
            circuit.pedal_id, signal_id
        )));
    };
    let producer = circuit
        .nodes
        .iter()
        .find(|node| &node.id == node_id)
        .ok_or_else(|| {
            VulkanBindingPlanError(format!(
                "{} state-view signal {:?} producer {:?} is not planned",
                circuit.pedal_id, signal_id, node_id
            ))
        })?;
    producer
        .state_writes
        .first()
        .or_else(|| producer.state_reads.first())
        .cloned()
        .ok_or_else(|| {
            VulkanBindingPlanError(format!(
                "{} state-view signal {:?} producer {:?} does not reference state",
                circuit.pedal_id, signal_id, node_id
            ))
        })
}

