fn infer_node_output_shapes(
    pedal_id: &str,
    node: &CircuitNode,
    signals: &BTreeMap<String, PlannedSignal>,
    params: &BTreeMap<String, ParameterRef>,
    tensor_index: Option<&TensorIndex>,
) -> Result<Vec<Option<Vec<usize>>>, CircuitPlanError> {
    let outputs = node.outputs.len();
    let unknown = || Ok(vec![None; outputs]);

    match node.op.as_str() {
        "rms_norm"
        | "rms_norm_per_head"
        | "rms_norm_per_head_unscaled"
        | "silu"
        | "gelu_tanh"
        | "rotary_position_embedding"
        | "scalar_multiply"
        | "softplus_multiply" => Ok(repeat_shape(first_input_shape(node, signals), outputs)),
        "per_layer_embedding" => {
            let output_shape = attr_usize(node, "per_layer_width").map(|width| vec![width]);
            Ok(repeat_shape(output_shape, outputs))
        }
        "multiply"
        | "residual_add"
        | "scaled_residual_add"
        | "silu_multiply"
        | "sigmoid_multiply" => Ok(repeat_shape(
            compatible_input_shape(pedal_id, node, signals)?,
            outputs,
        )),
        "sigmoid_scalar_multiply" => Ok(repeat_shape(first_input_shape(node, signals), outputs)),
        "parallel_head_norm_rope_2way" => {
            if node.inputs.len() != 2 || node.outputs.len() != 2 || node.params.len() != 2 {
                return Err(CircuitPlanError(format!(
                    "{} node {} requires two head-norm/rope inputs, outputs, and parameters",
                    pedal_id, node.id
                )));
            }
            Ok(node
                .inputs
                .iter()
                .map(|input| signals.get(input).and_then(|signal| signal.shape.clone()))
                .collect())
        }
        "parallel_linear_2way" | "parallel_linear_3way" => {
            infer_parallel_linear_output_shapes(pedal_id, node, signals, params, tensor_index)
        }
        "linear" | "linear_residual" | "linear_projection" | "parallel_linear_silu_multiply" => {
            infer_linear_output_shapes(pedal_id, node, signals, params, tensor_index)
        }
        "concatenate" => {
            let mut width = 0usize;
            for input in &node.inputs {
                let shape = signals
                    .get(input)
                    .and_then(|signal| signal.shape.as_ref())
                    .ok_or_else(|| {
                        CircuitPlanError(format!(
                            "{} node {} concatenate input {} has no known shape",
                            pedal_id, node.id, input
                        ))
                    })?;
                let [input_width] = shape.as_slice() else {
                    return Err(CircuitPlanError(format!(
                        "{} node {} concatenate input {} must be one-dimensional",
                        pedal_id, node.id, input
                    )));
                };
                width = width.checked_add(*input_width).ok_or_else(|| {
                    CircuitPlanError(format!(
                        "{} node {} concatenate width overflowed",
                        pedal_id, node.id
                    ))
                })?;
            }
            Ok(repeat_shape(Some(vec![width]), outputs))
        }
        "linear_split_3way" => {
            infer_linear_split_output_shapes(pedal_id, node, signals, params, tensor_index)
        }
        "split" => infer_split_output_shapes(pedal_id, node, signals),
        "rolling_state_update" => {
            let state_shape = node
                .inputs
                .get(1)
                .and_then(|input| signals.get(input))
                .and_then(|signal| signal.shape.clone());
            Ok(repeat_shape(state_shape, outputs))
        }
        "depthwise_conv1d" => {
            let output_shape = attr_usize(node, "groups")
                .map(|groups| vec![groups])
                .or_else(|| {
                    first_input_shape(node, signals)
                        .and_then(|shape| shape.last().copied().map(|last| vec![last]))
                });
            Ok(repeat_shape(output_shape, outputs))
        }
        "multiply_rolling_depthwise" | "multiply_rolling_depthwise_gate" => {
            let output_shape = node
                .attrs
                .get("depthwise")
                .and_then(|attrs| attrs.get("groups"))
                .and_then(|value| value.as_u64())
                .and_then(|value| usize::try_from(value).ok())
                .map(|groups| vec![groups]);
            Ok(repeat_shape(output_shape, outputs))
        }
        "linear_split_recurrent_depthwise_gate" => {
            let output_shape = node
                .attrs
                .get("recurrent")
                .and_then(|attrs| attrs.get("depthwise"))
                .and_then(|attrs| attrs.get("groups"))
                .and_then(|value| value.as_u64())
                .and_then(|value| usize::try_from(value).ok())
                .map(|groups| vec![groups]);
            Ok(repeat_shape(output_shape, outputs))
        }
        "causal_conv1d_silu" => Ok(repeat_shape(first_input_shape(node, signals), outputs)),
        "gated_delta_step" => {
            let output_shape = attr_usize(node, "value_heads")
                .zip(attr_usize(node, "value_head_width"))
                .map(|(heads, width)| vec![heads * width]);
            Ok(repeat_shape(output_shape, outputs))
        }
        "rg_lru_step" => {
            let output_shape = attr_usize(node, "width").map(|width| vec![width]);
            Ok(repeat_shape(output_shape, outputs))
        }
        "moe_topk" => {
            let output_shape = attr_usize(node, "experts_per_token").map(|routes| vec![routes, 2]);
            Ok(repeat_shape(output_shape, outputs))
        }
        "sparse_moe_gate_up" => {
            let output_shape = attr_usize(node, "experts_per_token")
                .zip(attr_usize(node, "intermediate_size"))
                .map(|(routes, intermediate)| vec![routes, intermediate]);
            Ok(repeat_shape(output_shape, outputs))
        }
        "sparse_moe_down" => {
            let output_shape = attr_usize(node, "experts_per_token")
                .zip(attr_usize(node, "hidden_size"))
                .map(|(routes, hidden)| vec![routes, hidden]);
            Ok(repeat_shape(output_shape, outputs))
        }
        "moe_reduce" => {
            let output_shape = attr_usize(node, "hidden_size").map(|hidden| vec![hidden]);
            Ok(repeat_shape(output_shape, outputs))
        }
        "append_state_update" => unknown(),
        "scaled_dot_product_attention" | "append_scaled_dot_product_attention" => {
            Ok(repeat_shape(first_input_shape(node, signals), outputs))
        }
        _ => unknown(),
    }
}

fn infer_linear_output_shapes(
    pedal_id: &str,
    node: &CircuitNode,
    signals: &BTreeMap<String, PlannedSignal>,
    params: &BTreeMap<String, ParameterRef>,
    tensor_index: Option<&TensorIndex>,
) -> Result<Vec<Option<Vec<usize>>>, CircuitPlanError> {
    let Some(tensor_index) = tensor_index else {
        return Ok(vec![None; node.outputs.len()]);
    };
    let Some(param_id) = node.params.first() else {
        return Ok(vec![None; node.outputs.len()]);
    };
    let Some(parameter) = params.get(param_id) else {
        return Ok(vec![None; node.outputs.len()]);
    };
    let Some(tensor) = parameter.tensor.as_deref() else {
        return Ok(vec![None; node.outputs.len()]);
    };
    let Some(weight_shape) = tensor_index.tensor_shape(tensor) else {
        return Ok(vec![None; node.outputs.len()]);
    };
    if weight_shape.len() != 2 {
        return Ok(vec![None; node.outputs.len()]);
    }

    let output_width = weight_shape[0];
    let input_width = weight_shape[1];
    let output_shape = match first_input_shape(node, signals) {
        Some(mut input_shape) => {
            let Some(last_dim) = input_shape.last_mut() else {
                return Ok(vec![None; node.outputs.len()]);
            };
            if *last_dim != input_width {
                return Err(CircuitPlanError(format!(
                    "{} node {} linear input width {} does not match parameter {:?} width {}",
                    pedal_id, node.id, *last_dim, param_id, input_width
                )));
            }
            *last_dim = output_width;
            Some(input_shape)
        }
        None => Some(vec![output_width]),
    };

    Ok(repeat_shape(output_shape, node.outputs.len()))
}

fn infer_parallel_linear_output_shapes(
    pedal_id: &str,
    node: &CircuitNode,
    signals: &BTreeMap<String, PlannedSignal>,
    params: &BTreeMap<String, ParameterRef>,
    tensor_index: Option<&TensorIndex>,
) -> Result<Vec<Option<Vec<usize>>>, CircuitPlanError> {
    let expected_branch_count = match node.op.as_str() {
        "parallel_linear_2way" => 2,
        "parallel_linear_3way" => 3,
        _ => unreachable!("parallel-linear shape inference called for {}", node.op),
    };
    let declared_branch_count = attr_usize(node, "branch_count");
    let branch_parameter_counts =
        parallel_linear_branch_parameter_counts(node, expected_branch_count)?;
    if node.outputs.len() != expected_branch_count
        || declared_branch_count != Some(expected_branch_count)
    {
        return Err(CircuitPlanError(format!(
            "{} node {} declares {:?} parallel-linear branches for {} parameters and {} outputs; expected {}",
            pedal_id,
            node.id,
            declared_branch_count,
            node.params.len(),
            node.outputs.len(),
            expected_branch_count
        )));
    }
    if branch_parameter_counts.iter().sum::<usize>() != node.params.len() {
        return Err(CircuitPlanError(format!(
            "{} node {} parallel-linear branch parameter counts {:?} do not cover {} parameters",
            pedal_id,
            node.id,
            branch_parameter_counts,
            node.params.len()
        )));
    }
    let Some(tensor_index) = tensor_index else {
        return Ok(vec![None; node.outputs.len()]);
    };
    let input_shape = first_input_shape(node, signals);
    let input_width = input_shape.as_ref().and_then(|shape| shape.last()).copied();
    let mut branch_weight_ids = Vec::with_capacity(expected_branch_count);
    let mut offset = 0usize;
    for count in branch_parameter_counts {
        branch_weight_ids.push(&node.params[offset]);
        offset += count;
    }
    branch_weight_ids
        .into_iter()
        .map(|param_id| {
            let parameter = params.get(param_id).ok_or_else(|| {
                CircuitPlanError(format!(
                    "{} node {} cannot resolve parallel-linear parameter {:?}",
                    pedal_id, node.id, param_id
                ))
            })?;
            let tensor = parameter.tensor.as_deref().ok_or_else(|| {
                CircuitPlanError(format!(
                    "{} node {} parallel-linear parameter {:?} has no tensor",
                    pedal_id, node.id, param_id
                ))
            })?;
            let weight_shape = tensor_index.tensor_shape(tensor).ok_or_else(|| {
                CircuitPlanError(format!(
                    "{} node {} parallel-linear tensor {:?} has no shape",
                    pedal_id, node.id, tensor
                ))
            })?;
            if weight_shape.len() != 2 {
                return Ok(None);
            }
            if input_width.is_some_and(|width| width != weight_shape[1]) {
                return Err(CircuitPlanError(format!(
                    "{} node {} parallel-linear input width {:?} does not match parameter {:?} width {}",
                    pedal_id, node.id, input_width, param_id, weight_shape[1]
                )));
            }
            let mut output_shape = input_shape
                .clone()
                .unwrap_or_else(|| vec![weight_shape[0]]);
            if let Some(last) = output_shape.last_mut() {
                *last = weight_shape[0];
            }
            Ok(Some(output_shape))
        })
        .collect()
}

fn parallel_linear_branch_parameter_counts(
    node: &CircuitNode,
    expected_branch_count: usize,
) -> Result<Vec<usize>, CircuitPlanError> {
    let Some(value) = node.attrs.get("branch_parameter_counts") else {
        return Ok(vec![1; expected_branch_count]);
    };
    let Some(counts) = value.as_array() else {
        return Err(CircuitPlanError(format!(
            "parallel-linear node {} branch_parameter_counts must be an array",
            node.id
        )));
    };
    if counts.len() != expected_branch_count {
        return Err(CircuitPlanError(format!(
            "parallel-linear node {} branch_parameter_counts has {} entries; expected {}",
            node.id,
            counts.len(),
            expected_branch_count
        )));
    }
    counts
        .iter()
        .map(|count| {
            count
                .as_u64()
                .and_then(|value| usize::try_from(value).ok())
                .filter(|value| *value > 0)
                .ok_or_else(|| {
                    CircuitPlanError(format!(
                        "parallel-linear node {} has invalid branch parameter count {:?}",
                        node.id, count
                    ))
                })
        })
        .collect()
}

fn infer_split_output_shapes(
    pedal_id: &str,
    node: &CircuitNode,
    signals: &BTreeMap<String, PlannedSignal>,
) -> Result<Vec<Option<Vec<usize>>>, CircuitPlanError> {
    let Some(mut input_shape) = first_input_shape(node, signals) else {
        return Ok(vec![None; node.outputs.len()]);
    };
    let Some(channel_dim) = input_shape.last_mut() else {
        return Ok(vec![None; node.outputs.len()]);
    };
    if let Some(part_widths) = node
        .attrs
        .get("part_widths")
        .and_then(|value| value.as_array())
    {
        let widths = part_widths
            .iter()
            .map(|value| value.as_u64().and_then(|width| usize::try_from(width).ok()))
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| {
                CircuitPlanError(format!(
                    "{} node {} has non-integer split part widths",
                    pedal_id, node.id
                ))
            })?;
        if widths.len() != node.outputs.len() || widths.iter().sum::<usize>() != *channel_dim {
            return Err(CircuitPlanError(format!(
                "{} node {} cannot split shape {:?} into widths {:?}",
                pedal_id,
                node.id,
                first_input_shape(node, signals),
                widths
            )));
        }
        return Ok(widths
            .into_iter()
            .map(|width| {
                let mut shape = input_shape.clone();
                *shape.last_mut().unwrap() = width;
                Some(shape)
            })
            .collect());
    }
    if node.outputs.is_empty() || *channel_dim % node.outputs.len() != 0 {
        return Err(CircuitPlanError(format!(
            "{} node {} cannot split shape {:?} across {} outputs",
            pedal_id,
            node.id,
            first_input_shape(node, signals),
            node.outputs.len()
        )));
    }
    *channel_dim /= node.outputs.len();
    Ok(repeat_shape(Some(input_shape), node.outputs.len()))
}

fn infer_linear_split_output_shapes(
    pedal_id: &str,
    node: &CircuitNode,
    signals: &BTreeMap<String, PlannedSignal>,
    params: &BTreeMap<String, ParameterRef>,
    tensor_index: Option<&TensorIndex>,
) -> Result<Vec<Option<Vec<usize>>>, CircuitPlanError> {
    let combined_shapes =
        infer_linear_output_shapes(pedal_id, node, signals, params, tensor_index)?;
    let Some(Some(combined_shape)) = combined_shapes.first() else {
        return Ok(vec![None; node.outputs.len()]);
    };
    let Some(combined_width) = combined_shape.last().copied() else {
        return Ok(vec![None; node.outputs.len()]);
    };
    let part_widths = node
        .attrs
        .get("part_widths")
        .and_then(|value| value.as_array())
        .and_then(|widths| {
            widths
                .iter()
                .map(|value| value.as_u64().and_then(|width| usize::try_from(width).ok()))
                .collect::<Option<Vec<_>>>()
        })
        .ok_or_else(|| {
            CircuitPlanError(format!(
                "{} node {} requires integer linear-split part widths",
                pedal_id, node.id
            ))
        })?;
    if part_widths.len() != node.outputs.len()
        || part_widths.iter().sum::<usize>() != combined_width
    {
        return Err(CircuitPlanError(format!(
            "{} node {} cannot split linear output shape {:?} into widths {:?}",
            pedal_id, node.id, combined_shape, part_widths
        )));
    }
    Ok(part_widths
        .into_iter()
        .map(|width| {
            let mut shape = combined_shape.clone();
            *shape.last_mut().unwrap() = width;
            Some(shape)
        })
        .collect())
}

fn compatible_input_shape(
    pedal_id: &str,
    node: &CircuitNode,
    signals: &BTreeMap<String, PlannedSignal>,
) -> Result<Option<Vec<usize>>, CircuitPlanError> {
    let mut known_shape = None;
    for input in &node.inputs {
        let shape = signals.get(input).and_then(|signal| signal.shape.clone());
        if let Some(shape) = shape {
            if let Some(existing) = &known_shape {
                if existing != &shape {
                    return Err(CircuitPlanError(format!(
                        "{} node {} input {:?} shape {:?} does not match {:?}",
                        pedal_id, node.id, input, shape, existing
                    )));
                }
            } else {
                known_shape = Some(shape);
            }
        }
    }
    Ok(known_shape)
}

fn first_input_shape(
    node: &CircuitNode,
    signals: &BTreeMap<String, PlannedSignal>,
) -> Option<Vec<usize>> {
    node.inputs
        .first()
        .and_then(|input| signals.get(input))
        .and_then(|signal| signal.shape.clone())
}

fn repeat_shape(shape: Option<Vec<usize>>, count: usize) -> Vec<Option<Vec<usize>>> {
    (0..count).map(|_| shape.clone()).collect()
}

fn attr_usize(node: &CircuitNode, attr: &str) -> Option<usize> {
    node.attrs
        .get(attr)
        .and_then(|value| value.as_u64())
        .and_then(|value| usize::try_from(value).ok())
}

fn product(shape: &[usize]) -> Option<usize> {
    shape
        .iter()
        .try_fold(1usize, |total, value| total.checked_mul(*value))
}

fn node_output_storage(node: &CircuitNode) -> SignalStorage {
    match node.op.as_str() {
        "append_state_update" | "rolling_state_update" => SignalStorage::StateView,
        _ => SignalStorage::Activation,
    }
}
