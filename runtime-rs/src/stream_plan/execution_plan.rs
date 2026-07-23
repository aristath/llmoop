#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StreamCircuitExecutionPlan {
    pub topology: String,
    pub circuits: Vec<CircuitActivationPlan>,
}

impl StreamCircuitExecutionPlan {
    pub fn from_graph(graph: &ResolvedLoweredExecutionGraph) -> Result<Self, CircuitPlanError> {
        let mut circuits = Vec::with_capacity(graph.circuits.len());
        for artifact in &graph.circuits {
            circuits.push(CircuitActivationPlan::from_artifact(artifact)?);
        }
        Ok(Self {
            topology: graph.index.graph.topology.clone(),
            circuits,
        })
    }

    pub fn from_graph_with_tensor_index(
        graph: &ResolvedLoweredExecutionGraph,
        tensor_index: &TensorIndex,
    ) -> Result<Self, CircuitPlanError> {
        let mut circuits = Vec::with_capacity(graph.circuits.len());
        for artifact in &graph.circuits {
            circuits.push(CircuitActivationPlan::from_artifact_with_tensor_index(
                artifact,
                tensor_index,
            )?);
        }
        Ok(Self {
            topology: graph.index.graph.topology.clone(),
            circuits,
        })
    }

    pub fn total_node_count(&self) -> usize {
        self.circuits
            .iter()
            .map(|circuit| circuit.nodes.len())
            .sum()
    }

    pub fn temporary_signal_count(&self) -> usize {
        self.circuits
            .iter()
            .map(|circuit| circuit.temporary_signals.len())
            .sum()
    }

    pub fn state_view_signal_count(&self) -> usize {
        self.circuits
            .iter()
            .map(|circuit| circuit.state_view_signals.len())
            .sum()
    }

    pub fn produced_signal_count(&self) -> usize {
        self.circuits
            .iter()
            .map(|circuit| circuit.produced_signal_count())
            .sum()
    }

    pub fn operator_counts(&self) -> BTreeMap<String, usize> {
        let mut counts = BTreeMap::new();
        for circuit in &self.circuits {
            for node in &circuit.nodes {
                *counts.entry(node.op.clone()).or_insert(0) += 1;
            }
        }
        counts
    }

    pub fn state_type_counts(&self) -> BTreeMap<String, usize> {
        let mut counts = BTreeMap::new();
        for circuit in &self.circuits {
            for state in &circuit.state_ports {
                *counts.entry(state.state_type.clone()).or_insert(0) += 1;
            }
        }
        counts
    }

    pub fn layer_local_activation_slot_count(&self) -> usize {
        self.circuits
            .iter()
            .map(|circuit| circuit.activation_frame_plan().slot_count)
            .sum()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StreamCircuitResourcePlan {
    pub circuit_count: usize,
    pub node_count: usize,
    pub parameter_ref_count: usize,
    pub parameters: Vec<PlannedParameterResource>,
    pub transducer_parameter_ref_count: usize,
    pub transducer_parameters: Vec<PlannedParameterResource>,
    pub state_allocations: Vec<PlannedStateResource>,
    pub activation_banks: Vec<PlannedActivationSlotBank>,
    pub temporary_signal_count: usize,
    pub state_view_signal_count: usize,
    pub layer_local_activation_slot_count: usize,
    pub unknown_temporary_shape_count: usize,
    pub unknown_state_view_shape_count: usize,
}

impl StreamCircuitResourcePlan {
    pub fn from_graph(graph: &ResolvedLoweredExecutionGraph) -> Result<Self, CircuitPlanError> {
        let execution_plan = StreamCircuitExecutionPlan::from_graph(graph)?;
        Self::from_graph_and_plan(graph, &execution_plan)
    }

    pub fn from_graph_with_tensor_index(
        graph: &ResolvedLoweredExecutionGraph,
        tensor_index: &TensorIndex,
    ) -> Result<Self, CircuitPlanError> {
        let execution_plan =
            StreamCircuitExecutionPlan::from_graph_with_tensor_index(graph, tensor_index)?;
        Self::from_graph_and_plan(graph, &execution_plan)
    }

    pub fn from_graph_and_plan(
        graph: &ResolvedLoweredExecutionGraph,
        execution_plan: &StreamCircuitExecutionPlan,
    ) -> Result<Self, CircuitPlanError> {
        if graph.circuits.len() != execution_plan.circuits.len() {
            return Err(CircuitPlanError(format!(
                "graph circuit count {} does not match plan circuit count {}",
                graph.circuits.len(),
                execution_plan.circuits.len()
            )));
        }

        let mut parameter_ref_count = 0;
        let mut parameters_by_tensor: BTreeMap<String, Vec<PlannedParameterUse>> = BTreeMap::new();
        let mut transducer_parameter_ref_count = 0;
        let mut transducer_parameters_by_tensor: BTreeMap<String, Vec<PlannedParameterUse>> =
            BTreeMap::new();
        let mut state_allocations = Vec::new();
        let mut activation_banks = Vec::new();
        let mut unknown_temporary_shape_count = 0;
        let mut unknown_state_view_shape_count = 0;

        for (artifact, activation_plan) in graph.circuits.iter().zip(&execution_plan.circuits) {
            if artifact.component.id != activation_plan.component_id {
                return Err(CircuitPlanError(format!(
                    "graph component {:?} does not match activation plan component {:?}",
                    artifact.component.id, activation_plan.component_id
                )));
            }
            if artifact.circuit.id != activation_plan.circuit_id {
                return Err(CircuitPlanError(format!(
                    "graph circuit {:?} does not match activation plan circuit {:?}",
                    artifact.circuit.id, activation_plan.circuit_id
                )));
            }

            for (param_id, parameter) in &artifact.params.refs {
                parameter_ref_count += 1;
                let tensor = parameter.tensor.clone().ok_or_else(|| {
                    CircuitPlanError(format!(
                        "{} parameter {:?} has no source tensor",
                        artifact.component.id, param_id
                    ))
                })?;
                parameters_by_tensor
                    .entry(tensor)
                    .or_default()
                    .push(PlannedParameterUse {
                        component_id: artifact.component.id.clone(),
                        circuit_id: artifact.circuit.id.clone(),
                        param_id: param_id.clone(),
                        role: parameter.role.clone(),
                        layout: artifact.params.layout.clone(),
                        storage: artifact.params.storage.clone(),
                    });
            }

            for state in &artifact.state.state_ports {
                state_allocations.push(PlannedStateResource {
                    component_id: artifact.component.id.clone(),
                    circuit_id: artifact.circuit.id.clone(),
                    state_id: state.id.clone(),
                    state_type: state.state_type.clone(),
                    shape: state.shape.clone(),
                    elements_per_activation: state.elements_per_activation(),
                    max_dynamic_activations: state.max_dynamic_activations,
                    update: state.update.clone(),
                    growth: state.growth.clone(),
                    sharing: state.sharing.clone(),
                    owner: state.owner.clone(),
                    layout: state.layout.clone(),
                    source_layout: state.source_layout.clone(),
                    element_bytes: state
                        .extra
                        .get("dtype")
                        .and_then(serde_json::Value::as_str)
                        .map(state_dtype_bytes)
                        .transpose()?,
                });
            }

            unknown_temporary_shape_count += activation_plan
                .temporary_signals
                .iter()
                .filter(|signal_id| {
                    activation_plan
                        .signal(signal_id)
                        .and_then(|signal| signal.shape.as_ref())
                        .is_none()
                })
                .count();
            unknown_state_view_shape_count += activation_plan
                .state_view_signals
                .iter()
                .filter(|signal_id| {
                    activation_plan
                        .signal(signal_id)
                        .and_then(|signal| signal.shape.as_ref())
                        .is_none()
                })
                .count();

            let activation_frame = activation_plan.activation_frame_plan();
            activation_banks.push(PlannedActivationSlotBank {
                component_id: artifact.component.id.clone(),
                circuit_id: artifact.circuit.id.clone(),
                temporary_signal_count: activation_plan.temporary_signals.len(),
                state_view_signal_count: activation_plan.state_view_signals.len(),
                slot_count: activation_frame.slot_count,
                slots: planned_activation_slots(activation_plan, &activation_frame),
                assignments: activation_frame.assignments,
            });
        }

        transducer_parameter_ref_count += collect_transducer_component_parameters(
            "input_transducer",
            &graph.index.graph.input_transducer,
            "input_transducer",
            &mut transducer_parameters_by_tensor,
        )?;
        transducer_parameter_ref_count += collect_output_transducer_parameters(
            &graph.index.graph.output_transducer,
            &mut transducer_parameters_by_tensor,
        )?;

        let parameters = parameters_by_tensor
            .into_iter()
            .map(|(tensor, uses)| PlannedParameterResource { tensor, uses })
            .collect();
        let transducer_parameters = transducer_parameters_by_tensor
            .into_iter()
            .map(|(tensor, uses)| PlannedParameterResource { tensor, uses })
            .collect();

        Ok(Self {
            circuit_count: graph.circuits.len(),
            node_count: execution_plan.total_node_count(),
            parameter_ref_count,
            parameters,
            transducer_parameter_ref_count,
            transducer_parameters,
            state_allocations,
            activation_banks,
            temporary_signal_count: execution_plan.temporary_signal_count(),
            state_view_signal_count: execution_plan.state_view_signal_count(),
            layer_local_activation_slot_count: execution_plan.layer_local_activation_slot_count(),
            unknown_temporary_shape_count,
            unknown_state_view_shape_count,
        })
    }

    pub fn unique_parameter_tensor_count(&self) -> usize {
        self.parameters.len()
    }

    pub fn unique_transducer_parameter_tensor_count(&self) -> usize {
        self.transducer_parameters.len()
    }

    pub fn stream_state_count(&self) -> usize {
        self.state_allocations.len()
    }

    pub fn intermediate_activation_shapes_known(&self) -> bool {
        self.unknown_temporary_shape_count == 0
    }
}

fn collect_output_transducer_parameters(
    output_transducer: &serde_json::Value,
    parameters_by_tensor: &mut BTreeMap<String, Vec<PlannedParameterUse>>,
) -> Result<usize, CircuitPlanError> {
    let Some(components) = output_transducer
        .get("components")
        .and_then(serde_json::Value::as_array)
    else {
        return collect_transducer_component_parameters(
            "output_transducer",
            output_transducer,
            "output_transducer",
            parameters_by_tensor,
        );
    };

    let mut parameter_ref_count = 0usize;
    for (component_index, component) in components.iter().enumerate() {
        parameter_ref_count += collect_transducer_component_parameters(
            "output_transducer",
            component,
            &format!("component_{component_index}"),
            parameters_by_tensor,
        )?;
    }
    Ok(parameter_ref_count)
}

fn collect_transducer_component_parameters(
    transducer_id: &str,
    component: &serde_json::Value,
    fallback_component_id: &str,
    parameters_by_tensor: &mut BTreeMap<String, Vec<PlannedParameterUse>>,
) -> Result<usize, CircuitPlanError> {
    if component.is_null() {
        return Ok(0);
    }

    let component_id = component
        .get("id")
        .and_then(serde_json::Value::as_str)
        .unwrap_or(fallback_component_id);
    let component_type = component
        .get("type")
        .and_then(serde_json::Value::as_str)
        .map(str::to_string);
    let Some(params) = component
        .get("params")
        .and_then(serde_json::Value::as_object)
    else {
        return Ok(0);
    };

    let mut parameter_ref_count = 0usize;
    for (param_id, param_ref) in params {
        let tensor = param_ref
            .get("tensor")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| {
                CircuitPlanError(format!(
                    "{transducer_id}.{component_id} transducer parameter {param_id:?} has no source tensor"
                ))
            })?;
        parameter_ref_count += 1;
        parameters_by_tensor
            .entry(tensor.to_string())
            .or_default()
            .push(PlannedParameterUse {
                component_id: format!("{transducer_id}.{component_id}"),
                circuit_id: transducer_id.to_string(),
                param_id: param_id.clone(),
                role: component_type.clone(),
                layout: "transducer".to_string(),
                storage: "source_tensor_refs".to_string(),
            });
    }

    Ok(parameter_ref_count)
}
