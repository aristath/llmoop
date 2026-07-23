#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CircuitParamsArtifact {
    pub schema: String,
    pub circuit: String,
    pub layout: String,
    pub storage: String,
    #[serde(default)]
    pub refs: BTreeMap<String, ParameterRef>,
}

impl CircuitParamsArtifact {
    pub fn from_json_file(path: impl AsRef<Path>) -> Result<Self, CircuitArtifactError> {
        read_json(path)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CircuitStateArtifact {
    pub schema: String,
    pub circuit: String,
    #[serde(default)]
    pub state_ports: Vec<StatePort>,
}

impl CircuitStateArtifact {
    pub fn from_json_file(path: impl AsRef<Path>) -> Result<Self, CircuitArtifactError> {
        read_json(path)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedCircuitArtifact {
    pub pedal: LoweredCircuitRef,
    pub circuit: StreamCircuit,
    pub params: CircuitParamsArtifact,
    pub state: CircuitStateArtifact,
}

impl ResolvedCircuitArtifact {
    pub fn validate(&self) -> Result<(), CircuitArtifactError> {
        self.circuit.validate_contract()?;
        if self.pedal.id != self.circuit.source.pedal_id {
            return Err(CircuitArtifactError(format!(
                "lowered circuit id {:?} does not match circuit source pedal {:?}",
                self.pedal.id, self.circuit.source.pedal_id
            )));
        }
        if self.pedal.operator_type != self.circuit.source.source_operator_type {
            return Err(CircuitArtifactError(format!(
                "lowered circuit {} operator {:?} does not match circuit source operator {:?}",
                self.pedal.id, self.pedal.operator_type, self.circuit.source.source_operator_type
            )));
        }
        if self.pedal.runtime_role != self.circuit.runtime_role {
            return Err(CircuitArtifactError(format!(
                "lowered circuit {} runtime role {:?} does not match circuit {:?}",
                self.pedal.id, self.pedal.runtime_role, self.circuit.runtime_role
            )));
        }
        if self.pedal.implementation != self.circuit.implementation {
            return Err(CircuitArtifactError(format!(
                "lowered circuit {} implementation {:?} does not match circuit {:?}",
                self.pedal.id, self.pedal.implementation, self.circuit.implementation
            )));
        }
        if self.params.schema != CIRCUIT_PARAMS_SCHEMA {
            return Err(CircuitArtifactError(format!(
                "{} params schema {:?} is unsupported",
                self.pedal.id, self.params.schema
            )));
        }
        if self.state.schema != CIRCUIT_STATE_SCHEMA {
            return Err(CircuitArtifactError(format!(
                "{} state schema {:?} is unsupported",
                self.pedal.id, self.state.schema
            )));
        }
        if self.params.circuit != self.circuit.id {
            return Err(CircuitArtifactError(format!(
                "{} params target {:?} does not match circuit {:?}",
                self.pedal.id, self.params.circuit, self.circuit.id
            )));
        }
        if self.state.circuit != self.circuit.id {
            return Err(CircuitArtifactError(format!(
                "{} state target {:?} does not match circuit {:?}",
                self.pedal.id, self.state.circuit, self.circuit.id
            )));
        }
        if self.params.refs.keys().collect::<BTreeSet<_>>()
            != self.circuit.parameters.refs.keys().collect::<BTreeSet<_>>()
        {
            return Err(CircuitArtifactError(format!(
                "{} params refs do not match circuit refs",
                self.pedal.id
            )));
        }
        if self.state.state_ports != self.circuit.state_ports {
            return Err(CircuitArtifactError(format!(
                "{} state port contracts do not match circuit state ports",
                self.pedal.id
            )));
        }
        Ok(())
    }
}
