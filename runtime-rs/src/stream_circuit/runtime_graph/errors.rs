#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CircuitPlacementError(pub String);

impl Display for CircuitPlacementError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for CircuitPlacementError {}

