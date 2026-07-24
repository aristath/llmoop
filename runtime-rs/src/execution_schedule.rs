use std::error::Error;
use std::fmt::{Display, Formatter};
use std::ops::Range;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RuntimeExecutionCost {
    pub work_units: u64,
    pub memory_bytes: u64,
    pub dispatches: u64,
}

impl RuntimeExecutionCost {
    pub const fn new(work_units: u64, memory_bytes: u64, dispatches: u64) -> Self {
        Self {
            work_units,
            memory_bytes,
            dispatches,
        }
    }

    pub fn checked_add(self, other: Self) -> Option<Self> {
        Some(Self {
            work_units: self.work_units.checked_add(other.work_units)?,
            memory_bytes: self.memory_bytes.checked_add(other.memory_bytes)?,
            dispatches: self.dispatches.checked_add(other.dispatches)?,
        })
    }

    fn exceeds(self, budget: RuntimeExecutionQuantumBudget) -> bool {
        budget
            .max_work_units
            .is_some_and(|limit| self.work_units > limit)
            || budget
                .max_memory_bytes
                .is_some_and(|limit| self.memory_bytes > limit)
            || budget
                .max_dispatches
                .is_some_and(|limit| self.dispatches > limit)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RuntimeExecutionQuantumBudget {
    pub max_work_units: Option<u64>,
    pub max_memory_bytes: Option<u64>,
    pub max_dispatches: Option<u64>,
    pub max_regions: Option<usize>,
}

impl RuntimeExecutionQuantumBudget {
    pub const fn one_region() -> Self {
        Self {
            max_work_units: None,
            max_memory_bytes: None,
            max_dispatches: None,
            max_regions: Some(1),
        }
    }

    pub fn validate(self) -> Result<Self, RuntimeExecutionScheduleError> {
        if self.max_work_units == Some(0)
            || self.max_memory_bytes == Some(0)
            || self.max_dispatches == Some(0)
            || self.max_regions == Some(0)
        {
            return Err(RuntimeExecutionScheduleError::ZeroBudget);
        }
        if self.max_work_units.is_none()
            && self.max_memory_bytes.is_none()
            && self.max_dispatches.is_none()
            && self.max_regions.is_none()
        {
            return Err(RuntimeExecutionScheduleError::UnboundedBudget);
        }
        Ok(self)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RuntimeExecutionRegion {
    pub id: String,
    pub component_id: String,
    pub cost: RuntimeExecutionCost,
    pub safe_yield_after: bool,
    pub commits_state_after: bool,
}

impl RuntimeExecutionRegion {
    pub fn new(
        id: impl Into<String>,
        component_id: impl Into<String>,
        cost: RuntimeExecutionCost,
    ) -> Self {
        Self {
            id: id.into(),
            component_id: component_id.into(),
            cost,
            safe_yield_after: true,
            commits_state_after: false,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RuntimeExecutionQuantum {
    pub region_range: Range<usize>,
    pub component_ids: Vec<String>,
    pub cost: RuntimeExecutionCost,
    pub commits_state_after: bool,
}

impl RuntimeExecutionQuantum {
    pub fn region_count(&self) -> usize {
        self.region_range.end - self.region_range.start
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RuntimeExecutionSchedule {
    pub quanta: Vec<RuntimeExecutionQuantum>,
    pub total_cost: RuntimeExecutionCost,
}

impl RuntimeExecutionSchedule {
    pub fn linear(
        regions: &[RuntimeExecutionRegion],
        budget: RuntimeExecutionQuantumBudget,
    ) -> Result<Self, RuntimeExecutionScheduleError> {
        if regions.is_empty() {
            return Err(RuntimeExecutionScheduleError::EmptyRegions);
        }
        let budget = budget.validate()?;
        let mut quanta = Vec::new();
        let mut quantum_start = 0usize;
        let mut quantum_cost = RuntimeExecutionCost::default();
        let mut quantum_components = Vec::<String>::new();
        let mut quantum_commits_state = false;
        let mut total_cost = RuntimeExecutionCost::default();

        for (region_index, region) in regions.iter().enumerate() {
            if region.cost.exceeds(budget) {
                return Err(RuntimeExecutionScheduleError::UnsplittableRegion {
                    region_id: region.id.clone(),
                    cost: region.cost,
                    budget,
                });
            }
            let candidate_cost = quantum_cost.checked_add(region.cost).ok_or_else(|| {
                RuntimeExecutionScheduleError::CostOverflow {
                    region_id: region.id.clone(),
                }
            })?;
            let candidate_region_count = region_index + 1 - quantum_start;
            let candidate_exceeds = candidate_cost.exceeds(budget)
                || budget
                    .max_regions
                    .is_some_and(|limit| candidate_region_count > limit);
            if candidate_exceeds {
                if region_index == quantum_start {
                    return Err(RuntimeExecutionScheduleError::UnsplittableRegion {
                        region_id: region.id.clone(),
                        cost: region.cost,
                        budget,
                    });
                }
                if !regions[region_index - 1].safe_yield_after {
                    return Err(RuntimeExecutionScheduleError::MissingSafeYield {
                        preceding_region_id: regions[region_index - 1].id.clone(),
                        next_region_id: region.id.clone(),
                    });
                }
                quanta.push(RuntimeExecutionQuantum {
                    region_range: quantum_start..region_index,
                    component_ids: std::mem::take(&mut quantum_components),
                    cost: quantum_cost,
                    commits_state_after: quantum_commits_state,
                });
                quantum_start = region_index;
                quantum_cost = RuntimeExecutionCost::default();
                quantum_commits_state = false;
            }

            quantum_cost = quantum_cost.checked_add(region.cost).ok_or_else(|| {
                RuntimeExecutionScheduleError::CostOverflow {
                    region_id: region.id.clone(),
                }
            })?;
            total_cost = total_cost.checked_add(region.cost).ok_or_else(|| {
                RuntimeExecutionScheduleError::CostOverflow {
                    region_id: region.id.clone(),
                }
            })?;
            if quantum_components.last() != Some(&region.component_id) {
                quantum_components.push(region.component_id.clone());
            }
            quantum_commits_state |= region.commits_state_after;
        }
        quanta.push(RuntimeExecutionQuantum {
            region_range: quantum_start..regions.len(),
            component_ids: quantum_components,
            cost: quantum_cost,
            commits_state_after: quantum_commits_state,
        });
        Ok(Self { quanta, total_cost })
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RuntimeExecutionTimingModel {
    observed_work_units: u64,
    observed_duration_ns: u64,
}

impl RuntimeExecutionTimingModel {
    pub fn observe(&mut self, cost: RuntimeExecutionCost, duration_ns: u64) {
        if cost.work_units == 0 || duration_ns == 0 {
            return;
        }
        if self.observed_work_units == 0 {
            self.observed_work_units = cost.work_units;
            self.observed_duration_ns = duration_ns;
            return;
        }

        // Keep a stable, integer EWMA. Recent mounted-graph behavior matters
        // more than old measurements after topology, placement, or signal shape
        // changes, while one noisy activation must not replace the model.
        self.observed_work_units = self
            .observed_work_units
            .saturating_mul(3)
            .saturating_add(cost.work_units)
            / 4;
        self.observed_duration_ns = self
            .observed_duration_ns
            .saturating_mul(3)
            .saturating_add(duration_ns)
            / 4;
    }

    pub fn has_observation(self) -> bool {
        self.observed_work_units != 0 && self.observed_duration_ns != 0
    }

    pub fn work_units_for_duration(self, target_duration_ns: u64) -> Option<u64> {
        if !self.has_observation() || target_duration_ns == 0 {
            return None;
        }
        let numerator =
            u128::from(self.observed_work_units).checked_mul(u128::from(target_duration_ns))?;
        let work_units = numerator / u128::from(self.observed_duration_ns);
        u64::try_from(work_units.max(1)).ok()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RuntimeExecutionScheduleError {
    EmptyRegions,
    ZeroBudget,
    UnboundedBudget,
    CostOverflow {
        region_id: String,
    },
    UnsplittableRegion {
        region_id: String,
        cost: RuntimeExecutionCost,
        budget: RuntimeExecutionQuantumBudget,
    },
    MissingSafeYield {
        preceding_region_id: String,
        next_region_id: String,
    },
}

impl Display for RuntimeExecutionScheduleError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyRegions => write!(f, "execution schedule has no regions"),
            Self::ZeroBudget => write!(f, "execution quantum budget contains a zero limit"),
            Self::UnboundedBudget => write!(f, "execution quantum budget has no limits"),
            Self::CostOverflow { region_id } => {
                write!(
                    f,
                    "execution cost overflowed while adding region {region_id:?}"
                )
            }
            Self::UnsplittableRegion {
                region_id,
                cost,
                budget,
            } => write!(
                f,
                "execution region {region_id:?} cost {cost:?} exceeds quantum budget {budget:?}; the compiled implementation must expose an internal safe yield"
            ),
            Self::MissingSafeYield {
                preceding_region_id,
                next_region_id,
            } => write!(
                f,
                "execution schedule must split between {preceding_region_id:?} and {next_region_id:?}, but the preceding region is not a safe yield point"
            ),
        }
    }
}

impl Error for RuntimeExecutionScheduleError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn region(id: &str, component: &str, work: u64, dispatches: u64) -> RuntimeExecutionRegion {
        RuntimeExecutionRegion::new(
            id,
            component,
            RuntimeExecutionCost::new(work, work * 2, dispatches),
        )
    }

    #[test]
    fn linear_schedule_respects_every_budget_dimension() {
        let regions = vec![
            region("a0", "a", 4, 1),
            region("a1", "a", 4, 1),
            region("b0", "b", 5, 2),
            region("c0", "c", 2, 1),
        ];
        let schedule = RuntimeExecutionSchedule::linear(
            &regions,
            RuntimeExecutionQuantumBudget {
                max_work_units: Some(10),
                max_memory_bytes: Some(20),
                max_dispatches: Some(3),
                max_regions: Some(3),
            },
        )
        .unwrap();

        assert_eq!(
            schedule
                .quanta
                .iter()
                .map(|quantum| quantum.region_range.clone())
                .collect::<Vec<_>>(),
            vec![0..2, 2..4]
        );
        assert_eq!(schedule.quanta[0].component_ids, vec!["a"]);
        assert_eq!(schedule.quanta[1].component_ids, vec!["b", "c"]);
        assert_eq!(schedule.total_cost, RuntimeExecutionCost::new(15, 30, 5));
    }

    #[test]
    fn linear_schedule_rejects_an_oversized_atomic_region() {
        let error = RuntimeExecutionSchedule::linear(
            &[region("large", "component", 11, 1)],
            RuntimeExecutionQuantumBudget {
                max_work_units: Some(10),
                ..RuntimeExecutionQuantumBudget::default()
            },
        )
        .unwrap_err();

        assert!(matches!(
            error,
            RuntimeExecutionScheduleError::UnsplittableRegion { region_id, .. }
                if region_id == "large"
        ));
    }

    #[test]
    fn linear_schedule_never_cuts_through_an_unsafe_boundary() {
        let mut first = region("state-write", "stateful", 6, 1);
        first.safe_yield_after = false;
        let error = RuntimeExecutionSchedule::linear(
            &[first, region("state-commit", "stateful", 6, 1)],
            RuntimeExecutionQuantumBudget {
                max_work_units: Some(10),
                ..RuntimeExecutionQuantumBudget::default()
            },
        )
        .unwrap_err();

        assert!(matches!(
            error,
            RuntimeExecutionScheduleError::MissingSafeYield {
                preceding_region_id,
                next_region_id,
            } if preceding_region_id == "state-write" && next_region_id == "state-commit"
        ));
    }

    #[test]
    fn state_commit_metadata_survives_coalescing() {
        let first = region("compute", "stateful", 2, 1);
        let mut second = region("commit", "stateful", 2, 1);
        second.commits_state_after = true;
        let schedule = RuntimeExecutionSchedule::linear(
            &[first, second],
            RuntimeExecutionQuantumBudget {
                max_regions: Some(2),
                ..RuntimeExecutionQuantumBudget::default()
            },
        )
        .unwrap();

        assert_eq!(schedule.quanta.len(), 1);
        assert!(schedule.quanta[0].commits_state_after);
    }

    #[test]
    fn timing_model_ignores_empty_samples_and_predicts_work_budget() {
        let mut timing = RuntimeExecutionTimingModel::default();
        timing.observe(RuntimeExecutionCost::default(), 100);
        timing.observe(RuntimeExecutionCost::new(100, 0, 1), 0);
        assert!(!timing.has_observation());

        timing.observe(RuntimeExecutionCost::new(1_000, 0, 1), 2_000);
        assert_eq!(timing.work_units_for_duration(500), Some(250));
        timing.observe(RuntimeExecutionCost::new(2_000, 0, 1), 2_000);
        assert_eq!(timing.observed_work_units, 1_250);
        assert_eq!(timing.observed_duration_ns, 2_000);
    }
}
