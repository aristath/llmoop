use std::collections::BTreeMap;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::ops::Range;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RuntimeExecutionCost {
    pub work_units: u64,
    pub memory_bytes: u64,
    pub dispatches: u64,
    pub predicted_duration_ns: u64,
}

impl RuntimeExecutionCost {
    pub const fn new(work_units: u64, memory_bytes: u64, dispatches: u64) -> Self {
        Self {
            work_units,
            memory_bytes,
            dispatches,
            predicted_duration_ns: 0,
        }
    }

    pub const fn with_predicted_duration_ns(mut self, predicted_duration_ns: u64) -> Self {
        self.predicted_duration_ns = predicted_duration_ns;
        self
    }

    pub fn checked_add(self, other: Self) -> Option<Self> {
        Some(Self {
            work_units: self.work_units.checked_add(other.work_units)?,
            memory_bytes: self.memory_bytes.checked_add(other.memory_bytes)?,
            dispatches: self.dispatches.checked_add(other.dispatches)?,
            predicted_duration_ns: self
                .predicted_duration_ns
                .checked_add(other.predicted_duration_ns)?,
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
            || budget
                .max_predicted_duration_ns
                .is_some_and(|limit| self.predicted_duration_ns > limit)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RuntimeExecutionQuantumBudget {
    pub max_work_units: Option<u64>,
    pub max_memory_bytes: Option<u64>,
    pub max_dispatches: Option<u64>,
    pub max_predicted_duration_ns: Option<u64>,
    pub max_regions: Option<usize>,
}

impl RuntimeExecutionQuantumBudget {
    pub const fn one_region() -> Self {
        Self {
            max_work_units: None,
            max_memory_bytes: None,
            max_dispatches: None,
            max_predicted_duration_ns: None,
            max_regions: Some(1),
        }
    }

    pub fn validate(self) -> Result<Self, RuntimeExecutionScheduleError> {
        if self.max_work_units == Some(0)
            || self.max_memory_bytes == Some(0)
            || self.max_dispatches == Some(0)
            || self.max_predicted_duration_ns == Some(0)
            || self.max_regions == Some(0)
        {
            return Err(RuntimeExecutionScheduleError::ZeroBudget);
        }
        if self.max_work_units.is_none()
            && self.max_memory_bytes.is_none()
            && self.max_dispatches.is_none()
            && self.max_predicted_duration_ns.is_none()
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
    pub kernel_families: Vec<String>,
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
            kernel_families: Vec::new(),
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
    pub kernel_families: Vec<String>,
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
        let mut quantum_kernel_families = Vec::<String>::new();
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
                    kernel_families: std::mem::take(&mut quantum_kernel_families),
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
            for family in &region.kernel_families {
                if !quantum_kernel_families.contains(family) {
                    quantum_kernel_families.push(family.clone());
                }
            }
            quantum_commits_state |= region.commits_state_after;
        }
        quanta.push(RuntimeExecutionQuantum {
            region_range: quantum_start..regions.len(),
            component_ids: quantum_components,
            kernel_families: quantum_kernel_families,
            cost: quantum_cost,
            commits_state_after: quantum_commits_state,
        });
        Ok(Self { quanta, total_cost })
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

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct RuntimeExecutionTimingModel {
    observed_cost: RuntimeExecutionCost,
    observed_duration_ns: u64,
}

impl RuntimeExecutionTimingModel {
    fn observe(&mut self, cost: RuntimeExecutionCost, duration_ns: u64) {
        if duration_ns == 0
            || (cost.work_units == 0 && cost.memory_bytes == 0 && cost.dispatches == 0)
        {
            return;
        }
        let cost = RuntimeExecutionCost {
            predicted_duration_ns: 0,
            ..cost
        };
        if self.observed_duration_ns == 0 {
            self.observed_cost = cost;
            self.observed_duration_ns = duration_ns;
            return;
        }
        self.observed_cost = RuntimeExecutionCost::new(
            ewma(self.observed_cost.work_units, cost.work_units),
            ewma(self.observed_cost.memory_bytes, cost.memory_bytes),
            ewma(self.observed_cost.dispatches, cost.dispatches),
        );
        self.observed_duration_ns = ewma(self.observed_duration_ns, duration_ns);
    }

    fn predict(self, cost: RuntimeExecutionCost) -> Option<u64> {
        if self.observed_duration_ns == 0 {
            return None;
        }
        let predictions = [
            scaled_duration(
                self.observed_duration_ns,
                cost.work_units,
                self.observed_cost.work_units,
            ),
            scaled_duration(
                self.observed_duration_ns,
                cost.memory_bytes,
                self.observed_cost.memory_bytes,
            ),
            scaled_duration(
                self.observed_duration_ns,
                cost.dispatches,
                self.observed_cost.dispatches,
            ),
        ];
        predictions
            .into_iter()
            .flatten()
            .max()
            .map(|value| value.max(1))
    }
}

fn ewma(previous: u64, observation: u64) -> u64 {
    previous
        .saturating_mul(3)
        .saturating_add(observation)
        .div_ceil(4)
}

fn scaled_duration(duration_ns: u64, current: u64, observed: u64) -> Option<u64> {
    if current == 0 || observed == 0 {
        return None;
    }
    let value = u128::from(duration_ns)
        .saturating_mul(u128::from(current))
        .div_ceil(u128::from(observed));
    Some(u64::try_from(value).unwrap_or(u64::MAX))
}

pub const RUNTIME_EXECUTION_TARGET_QUANTUM_DURATION_NS: u64 = 250_000_000;

#[derive(Debug)]
pub struct RuntimeExecutionQuantumCalibrator {
    target_duration_ns: u64,
    aggregate: RuntimeExecutionTimingModel,
    family_mixes: BTreeMap<String, RuntimeExecutionTimingModel>,
}

impl Default for RuntimeExecutionQuantumCalibrator {
    fn default() -> Self {
        Self::new(RUNTIME_EXECUTION_TARGET_QUANTUM_DURATION_NS)
            .expect("default execution quantum target is positive")
    }
}

impl RuntimeExecutionQuantumCalibrator {
    pub fn new(target_duration_ns: u64) -> Result<Self, RuntimeExecutionScheduleError> {
        if target_duration_ns == 0 {
            return Err(RuntimeExecutionScheduleError::ZeroBudget);
        }
        Ok(Self {
            target_duration_ns,
            aggregate: RuntimeExecutionTimingModel::default(),
            family_mixes: BTreeMap::new(),
        })
    }

    pub fn prepare_regions(
        &self,
        regions: &mut [RuntimeExecutionRegion],
    ) -> RuntimeExecutionQuantumBudget {
        let mut largest_region = RuntimeExecutionCost::default();
        for region in regions {
            let prediction = self
                .family_mixes
                .get(&region_family_mix(region))
                .and_then(|model| model.predict(region.cost))
                .unwrap_or(self.target_duration_ns);
            region.cost.predicted_duration_ns = prediction.max(1);
            largest_region.work_units = largest_region.work_units.max(region.cost.work_units);
            largest_region.memory_bytes = largest_region.memory_bytes.max(region.cost.memory_bytes);
            largest_region.dispatches = largest_region.dispatches.max(region.cost.dispatches);
        }

        RuntimeExecutionQuantumBudget {
            max_work_units: Some(
                self.aggregate
                    .budget_for_duration(self.target_duration_ns, |cost| cost.work_units)
                    .unwrap_or(largest_region.work_units)
                    .max(largest_region.work_units)
                    .max(1),
            ),
            max_memory_bytes: Some(
                self.aggregate
                    .budget_for_duration(self.target_duration_ns, |cost| cost.memory_bytes)
                    .unwrap_or(largest_region.memory_bytes)
                    .max(largest_region.memory_bytes)
                    .max(1),
            ),
            max_dispatches: Some(
                self.aggregate
                    .budget_for_duration(self.target_duration_ns, |cost| cost.dispatches)
                    .unwrap_or(largest_region.dispatches)
                    .max(largest_region.dispatches)
                    .max(1),
            ),
            max_predicted_duration_ns: Some(self.target_duration_ns),
            max_regions: None,
        }
    }

    pub fn observe_quantum(
        &mut self,
        cost: RuntimeExecutionCost,
        kernel_families: &[String],
        duration_ns: u64,
    ) {
        self.aggregate.observe(cost, duration_ns);
        self.family_mixes
            .entry(family_mix_key(kernel_families))
            .or_default()
            .observe(cost, duration_ns);
    }

    pub fn target_duration_ns(&self) -> u64 {
        self.target_duration_ns
    }
}

impl RuntimeExecutionTimingModel {
    fn budget_for_duration(
        self,
        target_duration_ns: u64,
        dimension: impl Fn(RuntimeExecutionCost) -> u64,
    ) -> Option<u64> {
        let observed = dimension(self.observed_cost);
        if observed == 0 || self.observed_duration_ns == 0 {
            return None;
        }
        let value = u128::from(observed)
            .saturating_mul(u128::from(target_duration_ns))
            .checked_div(u128::from(self.observed_duration_ns))?;
        Some(u64::try_from(value.max(1)).unwrap_or(u64::MAX))
    }
}

fn region_family_mix(region: &RuntimeExecutionRegion) -> String {
    family_mix_key(&region.kernel_families)
}

fn family_mix_key(kernel_families: &[String]) -> String {
    if kernel_families.is_empty() {
        "unlabeled".to_string()
    } else {
        kernel_families.join("+")
    }
}

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
                max_predicted_duration_ns: None,
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
    fn uncalibrated_regions_start_at_one_safe_region_per_quantum() {
        let calibrator = RuntimeExecutionQuantumCalibrator::new(100).unwrap();
        let mut regions = vec![region("a", "a", 10, 1), region("b", "b", 10, 1)];
        let budget = calibrator.prepare_regions(&mut regions);
        let schedule = RuntimeExecutionSchedule::linear(&regions, budget).unwrap();

        assert_eq!(regions[0].cost.predicted_duration_ns, 100);
        assert_eq!(regions[1].cost.predicted_duration_ns, 100);
        assert_eq!(
            schedule
                .quanta
                .iter()
                .map(|quantum| quantum.region_range.clone())
                .collect::<Vec<_>>(),
            vec![0..1, 1..2]
        );
    }

    #[test]
    fn calibrated_family_mix_coalesces_work_within_duration_target() {
        let mut calibrator = RuntimeExecutionQuantumCalibrator::new(100).unwrap();
        let mut observed = region("observed", "a", 10, 1);
        observed.kernel_families = vec!["linear".to_string()];
        calibrator.observe_quantum(observed.cost, &observed.kernel_families, 25);

        let mut regions = (0..5)
            .map(|index| {
                let mut candidate = region(&format!("r{index}"), "a", 10, 1);
                candidate.kernel_families = vec!["linear".to_string()];
                candidate
            })
            .collect::<Vec<_>>();
        let budget = calibrator.prepare_regions(&mut regions);
        let schedule = RuntimeExecutionSchedule::linear(&regions, budget).unwrap();

        assert_eq!(regions[0].cost.predicted_duration_ns, 25);
        assert_eq!(
            schedule
                .quanta
                .iter()
                .map(|quantum| quantum.region_range.clone())
                .collect::<Vec<_>>(),
            vec![0..4, 4..5]
        );
        assert_eq!(schedule.quanta[0].kernel_families, vec!["linear"]);
    }
}
