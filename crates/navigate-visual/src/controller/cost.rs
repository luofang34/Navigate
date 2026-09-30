//! Bounded cost histories include rejected and failed work.
use super::{ExecutionProfile, WorkCost};
use std::{collections::VecDeque, time::Duration};

pub(super) struct ProfileCost {
    pub profile: ExecutionProfile,
    history: VecDeque<WorkCost>,
}
impl ProfileCost {
    pub fn new(profile: ExecutionProfile) -> Self {
        Self {
            profile,
            history: VecDeque::with_capacity(32),
        }
    }
    pub fn record(&mut self, cost: WorkCost) {
        if self.history.len() == 32 {
            let _oldest = self.history.pop_front();
        }
        self.history.push_back(cost);
    }
    pub fn estimate(&self) -> WorkCost {
        WorkCost {
            total: tail(
                self.history.iter().map(|c| c.total),
                self.profile.initial_cost,
            ),
            longest_call: tail(
                self.history.iter().map(|c| c.longest_call),
                self.profile.initial_call,
            ),
        }
    }
}
fn tail(values: impl Iterator<Item = Duration>, floor: Duration) -> Duration {
    // A short history must not discard a cost spike as a percentile outlier.
    values.max().unwrap_or(floor).max(floor)
}
