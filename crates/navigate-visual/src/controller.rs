//! Bounded VPS admission under host-owned CPU and accelerator grants.
//!
//! All timestamps use one host monotonic clock. The host translates flight-controller
//! clocks before admission. MCU attitude and directly integrated IMU estimates use
//! the same demand input. This policy does not integrate sensors or fuse poses.
mod cost;
mod types;
mod worker;
use crate::FrameStamp;
use cost::ProfileCost;
use std::time::Duration;
pub use types::*;
pub use worker::{
    CandidateCompletion, CandidateInput, CandidateWork, CandidateWorker, CandidateWorkerError,
};

/// A controller input or completion violates its execution contract.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ControllerError {
    /// A zero or inconsistent timing limit was supplied.
    #[error("invalid VPS timing or memory profile: {field}")]
    Invalid {
        /// Field that failed validation.
        field: &'static str,
    },
    /// Two profiles use the same identity.
    #[error("duplicate VPS profile {0:?}")]
    DuplicateProfile(ProfileId),
    /// Host clock values moved backwards.
    #[error("VPS clock moved backwards")]
    ClockOrder,
    /// Completion does not identify the active work.
    #[error("VPS completion does not match the active ticket")]
    CompletionMismatch,
}

/// One active VPS job with no pending-frame queue.
///
/// A profile covers one bounded candidate batch or refinement. The caller keeps
/// alternative candidates and may submit more work for the same observation.
/// No admission or completion creates confidence or accepts a pose.
pub struct VisualController {
    config: ControllerConfig,
    profiles: Vec<ProfileCost>,
    active: Option<(WorkTicket, Duration)>,
    last_clock: Option<Duration>,
    previous_start: Option<Duration>,
    next_generation: u64,
}
impl VisualController {
    /// Construct a controller from quality-validated, configured execution profiles.
    ///
    /// # Errors
    /// Rejects inconsistent limits, duplicate identities, and empty profiles.
    pub fn new(
        config: ControllerConfig,
        profiles: Vec<ExecutionProfile>,
    ) -> Result<Self, ControllerError> {
        if config.maximum_capture_age.is_zero()
            || config.maximum_result_age < config.maximum_capture_age
            || config.minimum_interval.is_zero()
            || profiles.is_empty()
        {
            return Err(ControllerError::Invalid {
                field: "controller limits",
            });
        }
        for (index, profile) in profiles.iter().enumerate() {
            if profile.initial_cost.is_zero()
                || profile.initial_call.is_zero()
                || profile.initial_call > profile.initial_cost
                || profile.peak_bytes == 0
            {
                return Err(ControllerError::Invalid {
                    field: "execution profile",
                });
            }
            if profiles[..index].iter().any(|p| p.id == profile.id) {
                return Err(ControllerError::DuplicateProfile(profile.id));
            }
        }
        Ok(Self {
            config,
            profiles: profiles.into_iter().map(ProfileCost::new).collect(),
            active: None,
            last_clock: None,
            previous_start: None,
            next_generation: 0,
        })
    }

    /// Select a useful profile that fits an explicit host grant.
    ///
    /// This method never waits. Discard or replace an unadmitted observation.
    /// Recheck the host grant before each non-preemptible call in an admitted job.
    ///
    /// # Errors
    /// Rejects a reversed host clock or a capture timestamp in the future.
    pub fn consider(
        &mut self,
        now: Duration,
        observation: FrameStamp,
        demand: WorkDemand,
        grants: &[ResourceGrant],
    ) -> Result<Admission, ControllerError> {
        self.clock(now)?;
        let captured = Duration::from_nanos(observation.capture_time_ns);
        if captured > now {
            return Err(ControllerError::Invalid {
                field: "future capture",
            });
        }
        if self.active.is_some() {
            return Ok(Admission::Busy);
        }
        if now.saturating_sub(captured) > self.config.maximum_capture_age {
            return Ok(Admission::StaleObservation);
        }
        if let Some(last) = self.previous_start {
            let next = last.saturating_add(self.config.minimum_interval);
            if now < next {
                return Ok(Admission::NotDue(next));
            }
        }
        let due = demand.due_by.unwrap_or(now);
        let selected = self
            .profiles
            .iter()
            .filter(|p| p.profile.work == demand.work)
            .filter_map(|p| {
                self.fits(p, now, captured, grants)
                    .map(|cost| (p.profile.id, cost))
            })
            .min_by_key(|(_, cost)| *cost);
        let Some((profile, cost)) = selected else {
            return Ok(Admission::NoResources);
        };
        let start_by = due.saturating_sub(cost);
        if now < start_by {
            return Ok(Admission::NotDue(start_by));
        }
        let ticket = WorkTicket {
            generation: self.next_generation,
            profile,
            observation,
        };
        self.next_generation = self.next_generation.wrapping_add(1);
        self.active = Some((ticket, now));
        self.previous_start = Some(now);
        Ok(Admission::Start {
            ticket,
            late: demand
                .due_by
                .map(|deadline| now.saturating_add(cost) > deadline),
        })
    }

    /// Account for work even when matching, geometry, or device execution fails.
    ///
    /// Completion changes cost estimates only. The host owns evidence acceptance.
    /// The returned flag states whether the result is still within the age limit.
    ///
    /// # Errors
    /// Rejects the wrong ticket, a reversed clock, or inconsistent elapsed times.
    pub fn complete(
        &mut self,
        ticket: WorkTicket,
        now: Duration,
        cost: WorkCost,
    ) -> Result<bool, ControllerError> {
        let Some((active, started)) = self.active else {
            return Err(ControllerError::CompletionMismatch);
        };
        if ticket != active {
            return Err(ControllerError::CompletionMismatch);
        }
        if cost.total.is_zero()
            || cost.longest_call > cost.total
            || now.checked_sub(started) != Some(cost.total)
        {
            return Err(ControllerError::Invalid {
                field: "measured work cost",
            });
        }
        self.clock(now)?;
        let profile = self
            .profiles
            .iter_mut()
            .find(|p| p.profile.id == ticket.profile)
            .ok_or(ControllerError::CompletionMismatch)?;
        profile.record(cost);
        self.active = None;
        Ok(
            now.saturating_sub(Duration::from_nanos(ticket.observation.capture_time_ns))
                <= self.config.maximum_result_age,
        )
    }

    fn fits(
        &self,
        profile: &ProfileCost,
        now: Duration,
        captured: Duration,
        grants: &[ResourceGrant],
    ) -> Option<Duration> {
        let cost = profile.estimate();
        let reserved = cost.total.saturating_add(self.config.reserve);
        let completion = now.saturating_add(reserved);
        if completion.saturating_sub(captured) > self.config.maximum_result_age {
            return None;
        }
        grants
            .iter()
            .any(|grant| {
                grant.device == profile.profile.device
                    && grant.until >= completion
                    && grant.maximum_call >= cost.longest_call
                    && grant.memory_bytes >= profile.profile.peak_bytes
            })
            .then_some(reserved)
    }

    fn clock(&mut self, now: Duration) -> Result<(), ControllerError> {
        if self.last_clock.is_some_and(|last| now < last) {
            return Err(ControllerError::ClockOrder);
        }
        self.last_clock = Some(now);
        Ok(())
    }
}
#[cfg(test)]
mod tests;
