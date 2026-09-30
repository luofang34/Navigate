//! Resource admission for one image match and geometric evaluation.
use super::{
    Admission, ControllerConfig, ControllerError, ExecutionProfile, ResourceGrant,
    VisualController, WorkCost, WorkDemand, WorkKind, WorkTicket,
};
use crate::{
    CandidateEvaluation, Frame, ImageMatcher, LocalizerConfig, PosePrior, PoseVerifier,
    ReferenceView, VisualError,
};
use std::time::Duration;

/// Inputs to one candidate evaluation. The reference must already be complete.
pub struct CandidateInput<'a> {
    /// Query observation. Repeated refinements retain this exact frame.
    pub observation: &'a Frame,
    /// Pixels, depth, and map identity for this candidate.
    pub reference: &'a ReferenceView,
    /// Acceptance bounds supplied separately from the rendered candidate.
    pub prior: &'a PosePrior,
}

/// A candidate worker could not initialize or validate its inputs.
#[derive(Debug, thiserror::Error)]
pub enum CandidateWorkerError {
    /// Resource profile or host clock error.
    #[error("candidate execution control: {0}")]
    Controller(#[from] ControllerError),
    /// Invalid geometry policy or observation inputs.
    #[error("candidate inputs: {0}")]
    Visual(#[from] VisualError),
}

/// One executed candidate. Execution eligibility and geometric acceptance differ.
pub struct CandidateCompletion {
    /// Original acquisition stamp and admitted profile identity.
    pub ticket: WorkTicket,
    /// Matcher or geometry rejection, or a geometrically accepted map-relative pose.
    pub evaluation: CandidateEvaluation,
    /// Measured matching and verification duration, including rejected work.
    pub cost: WorkCost,
    /// Completion is within the host's useful result age.
    pub within_age: bool,
    /// Actual duration fits an admission grant. False reports a measured overrun.
    pub within_grant: bool,
}

/// Work was deferred without inference, or one candidate was evaluated.
pub enum CandidateWork {
    /// No matcher was called. The host can submit newer evidence or new grants.
    Deferred(Admission),
    /// One evaluation, not a choice between unresolved geographic alternatives.
    Completed(Box<CandidateCompletion>),
}

/// One configured execution lane for map checks or geographic recovery.
///
/// The worker owns one resident matcher and its cost history. The host admits rendering
/// separately. This worker grants one complete match-and-verify call. The host
/// supplies its monotonic clock. No frames or candidate alternatives are queued.
/// An overrun is observable but cannot be preempted inside a device call.
/// Matchers and their scores do not control geometric acceptance.
pub struct CandidateWorker<M: ImageMatcher> {
    matcher: M,
    controller: VisualController,
    verifier: PoseVerifier,
    profile: ExecutionProfile,
}
impl<M: ImageMatcher> CandidateWorker<M> {
    /// Configure one validated execution path and independent geometry thresholds.
    ///
    /// `initial_call` must equal `initial_cost`: the whole call cannot be paused.
    /// Profile memory must cover all resident models and temporary allocations.
    ///
    /// # Errors
    /// Rejects invalid geometry, timing, memory, and relative-tracking profiles.
    pub fn new(
        matcher: M,
        config: ControllerConfig,
        profile: ExecutionProfile,
        geometry: LocalizerConfig,
    ) -> Result<Self, CandidateWorkerError> {
        if profile.initial_call != profile.initial_cost || profile.work == WorkKind::Tracking {
            return Err(ControllerError::Invalid {
                field: "one-call candidate profile",
            }
            .into());
        }
        Ok(Self {
            matcher,
            controller: VisualController::new(config, vec![profile])?,
            verifier: PoseVerifier::new(geometry)?,
            profile,
        })
    }

    /// Evaluate one candidate if resources and observation age permit it.
    ///
    /// Call from a worker that may block. `clock` and capture time must use one
    /// monotonic domain. A grant permits this whole call; it is not a device-usage
    /// measurement. Timing results include preprocessing inside the adapter.
    /// Every backend or geometry failure consumes time and updates the cost model.
    /// Repeating this call for one frame never creates independent evidence.
    ///
    /// # Errors
    /// Returns input or controller errors. Matcher and geometric errors are in
    /// `CandidateCompletion::evaluation`, with their source context intact.
    pub fn evaluate_blocking(
        &mut self,
        input: CandidateInput<'_>,
        due_by: Option<Duration>,
        grants: &[ResourceGrant],
        mut clock: impl FnMut() -> Duration,
    ) -> Result<CandidateWork, CandidateWorkerError> {
        input.reference.validate(input.observation)?;
        input.prior.validate()?;
        let started = clock();
        let decision = self.controller.consider(
            started,
            input.observation.stamp,
            WorkDemand {
                work: self.profile.work,
                due_by,
            },
            grants,
        )?;
        let Admission::Start { ticket, .. } = decision else {
            return Ok(CandidateWork::Deferred(decision));
        };
        let evaluation = match self
            .matcher
            .match_images_blocking(&input.reference.image, &input.observation.image)
        {
            Ok(pairs) => self.verifier.evaluate(
                input.observation,
                input.reference,
                input.prior,
                &pairs,
                self.matcher.identity(),
            ),
            Err(error) => CandidateEvaluation {
                acceptance: Err(error),
                refinement: None,
            },
        };
        let ended = clock();
        let elapsed = ended
            .checked_sub(started)
            .ok_or(ControllerError::ClockOrder)?;
        let cost = WorkCost {
            total: elapsed,
            longest_call: elapsed,
        };
        let within_age = self.controller.complete(ticket, ended, cost)?;
        let within_grant = grants.iter().any(|grant| {
            grant.device == self.profile.device
                && grant.until >= ended
                && grant.maximum_call >= elapsed
                && grant.memory_bytes >= self.profile.peak_bytes
        });
        Ok(CandidateWork::Completed(Box::new(CandidateCompletion {
            ticket,
            evaluation,
            cost,
            within_age,
            within_grant,
        })))
    }
}

#[cfg(test)]
mod tests;
