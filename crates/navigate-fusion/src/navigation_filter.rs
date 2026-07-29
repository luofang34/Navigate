//! The navigation filter: admission-gated fusion over a local-level NED
//! state anchored at the first admitted position fix (ADR-0003), with an
//! integrity assessment on every published solution (ADR-0004).

mod integrity;
#[cfg(test)]
mod tests;

use std::collections::HashMap;

use nalgebra::{Matrix3, Vector3};
use navigate_contract::{
    ClockDomainId, DurationNanos, GeodeticPosition, MonotonicNanos, NavigationSolution,
    NedVelocity, ObservationStamp, SolutionStamp, SourceComposition, SourceEpoch, SourceId,
    SymmetricCov3, WrappingSequence,
};
use navigate_geodesy::{LocalTangentPlane, NedOffset};

use crate::config::FusionConfig;
use crate::filter::{self, KalmanState, MeasurementBlock};
use crate::observation::{Observation, ObservationValue};
use crate::rejection::{IngestOutcome, RejectionCounters, RejectionReason};

/// Per-source admission tracking, valid within one source epoch.
#[derive(Debug, Clone, Copy)]
struct SourceTrack {
    epoch: SourceEpoch,
    sequence: WrappingSequence,
    acquired_at: MonotonicNanos,
}

/// One admitted observation's contribution to the assessment window.
#[derive(Debug, Clone, Copy)]
pub(crate) struct AdmittedContribution {
    pub(crate) source: SourceId,
    pub(crate) composition: SourceComposition,
    admitted_at: MonotonicNanos,
}

/// The initialized core: the tangent plane anchored at the first admitted
/// fix, the Kalman state, and the time the state refers to.
#[derive(Debug, Clone)]
struct Core {
    plane: LocalTangentPlane,
    state: KalmanState,
    state_at: MonotonicNanos,
}

/// Sans-IO navigation filter fusing admission-gated observations
/// (ADR-0003) into integrity-honest solutions (ADR-0004).
///
/// All time is caller-supplied [`MonotonicNanos`] on the single clock
/// domain fixed at construction; identical observation scripts yield
/// field-identical solutions.
#[derive(Debug, Clone)]
pub struct NavigationFilter {
    config: FusionConfig,
    clock: ClockDomainId,
    epoch: SourceEpoch,
    next_solution_sequence: WrappingSequence,
    rejections: RejectionCounters,
    sources: HashMap<SourceId, SourceTrack>,
    admitted: Vec<AdmittedContribution>,
    // Union of every admitted observation's composition this filter
    // epoch: published provenance describes everything the state ever
    // absorbed, not just the assessment window.
    epoch_composition: SourceComposition,
    last_admitted_at: Option<MonotonicNanos>,
    core: Option<Core>,
}

impl NavigationFilter {
    /// Builds an uninitialized filter. `clock` is the domain every
    /// observation stamp and every `now` must come from.
    #[must_use]
    pub fn new(config: FusionConfig, clock: ClockDomainId) -> Self {
        Self {
            config,
            clock,
            epoch: SourceEpoch::new(0),
            next_solution_sequence: WrappingSequence::new(0),
            rejections: RejectionCounters::default(),
            sources: HashMap::new(),
            admitted: Vec::new(),
            epoch_composition: SourceComposition::empty(),
            last_admitted_at: None,
            core: None,
        }
    }

    /// Offers one observation. Gates run in order — clock domain,
    /// composition, value plausibility, freshness, per-source ordering,
    /// then the innovation gate — and a rejected observation changes
    /// nothing, including per-source tracking. `now` is the admission
    /// reference on the filter's clock domain.
    pub fn ingest(&mut self, obs: &Observation, now: MonotonicNanos) -> IngestOutcome {
        match self.admit(obs, now) {
            Ok(()) => IngestOutcome::Accepted,
            Err(reason) => {
                self.rejections.record(&reason);
                IngestOutcome::Rejected(reason)
            }
        }
    }

    /// Publishes the solution for `now`: `None` before initialization,
    /// otherwise a copy of the state propagated to `now` with its
    /// integrity assessment. Publication never commits propagation — the
    /// retained state stays at its last measurement time, so in-order
    /// measurements apply with their true positive `dt` even inside a
    /// tick loop. A `now` earlier than the state's time is clock misuse
    /// and yields `None`, never an absorbed or backdated publication.
    /// Each publication advances the solution sequence.
    pub fn tick(&mut self, now: MonotonicNanos) -> Option<NavigationSolution> {
        let psd = self.config.process_noise_accel_psd;
        let core = self.core.as_ref()?;
        let elapsed = now.elapsed_since(core.state_at)?;
        let published = if elapsed.as_nanos() > 0 {
            core.state.propagated(seconds(elapsed), psd)
        } else {
            core.state
        };
        self.prune_window(now);
        self.publish(&published, now)
    }

    /// Starts a new filter epoch: forgets the origin, the state, the
    /// assessment window, and all per-source tracking, and rewinds the
    /// solution sequence. Rejection counters persist — they describe the
    /// process lifetime, not one epoch.
    pub fn reset(&mut self) {
        self.epoch = self.epoch.next();
        self.next_solution_sequence = WrappingSequence::new(0);
        self.sources.clear();
        self.admitted.clear();
        self.epoch_composition = SourceComposition::empty();
        self.last_admitted_at = None;
        self.core = None;
    }

    /// Per-reason rejection counters.
    #[must_use]
    pub const fn rejections(&self) -> &RejectionCounters {
        &self.rejections
    }

    /// The filter's restart epoch, advanced by [`Self::reset`].
    #[must_use]
    pub const fn epoch(&self) -> SourceEpoch {
        self.epoch
    }

    /// The clock domain every stamp is checked against.
    #[must_use]
    pub const fn clock(&self) -> ClockDomainId {
        self.clock
    }

    /// The active configuration.
    #[must_use]
    pub const fn config(&self) -> &FusionConfig {
        &self.config
    }

    /// Whether a first position fix has anchored the state.
    #[must_use]
    pub const fn is_initialized(&self) -> bool {
        self.core.is_some()
    }

    fn admit(&mut self, obs: &Observation, now: MonotonicNanos) -> Result<(), RejectionReason> {
        self.check_clock(&obs.stamp)?;
        check_composition(obs.composition)?;
        check_value(&obs.value)?;
        self.check_freshness(&obs.stamp, now)?;
        self.check_source_ordering(&obs.stamp)?;
        self.apply(obs)?;
        self.commit(obs, now);
        Ok(())
    }

    fn check_clock(&self, stamp: &ObservationStamp) -> Result<(), RejectionReason> {
        if stamp.clock != self.clock {
            return Err(RejectionReason::ClockDomainMismatch {
                expected: self.clock,
                got: stamp.clock,
            });
        }
        Ok(())
    }

    fn check_freshness(
        &self,
        stamp: &ObservationStamp,
        now: MonotonicNanos,
    ) -> Result<(), RejectionReason> {
        // An acquisition time ahead of the admission reference is the
        // same time-discipline violation as a per-source regression.
        let Some(age) = now.elapsed_since(stamp.acquired_at) else {
            return Err(RejectionReason::AcquisitionTimeRegression {
                last: now,
                got: stamp.acquired_at,
            });
        };
        if age > self.config.staleness_bound {
            return Err(RejectionReason::Stale {
                age,
                bound: self.config.staleness_bound,
            });
        }
        Ok(())
    }

    fn check_source_ordering(&self, stamp: &ObservationStamp) -> Result<(), RejectionReason> {
        let Some(track) = self.sources.get(&stamp.source) else {
            // A source's first observation is always sequence-admissible.
            return Ok(());
        };
        if track.epoch != stamp.epoch {
            // Only a strictly advancing epoch resets the source's
            // sequence and time tracking; anything else replays a prior
            // incarnation whose stamps were already judged.
            if track.epoch.advances(stamp.epoch) {
                return Ok(());
            }
            return Err(RejectionReason::EpochRegression {
                last: track.epoch,
                got: stamp.epoch,
            });
        }
        if !track.sequence.admits(stamp.sequence) {
            return Err(RejectionReason::SequenceNotAdmitted {
                last: track.sequence,
                got: stamp.sequence,
            });
        }
        if stamp.acquired_at < track.acquired_at {
            return Err(RejectionReason::AcquisitionTimeRegression {
                last: track.acquired_at,
                got: stamp.acquired_at,
            });
        }
        Ok(())
    }

    fn apply(&mut self, obs: &Observation) -> Result<(), RejectionReason> {
        match &obs.value {
            ObservationValue::PositionFix {
                position,
                covariance,
            } => self.apply_position(position, covariance, obs.stamp.acquired_at),
            ObservationValue::VelocityFix {
                velocity,
                covariance,
            } => self.apply_velocity(velocity, covariance, obs.stamp.acquired_at),
        }
    }

    fn apply_position(
        &mut self,
        position: &GeodeticPosition,
        covariance: &SymmetricCov3,
        acquired_at: MonotonicNanos,
    ) -> Result<(), RejectionReason> {
        let r = filter::cov3_to_matrix(covariance);
        match self.core.as_mut() {
            None => self.initialize(position, &r, acquired_at),
            Some(core) => {
                let offset = core.plane.to_ned(position);
                let z = Vector3::new(offset.north_m, offset.east_m, offset.down_m);
                update_core(
                    core,
                    &self.config,
                    MeasurementBlock::Position,
                    &z,
                    &r,
                    acquired_at,
                )
            }
        }
    }

    fn apply_velocity(
        &mut self,
        velocity: &NedVelocity,
        covariance: &SymmetricCov3,
        acquired_at: MonotonicNanos,
    ) -> Result<(), RejectionReason> {
        let r = filter::cov3_to_matrix(covariance);
        let Some(core) = self.core.as_mut() else {
            return Err(RejectionReason::NotInitialized);
        };
        let z = Vector3::new(velocity.north_mps, velocity.east_mps, velocity.down_mps);
        update_core(
            core,
            &self.config,
            MeasurementBlock::Velocity,
            &z,
            &r,
            acquired_at,
        )
    }

    fn initialize(
        &mut self,
        position: &GeodeticPosition,
        r: &Matrix3<f64>,
        acquired_at: MonotonicNanos,
    ) -> Result<(), RejectionReason> {
        // The plausibility gate ran first, so plane construction cannot
        // fail; mapping the error keeps this path total without a panic.
        let plane =
            LocalTangentPlane::new(*position).map_err(|_| RejectionReason::NonFiniteValue)?;
        // The fix anchors the origin, so its covariance IS the position
        // uncertainty; a measurement update against a diffuse prior the
        // fix itself centers would understate it. Velocity keeps the
        // diffuse prior until velocity measurements arrive.
        let state =
            KalmanState::from_position_fix(r, self.config.initial_velocity_variance_m2_per_s2);
        self.core = Some(Core {
            plane,
            state,
            state_at: acquired_at,
        });
        Ok(())
    }

    fn commit(&mut self, obs: &Observation, now: MonotonicNanos) {
        self.sources.insert(
            obs.stamp.source,
            SourceTrack {
                epoch: obs.stamp.epoch,
                sequence: obs.stamp.sequence,
                acquired_at: obs.stamp.acquired_at,
            },
        );
        self.admitted.push(AdmittedContribution {
            source: obs.stamp.source,
            composition: obs.composition,
            admitted_at: now,
        });
        self.epoch_composition = self.epoch_composition.union(obs.composition);
        self.prune_window(now);
        self.last_admitted_at = Some(now);
    }

    fn prune_window(&mut self, now: MonotonicNanos) {
        let window = self.config.assessment_window;
        self.admitted.retain(|c| {
            now.elapsed_since(c.admitted_at)
                .is_none_or(|age| age <= window)
        });
    }

    /// Publishes `state` — already propagated to `now` by the caller —
    /// without touching the retained state.
    fn publish(&mut self, state: &KalmanState, now: MonotonicNanos) -> Option<NavigationSolution> {
        let core = self.core.as_ref()?;
        let silence = self.last_admitted_at.and_then(|at| now.elapsed_since(at));
        let assessment = integrity::derive(
            &self.admitted,
            filter::horizontal_1sigma_m(&state.p),
            filter::vertical_1sigma_m(&state.p),
            silence,
            &self.config,
        );
        let x = &state.x;
        let position = core.plane.from_ned(&NedOffset::new(x[0], x[1], x[2]));
        let velocity = NedVelocity::new(x[3], x[4], x[5]);
        let position_cov = filter::matrix_to_cov3(&state.p.fixed_view::<3, 3>(0, 0).into_owned());
        let velocity_cov = filter::matrix_to_cov3(&state.p.fixed_view::<3, 3>(3, 3).into_owned());
        let sequence = self.next_solution_sequence;
        self.next_solution_sequence = sequence.next();
        let stamp = SolutionStamp::new(self.epoch, sequence, now, self.clock);
        Some(NavigationSolution::new(
            stamp,
            position,
            velocity,
            position_cov,
            velocity_cov,
            assessment,
            // Provenance is epoch-cumulative: everything the state ever
            // absorbed, while `integrity.contributing` stays
            // window-scoped.
            self.epoch_composition,
        ))
    }
}

/// Propagates a candidate to the measurement time, gates it, and commits
/// only on acceptance so a rejected observation changes nothing
/// (ADR-0003). A measurement acquired at or before the state's time is
/// applied at the state's present time; bounded-history re-propagation is
/// the documented extension for delayed measurements.
fn update_core(
    core: &mut Core,
    config: &FusionConfig,
    block: MeasurementBlock,
    z: &Vector3<f64>,
    r: &Matrix3<f64>,
    acquired_at: MonotonicNanos,
) -> Result<(), RejectionReason> {
    let dt_s = acquired_at
        .elapsed_since(core.state_at)
        .map_or(0.0, seconds);
    let candidate = core.state.propagated(dt_s, config.process_noise_accel_psd);
    let prepared = candidate
        .prepare_update(block, z, r)
        .ok_or(RejectionReason::ImplausibleCovariance)?;
    if prepared.chi2 > config.innovation_gate_chi2 {
        return Err(RejectionReason::InnovationGate {
            chi2: prepared.chi2,
            threshold: config.innovation_gate_chi2,
        });
    }
    core.state = candidate.apply_update(block, r, &prepared);
    if acquired_at > core.state_at {
        core.state_at = acquired_at;
    }
    Ok(())
}

fn check_composition(composition: SourceComposition) -> Result<(), RejectionReason> {
    if composition.is_empty() {
        return Err(RejectionReason::EmptyComposition);
    }
    if composition.is_estimator_derived() {
        return Err(RejectionReason::EstimatorDerived);
    }
    Ok(())
}

fn check_value(value: &ObservationValue) -> Result<(), RejectionReason> {
    let covariance = match value {
        ObservationValue::PositionFix {
            position,
            covariance,
        } => {
            if !position.is_plausible() {
                return Err(RejectionReason::NonFiniteValue);
            }
            covariance
        }
        ObservationValue::VelocityFix {
            velocity,
            covariance,
        } => {
            if !velocity.is_finite() {
                return Err(RejectionReason::NonFiniteValue);
            }
            covariance
        }
    };
    if !covariance.is_plausible()
        || !filter::is_positive_definite(&filter::cov3_to_matrix(covariance))
    {
        return Err(RejectionReason::ImplausibleCovariance);
    }
    Ok(())
}

fn seconds(duration: DurationNanos) -> f64 {
    duration.as_nanos() as f64 * 1e-9
}
