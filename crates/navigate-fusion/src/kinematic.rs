//! The navigation filter: admission-gated fusion over a local-level NED
//! state anchored at the first admitted position fix (ADR-0003), with an
//! integrity assessment on every published solution (ADR-0004). A
//! bounded history applies delayed observations at their acquisition
//! time (ADR-0012). The state is on one reference ellipsoid, and the
//! filter refuses observations in another frame (ADR-0013).
//!
//! The filter does not allocate (ADR-0016). The history is inline with
//! at most [`MAX_KINEMATIC_HISTORY`] updates, and the innovation
//! statistics have one inline slot for each measurement kind.

mod history;
pub(crate) mod reanchor;
pub(crate) mod stats;
mod update;

#[cfg(test)]
mod tests;

use nalgebra::{Matrix3, Vector3};
use navigate_contract::{
    ClockDomainId, GeodeticPosition, MonotonicNanos, NavigationSolution, NedVelocity,
    SolutionStamp, SourceEpoch, WrappingSequence,
};
use navigate_geodesy::{BodyEllipsoid, LocalTangentPlane, NedOffset};

use crate::admission::{
    check_clock, check_composition, check_frame, check_freshness, check_value, seconds,
};
use crate::config::FusionConfig;
use crate::integrity;
use crate::kalman::{self, KalmanState};
use crate::observation::{MeasurementKind, Observation, ObservationValue};
use crate::provenance::ProvenanceLedger;
use crate::rejection::{IngestOutcome, RejectionCounters, RejectionReason};
use crate::solution_sequence::SolutionSequencer;
use history::History;
pub use history::MAX_KINEMATIC_HISTORY;
pub use stats::InnovationStats;
use stats::StatsTable;
use update::Mode;

/// The initialized core: the tangent plane anchored at the first admitted
/// fix, the Kalman state, and the time the state refers to.
#[derive(Debug, Clone, Copy)]
struct Core {
    pub(super) plane: LocalTangentPlane,
    pub(super) state: KalmanState<6>,
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
    ellipsoid: BodyEllipsoid,
    epoch: SourceEpoch,
    sequencer: SolutionSequencer,
    rejections: RejectionCounters,
    ledger: ProvenanceLedger,
    core: Option<Core>,
    history: History,
    stats: StatsTable,
}

impl NavigationFilter {
    /// Builds an uninitialized filter on [`BodyEllipsoid::WGS84`].
    /// `clock` is the domain every observation stamp and every `now` must
    /// come from. The filter keeps `config.require_explicit_frame`.
    #[must_use]
    pub fn new(config: FusionConfig, clock: ClockDomainId) -> Self {
        Self::build(config, clock, BodyEllipsoid::WGS84)
    }

    /// Builds an uninitialized filter on `ellipsoid` (ADR-0013). The
    /// filter sets `config.require_explicit_frame` to `true`: every
    /// observation must declare the frame of `ellipsoid`.
    #[must_use]
    pub fn new_for_ellipsoid(
        config: FusionConfig,
        clock: ClockDomainId,
        ellipsoid: BodyEllipsoid,
    ) -> Self {
        let config = FusionConfig {
            require_explicit_frame: true,
            ..config
        };
        Self::build(config, clock, ellipsoid)
    }

    fn build(config: FusionConfig, clock: ClockDomainId, ellipsoid: BodyEllipsoid) -> Self {
        Self {
            config,
            clock,
            ellipsoid,
            epoch: SourceEpoch::new(0),
            sequencer: SolutionSequencer::new(),
            rejections: RejectionCounters::default(),
            ledger: ProvenanceLedger::default(),
            core: None,
            history: History::default(),
            stats: StatsTable::default(),
        }
    }

    /// Offers one observation. Gates run in order — clock domain, frame,
    /// composition, value plausibility, freshness, per-source ordering,
    /// then the model checks and the innovation gate — and a rejected
    /// observation changes nothing, including per-source tracking and the
    /// history. `now` is the admission reference on the filter's clock
    /// domain. An observation acquired before the state time goes through
    /// the delayed-measurement history (ADR-0012).
    pub fn ingest(&mut self, obs: &Observation, now: MonotonicNanos) -> IngestOutcome {
        match self.admit(obs, now) {
            Ok(()) => IngestOutcome::Accepted,
            Err(reason) => {
                self.rejections.record(&reason);
                if matches!(reason, RejectionReason::InnovationGate { .. }) {
                    self.stats.record_gate_rejection(obs.value.kind());
                    self.ledger
                        .record_gate_rejection(obs.value.constrained_state());
                }
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
    /// The solution sequence advances only when the published solution
    /// changes: a second tick for the same state at the same `now` gives
    /// the same solution with the same sequence.
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
    /// assessment window, all per-source tracking, the delayed-measurement
    /// history, and the innovation statistics, and rewinds the solution
    /// sequence. Rejection counters persist — they describe the process
    /// lifetime, not one epoch.
    pub fn reset(&mut self) {
        self.epoch = self.epoch.next();
        self.sequencer = SolutionSequencer::new();
        self.ledger.clear();
        self.core = None;
        self.history.clear();
        self.stats.clear();
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

    /// The reference ellipsoid of the state and of every published
    /// solution.
    #[must_use]
    pub const fn ellipsoid(&self) -> BodyEllipsoid {
        self.ellipsoid
    }

    /// The active configuration.
    #[must_use]
    pub const fn config(&self) -> &FusionConfig {
        &self.config
    }

    /// Innovation statistics of one measurement kind in this filter
    /// epoch. `None` when no update and no gate refusal of that kind
    /// occurred.
    #[must_use]
    pub fn innovation_stats(&self, kind: MeasurementKind) -> Option<InnovationStats> {
        self.stats.get(kind)
    }

    /// Whether a first position fix has anchored the state.
    #[must_use]
    pub const fn is_initialized(&self) -> bool {
        self.core.is_some()
    }

    fn admit(&mut self, obs: &Observation, now: MonotonicNanos) -> Result<(), RejectionReason> {
        check_clock(self.clock, &obs.stamp)?;
        check_frame(
            self.ellipsoid.frame(),
            obs.frame,
            self.config.require_explicit_frame,
        )?;
        check_composition(obs.composition)?;
        if !obs.value.is_supported() {
            return Err(RejectionReason::UnsupportedMeasurement {
                kind: obs.value.kind(),
            });
        }
        if obs.correlated.is_some() {
            // This filter has no correlated-error states.
            return Err(RejectionReason::CorrelationNotModelled {
                kind: obs.value.kind(),
            });
        }
        check_value(&obs.value)?;
        check_freshness(&obs.stamp, now, self.config.staleness_bound)?;
        self.ledger.check_source_ordering(&obs.stamp)?;
        if let Some(nis) = self.apply(obs)? {
            self.stats.record_admitted(obs.value.kind(), nis);
        }
        self.ledger.commit(
            &obs.stamp,
            obs.composition,
            obs.value.constrained_state(),
            now,
            self.config.assessment_window,
        );
        Ok(())
    }

    /// Applies an admissible observation. The result is the chi-square
    /// statistic of the update, or `None` for the fix that initializes the
    /// filter.
    fn apply(&mut self, obs: &Observation) -> Result<Option<f64>, RejectionReason> {
        let acquired_at = obs.stamp.acquired_at;
        let Some(core) = self.core else {
            return match &obs.value {
                ObservationValue::PositionFix {
                    position,
                    covariance,
                } => {
                    self.initialize(position, &kalman::cov3_to_matrix(covariance), acquired_at)?;
                    Ok(None)
                }
                _ => Err(RejectionReason::NotInitialized),
            };
        };
        if acquired_at < core.state_at {
            let (next, chi2) =
                self.history
                    .apply_delayed(&core, &obs.value, acquired_at, &self.config)?;
            self.core = Some(next);
            return Ok(Some(chi2));
        }
        let mut candidate = core;
        let chi2 = update::apply_value(
            &mut candidate,
            &obs.value,
            acquired_at,
            &self.config,
            Mode::Gated,
        )?;
        self.history
            .record(core, obs.value, acquired_at, self.config.history_capacity);
        self.core = Some(candidate);
        Ok(Some(chi2))
    }

    fn initialize(
        &mut self,
        position: &GeodeticPosition,
        r: &Matrix3<f64>,
        acquired_at: MonotonicNanos,
    ) -> Result<(), RejectionReason> {
        // The plausibility gate ran first, so plane construction cannot
        // fail; mapping the error keeps this path total without a panic.
        let plane = LocalTangentPlane::new(*position, self.ellipsoid)
            .map_err(|_| RejectionReason::NonFiniteValue)?;
        // The fix anchors the origin, so its covariance IS the position
        // uncertainty; a measurement update against a diffuse prior the
        // fix itself centers would understate it. Velocity keeps the
        // diffuse prior until velocity measurements arrive.
        let state =
            KalmanState::from_position_fix(r, self.config.initial_velocity_variance_m2_per_s2);
        let core = Core {
            plane,
            state,
            state_at: acquired_at,
        };
        self.history.initialize(core);
        self.core = Some(core);
        Ok(())
    }

    fn prune_window(&mut self, now: MonotonicNanos) {
        self.ledger.prune_window(now, self.config.assessment_window);
    }

    /// Publishes `state` — already propagated to `now` by the caller —
    /// without touching the retained state.
    fn publish(
        &mut self,
        state: &KalmanState<6>,
        now: MonotonicNanos,
    ) -> Option<NavigationSolution> {
        let core = self.core.as_ref()?;
        let assessment = integrity::derive(
            &self.ledger,
            now,
            kalman::horizontal_1sigma_m(&state.p),
            kalman::vertical_1sigma_m(&state.p),
            &self.config,
        );
        let x = &state.x;
        let position = core.plane.from_ned(&NedOffset::new(x[0], x[1], x[2]));
        let (v_local, p_cov_local, v_cov_local) = reanchor::plane_to_vehicle(
            &core.plane,
            position,
            Vector3::new(x[3], x[4], x[5]),
            state.p.fixed_view::<3, 3>(0, 0).into_owned(),
            state.p.fixed_view::<3, 3>(3, 3).into_owned(),
        );
        let velocity = NedVelocity::new(v_local[0], v_local[1], v_local[2]);
        let position_cov = kalman::matrix_to_cov3(&p_cov_local);
        let velocity_cov = kalman::matrix_to_cov3(&v_cov_local);
        let stamp = SolutionStamp::new(self.epoch, WrappingSequence::new(0), now, self.clock);
        Some(self.sequencer.surface(NavigationSolution::new(
            stamp,
            position,
            velocity,
            position_cov,
            velocity_cov,
            assessment,
            // Provenance is epoch-cumulative: everything the state ever
            // absorbed, while `integrity.contributing` stays
            // window-scoped.
            self.ledger.epoch_composition(),
            self.ellipsoid.contract(),
        )))
    }
}
