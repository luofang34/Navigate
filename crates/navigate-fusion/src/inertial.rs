//! Strapdown error-state inertial navigation with seventeen states
//! (ADR-0014).
//!
//! [`InertialNavigationFilter`] propagates a nominal strapdown state with
//! IMU samples and corrects it with admission-gated observations. The
//! error state is `[δp, δv, δθ, δb_a, δb_g, δb_alt, δb_mag]`. The
//! attitude error is a right perturbation in the body axes. Each update
//! injects the error estimate into the nominal state and resets the
//! covariance (closed loop).
//!
//! The filter has two regimes. In the body-surface regime the state is in
//! a tangent plane fixed to a rotating [`crate::PlanetaryBody`]. In the
//! Cartesian space regime the state is in the frame of a
//! [`crate::SpaceEnvironment`], and [`InertialNavigationFilter::recenter_space_frame`]
//! moves it to the frame of another body. A bounded history applies a
//! delayed IMU sample or observation at its acquisition time.

mod config;
mod discretization;
pub(crate) mod estimate;
mod history;
mod imu;
mod initialize;
pub(crate) mod mechanization;
mod models;
pub(crate) mod nominal;
mod publish;
pub(crate) mod regime;
mod space_frame;
mod update;

#[cfg(test)]
mod tests;

use std::sync::Arc;

use navigate_contract::{ClockDomainId, FrameId, MonotonicNanos, SourceEpoch, WrappingSequence};

use self::estimate::InertialCore;
use crate::InnovationStats;
use crate::admission::{check_clock, check_composition, check_freshness, check_value};
use crate::environment::{PlanetaryBody, SpaceEnvironment};
use crate::navigation_filter::stats::StatsTable;
use crate::observation::{MeasurementKind, Observation};
use crate::provenance::ProvenanceLedger;
use crate::rejection::{IngestOutcome, RejectionCounters, RejectionReason};
pub use config::InertialConfig;
use history::History;
pub use imu::ImuSample;
use regime::{Setup, supports};
use update::Event;

/// Sans-IO strapdown inertial filter with seventeen error states
/// (ADR-0014).
///
/// All time is caller-supplied [`MonotonicNanos`] on the clock domain of
/// the filter. Identical event scripts give identical solutions.
#[derive(Debug, Clone)]
pub struct InertialNavigationFilter {
    config: InertialConfig,
    clock: ClockDomainId,
    setup: Setup,
    epoch: SourceEpoch,
    next_solution_sequence: WrappingSequence,
    rejections: RejectionCounters,
    ledger: ProvenanceLedger,
    core: Option<InertialCore>,
    history: History,
    stats: StatsTable,
}

impl InertialNavigationFilter {
    /// Builds an uninitialized filter on the surface of `body`. A body
    /// that is not on [`navigate_geodesy::BodyEllipsoid::WGS84`] sets
    /// `config.fusion.require_explicit_frame` to `true`, as
    /// [`crate::NavigationFilter::new_for_ellipsoid`] does.
    #[must_use]
    pub fn new_surface(config: InertialConfig, clock: ClockDomainId, body: PlanetaryBody) -> Self {
        let mut config = config;
        if body.ellipsoid != navigate_geodesy::BodyEllipsoid::WGS84 {
            config.fusion.require_explicit_frame = true;
        }
        Self::build(config, clock, Setup::BodySurface(body))
    }

    /// Builds an uninitialized filter in the frame of `env`. The filter
    /// sets `config.fusion.require_explicit_frame` to `true`.
    #[must_use]
    pub fn new_space(config: InertialConfig, clock: ClockDomainId, env: SpaceEnvironment) -> Self {
        let mut config = config;
        config.fusion.require_explicit_frame = true;
        Self::build(config, clock, Setup::CartesianSpace(Arc::new(env)))
    }

    fn build(config: InertialConfig, clock: ClockDomainId, setup: Setup) -> Self {
        Self {
            config,
            clock,
            setup,
            epoch: SourceEpoch::new(0),
            next_solution_sequence: WrappingSequence::new(0),
            rejections: RejectionCounters::default(),
            ledger: ProvenanceLedger::default(),
            core: None,
            history: History::default(),
            stats: StatsTable::default(),
        }
    }

    /// Offers one IMU sample. The gates run in order: clock domain,
    /// composition, plausibility, freshness, per-source ordering, then the
    /// frame at the end time of the sample. A sample enters the
    /// provenance. It does not reset the observation silence, because a
    /// propagation sensor cannot cross-check the state (ADR-0004).
    pub fn ingest_imu(&mut self, sample: &ImuSample, now: MonotonicNanos) -> IngestOutcome {
        let result = self.admit_imu(sample, now);
        self.outcome(result, None)
    }

    /// Offers one observation. The gates run in order: clock domain,
    /// composition, regime support, value plausibility, freshness,
    /// per-source ordering, the frame at the acquisition time, then the
    /// model checks and the innovation gate. A refused observation
    /// changes nothing. An observation acquired before the state time
    /// goes through the history.
    pub fn ingest(&mut self, obs: &Observation, now: MonotonicNanos) -> IngestOutcome {
        let result = self.admit(obs, now);
        self.outcome(result, Some(obs.value.kind()))
    }

    fn outcome(
        &mut self,
        result: Result<(), RejectionReason>,
        kind: Option<MeasurementKind>,
    ) -> IngestOutcome {
        match result {
            Ok(()) => IngestOutcome::Accepted,
            Err(reason) => {
                self.rejections.record(&reason);
                if let (RejectionReason::InnovationGate { .. }, Some(kind)) = (&reason, kind) {
                    self.stats.record_gate_rejection(kind);
                    self.ledger.record_gate_rejection();
                }
                IngestOutcome::Rejected(reason)
            }
        }
    }

    fn admit_imu(
        &mut self,
        sample: &ImuSample,
        now: MonotonicNanos,
    ) -> Result<(), RejectionReason> {
        check_clock(self.clock, &sample.stamp)?;
        check_composition(sample.composition)?;
        if !sample.is_plausible() {
            return Err(RejectionReason::NonFiniteValue);
        }
        check_freshness(&sample.stamp, now, self.config.fusion.staleness_bound)?;
        self.ledger.check_source_ordering(&sample.stamp)?;
        self.apply_event(Event::Imu(*sample), sample.stamp.acquired_at)?;
        self.ledger.track(&sample.stamp, sample.composition);
        Ok(())
    }

    fn admit(&mut self, obs: &Observation, now: MonotonicNanos) -> Result<(), RejectionReason> {
        check_clock(self.clock, &obs.stamp)?;
        check_composition(obs.composition)?;
        let kind = obs.value.kind();
        if !supports(&self.setup, kind) {
            return Err(RejectionReason::UnsupportedMeasurement { kind });
        }
        check_value(&obs.value)?;
        check_freshness(&obs.stamp, now, self.config.fusion.staleness_bound)?;
        self.ledger.check_source_ordering(&obs.stamp)?;
        if let Some(nis) = self.apply_event(Event::Observation(*obs), obs.stamp.acquired_at)? {
            self.stats.record_admitted(kind, nis);
        }
        let window = self.config.fusion.assessment_window;
        self.ledger.commit(&obs.stamp, obs.composition, now, window);
        Ok(())
    }

    /// Counts a refusal of a state operation and passes the result on.
    fn record_failure(
        &mut self,
        result: Result<(), RejectionReason>,
    ) -> Result<(), RejectionReason> {
        if let Err(reason) = &result {
            self.rejections.record(reason);
        }
        result
    }

    /// Applies an admissible event in order or through the history.
    fn apply_event(
        &mut self,
        event: Event,
        at: MonotonicNanos,
    ) -> Result<Option<f64>, RejectionReason> {
        let Some(core) = self.core.as_ref() else {
            return Err(RejectionReason::NotInitialized);
        };
        if at < core.state_at {
            let (next, chi2) = self.history.apply_delayed(core, &event, at, &self.config)?;
            self.core = Some(next);
            return Ok(chi2);
        }
        let mut candidate = core.clone();
        let chi2 = update::apply_event(
            &mut candidate,
            &event,
            at,
            &self.config,
            estimate::Mode::Gated,
            None,
        )?;
        let capacity = self.config.fusion.history_capacity;
        self.history.record(core.clone(), event, at, capacity);
        self.core = Some(candidate);
        Ok(chi2)
    }

    /// Starts a new filter epoch: forgets the state, the history, the
    /// per-source tracking, the window, and the innovation statistics.
    /// Rejection counters persist.
    pub fn reset(&mut self) {
        self.epoch = self.epoch.next();
        self.next_solution_sequence = WrappingSequence::new(0);
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

    /// The restart epoch of the filter, advanced by [`Self::reset`].
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
    pub const fn config(&self) -> &InertialConfig {
        &self.config
    }

    /// The active frame: the frame of the state, the observations, and
    /// the IMU samples.
    #[must_use]
    pub fn frame(&self) -> FrameId {
        self.core
            .as_ref()
            .map_or_else(|| self.setup.frame(), |core| core.regime.frame())
    }

    /// Whether an initialization has set the state.
    #[must_use]
    pub const fn is_initialized(&self) -> bool {
        self.core.is_some()
    }

    /// Innovation statistics of one measurement kind in this filter
    /// epoch.
    #[must_use]
    pub fn innovation_stats(&self, kind: MeasurementKind) -> Option<InnovationStats> {
        self.stats.get(kind)
    }
}
