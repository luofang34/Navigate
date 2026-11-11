//! Error-state inertial navigation with twenty-two states (ADR-0014,
//! ADR-0017, ADR-0018).
//!
//! [`InertialNavigationFilter`] propagates a nominal state and corrects
//! it with admission-gated observations. The error state is
//! `[δp, δv, δθ, δb_a, δb_g, δb_alt, δb_mag, δa_dmc, c·δt, c·δṫ]`. The
//! attitude error is a right perturbation in the body axes. The
//! empirical acceleration `a_dmc` of the dynamic model compensation and
//! the receiver clock states `c·δt` and `c·δṫ` are optional. A state
//! without a configured model keeps zero variance and does not couple
//! into the other states. Each update injects the error estimate into
//! the nominal state and resets the covariance (closed loop).
//!
//! In [`PropagationMode::SensedImu`] IMU samples or IMU increments drive
//! the state. In [`PropagationMode::ForceModel`] the orbital force model
//! of the environment drives the state with fourth-order Runge–Kutta
//! steps, for a coast without IMU data.
//! [`InertialNavigationFilter::apply_impulsive_burn`] adds a velocity
//! change that no sensor measures.
//!
//! The state is always in one body-centered Cartesian frame: the frame
//! of a [`crate::SpaceEnvironment`]. On a body surface this frame is the
//! rotating body-fixed frame of a [`crate::PlanetaryBody`]. A reference
//! ellipsoid on the frame is optional. With it, the filter has the
//! geodetic observation models and publishes a geodetic solution with
//! the local NED axes at the vehicle, at each latitude and without a
//! tangent plane. [`InertialNavigationFilter::change_frame`] moves the
//! state into another frame with a different center, axes, or rotation
//! rate, and [`InertialNavigationFilter::recenter_space_frame`] is its
//! form for frames with the same rotation rate. A bounded history
//! applies a delayed IMU event or observation at its acquisition time,
//! and replays a change of mode, a burn, and a coast.
//!
//! The filter does not allocate (ADR-0016). The environment is `Copy`,
//! and the history is inline with at most [`MAX_INERTIAL_HISTORY`]
//! events. [`initial_error_covariance`] and [`error_transition`] give the
//! covariance model of the filter to a covariance analysis that has no
//! filter state, such as a Cramér–Rao bound.

mod bias_prior;
mod clock;
mod coast;
mod config;
mod discretization;
pub(crate) mod estimate;
mod frame_change;
mod history;
mod imu;
mod initialize;
mod layout;
mod local_level;
pub(crate) mod mechanization;
mod models;
pub(crate) mod nominal;
mod past;
mod propagation;
mod publish;
mod reference;
pub(crate) mod regime;
mod stochastic;
mod update;

#[cfg(test)]
mod tests;

use nalgebra::{Const, DimMin};
use navigate_contract::{
    ClockDomainId, FrameId, MonotonicNanos, ObservationStamp, SourceComposition, SourceEpoch,
};

use self::estimate::InertialCore;
use self::nominal::FULL_STATES;
use crate::InnovationStats;
use crate::admission::{check_clock, check_composition, check_freshness, check_value};
use crate::kinematic::stats::StatsTable;
use crate::observation::{MeasurementKind, Observation};
use crate::provenance::ProvenanceLedger;
use crate::rejection::{IngestOutcome, RejectionCounters, RejectionReason};
use crate::solution_sequence::SolutionSequencer;
pub use bias_prior::ImuBiasPrior;
pub use clock::ClockNoiseModel;
pub use config::InertialConfig;
pub use frame_change::FrameChange;
use history::History;
pub use history::MAX_INERTIAL_HISTORY;
pub use imu::{ImuIncrement, ImuSample};
pub use layout::{CORE_STATES, StateLayout};
pub use local_level::ned_to_body_fixed_dcm;
use navigate_astrodynamics::{PlanetaryBody, SpaceEnvironment};
pub use past::{PastState, PastStateRefusal};
pub use propagation::{ForceModelCoast, PropagationMode};
pub use reference::{ErrorMatrix, NominalPoint, error_transition, initial_error_covariance};
use regime::Regime;
use update::Event;

/// Sans-IO inertial filter with twenty-two error states (ADR-0014,
/// ADR-0018).
///
/// All time is caller-supplied [`MonotonicNanos`] on the clock domain of
/// the filter. Identical event scripts give identical solutions.
#[derive(Debug, Clone)]
pub struct InertialNavigationFilter<const N: usize = 22> {
    config: InertialConfig,
    clock: ClockDomainId,
    /// The regime before the initialization and after a reset.
    setup: Regime,
    /// The propagation mode before the initialization and after a reset.
    setup_mode: PropagationMode,
    epoch: SourceEpoch,
    sequencer: SolutionSequencer,
    rejections: RejectionCounters,
    ledger: ProvenanceLedger,
    core: Option<InertialCore<N>>,
    history: History<N>,
    stats: StatsTable,
    /// The IMU bias prior of the next initialization.
    bias_prior: Option<ImuBiasPrior>,
}

impl InertialNavigationFilter<FULL_STATES> {
    /// Builds an uninitialized filter in the rotating body-fixed frame of
    /// `body`, from [`SpaceEnvironment::from_body`], with the ellipsoid of
    /// `body` attached. A body that is not on
    /// [`navigate_geodesy::BodyEllipsoid::WGS84`] sets
    /// `config.fusion.require_explicit_frame` to `true`, as
    /// [`crate::NavigationFilter::new_for_ellipsoid`] does. The error
    /// state has every block: [`StateLayout::FULL`].
    #[must_use]
    pub fn new_surface(config: InertialConfig, clock: ClockDomainId, body: PlanetaryBody) -> Self {
        let mut config = config;
        config.layout = StateLayout::FULL;
        if body.ellipsoid != navigate_geodesy::BodyEllipsoid::WGS84 {
            config.fusion.require_explicit_frame = true;
        }
        Self::build(config, clock, Regime::surface_of(&body))
    }

    /// Builds an uninitialized filter in the frame of `env` without a
    /// reference ellipsoid. The filter sets
    /// `config.fusion.require_explicit_frame` to `true`. The error state
    /// has every block: [`StateLayout::FULL`].
    #[must_use]
    pub fn new_space(config: InertialConfig, clock: ClockDomainId, env: SpaceEnvironment) -> Self {
        let mut config = config;
        config.layout = StateLayout::FULL;
        config.fusion.require_explicit_frame = true;
        Self::build(config, clock, Regime::space(env))
    }
}

impl<const N: usize> InertialNavigationFilter<N>
where
    Const<N>: DimMin<Const<N>, Output = Const<N>>,
{
    fn build(config: InertialConfig, clock: ClockDomainId, setup: Regime) -> Self {
        Self {
            config,
            clock,
            setup,
            setup_mode: PropagationMode::SensedImu,
            epoch: SourceEpoch::new(0),
            sequencer: SolutionSequencer::new(),
            rejections: RejectionCounters::default(),
            ledger: ProvenanceLedger::default(),
            core: None,
            history: History::default(),
            stats: StatsTable::default(),
            bias_prior: None,
        }
    }

    /// Offers one IMU sample. The gates run in order: clock domain,
    /// composition, plausibility, freshness, per-source ordering, then the
    /// frame at the end time of the sample. A sample enters the
    /// provenance. It does not reset the observation silence, because a
    /// propagation sensor cannot cross-check the state (ADR-0004).
    pub fn ingest_imu(&mut self, sample: &ImuSample, now: MonotonicNanos) -> IngestOutcome {
        let event = Event::Imu(*sample);
        let result = self.admit_imu(
            &sample.stamp,
            sample.composition,
            sample.is_plausible(),
            event,
            now,
        );
        self.outcome(result, None)
    }

    /// Offers one IMU increment with the gates of [`Self::ingest_imu`].
    /// A full interval propagates with the rotation correction of the
    /// velocity increment. A part of an interval propagates with the mean
    /// rates of the increment, so a delayed observation splits the
    /// interval as it splits the interval of an [`ImuSample`].
    pub fn ingest_imu_increment(
        &mut self,
        increment: &ImuIncrement,
        now: MonotonicNanos,
    ) -> IngestOutcome {
        let event = Event::Increment(*increment);
        let plausible = increment.is_plausible();
        let result = self.admit_imu(
            &increment.stamp,
            increment.composition,
            plausible,
            event,
            now,
        );
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
        self.outcome(
            result,
            Some((obs.value.kind(), obs.value.constrained_state())),
        )
    }

    fn outcome(
        &mut self,
        result: Result<(), RejectionReason>,
        measurement: Option<(MeasurementKind, crate::provenance::ConstrainedState)>,
    ) -> IngestOutcome {
        match result {
            Ok(()) => IngestOutcome::Accepted,
            Err(reason) => {
                self.rejections.record(&reason);
                if let (RejectionReason::InnovationGate { .. }, Some((kind, constrained))) =
                    (&reason, measurement)
                {
                    self.stats.record_gate_rejection(kind);
                    self.ledger.record_gate_rejection(constrained);
                }
                IngestOutcome::Rejected(reason)
            }
        }
    }

    fn admit_imu(
        &mut self,
        stamp: &ObservationStamp,
        composition: SourceComposition,
        plausible: bool,
        event: Event,
        now: MonotonicNanos,
    ) -> Result<(), RejectionReason> {
        check_clock(self.clock, stamp)?;
        check_composition(composition)?;
        if !plausible {
            return Err(RejectionReason::NonFiniteValue);
        }
        check_freshness(stamp, now, self.config.fusion.staleness_bound)?;
        self.ledger.check_source_ordering(stamp)?;
        self.apply_event(event, stamp.acquired_at)?;
        self.ledger.track(stamp, composition);
        Ok(())
    }

    fn admit(&mut self, obs: &Observation, now: MonotonicNanos) -> Result<(), RejectionReason> {
        check_clock(self.clock, &obs.stamp)?;
        check_composition(obs.composition)?;
        let kind = obs.value.kind();
        // A delayed observation meets the regime of its acquisition time
        // in the replay.
        let delayed = self
            .core
            .as_ref()
            .is_some_and(|core| obs.stamp.acquired_at < core.state_at);
        if !delayed && !self.active_regime().supports(kind) {
            return Err(RejectionReason::UnsupportedMeasurement { kind });
        }
        check_value(&obs.value)?;
        check_freshness(&obs.stamp, now, self.config.fusion.staleness_bound)?;
        self.ledger.check_source_ordering(&obs.stamp)?;
        if let Some(nis) = self.apply_event(Event::Observation(*obs), obs.stamp.acquired_at)? {
            self.stats.record_admitted(kind, nis);
        }
        let window = self.config.fusion.assessment_window;
        self.ledger.commit(
            &obs.stamp,
            obs.composition,
            obs.value.constrained_state(),
            now,
            window,
        );
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
        let mut candidate = *core;
        let chi2 = update::apply_event(
            &mut candidate,
            &event,
            at,
            &self.config,
            estimate::Mode::Gated,
            None,
        )?;
        let capacity = self.config.fusion.history_capacity;
        self.history.record(*core, event, at, capacity);
        self.core = Some(candidate);
        Ok(chi2)
    }

    /// Starts a new filter epoch: forgets the state, the history, the
    /// per-source tracking, the window, and the innovation statistics.
    /// Rejection counters persist. The active regime and the active
    /// propagation mode persist, so a filter that changed its frame
    /// starts again in the new frame.
    pub fn reset(&mut self) {
        self.setup = self.active_regime();
        self.setup_mode = self.propagation_mode();
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
        self.active_regime().frame()
    }

    /// The regime of the state, or the regime of the next
    /// initialization.
    fn active_regime(&self) -> Regime {
        self.core.as_ref().map_or(self.setup, |core| core.regime)
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
