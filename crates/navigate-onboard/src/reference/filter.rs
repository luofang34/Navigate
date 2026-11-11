//! The inertial filter of the reference navigator, with the layout that
//! its sensor suite needs.

use navigate_contract::{
    AttitudeQuaternion, FrameId, GeodeticPosition, MonotonicNanos, NavigationSolution, NedVelocity,
    SourceComposition, SymmetricCov3,
};
use navigate_fusion::{
    ImuIncrement, InertialConfig, InertialNavigationFilter, IngestOutcome, LayoutError,
    Observation, PastState, PastStateRefusal, PlanetaryBody, RejectionCounters, RejectionReason,
    StateLayout,
};

/// The dimension of [`StateLayout::FULL`] with one correlated-error
/// block.
const CORRELATED_STATES: usize = 25;

/// The inertial filter of a navigator: the 22 states of
/// [`StateLayout::FULL`], or those states and one correlated-error block
/// when a sensor of the suite declares a time-correlated error. The
/// block holds the error of the first such source that sends a fix. A
/// second source has its fixes inflated by the filter instead.
#[derive(Debug, Clone)]
pub enum NavigatorFilter {
    /// The 22 states of [`StateLayout::FULL`]. Boxed: the inline history of
    /// a filter is hundreds of kilobytes.
    Full(Box<InertialNavigationFilter>),
    /// [`StateLayout::FULL`] with one correlated-error block.
    Correlated(Box<InertialNavigationFilter<CORRELATED_STATES>>),
}

/// Runs the same expression on the filter of either layout.
macro_rules! each {
    ($self:expr, $filter:ident => $body:expr) => {
        match $self {
            NavigatorFilter::Full($filter) => $body,
            NavigatorFilter::Correlated($filter) => $body,
        }
    };
}

impl NavigatorFilter {
    /// An uninitialized filter on WGS84 with `config`, with a
    /// correlated-error block when `correlated` is set.
    ///
    /// # Errors
    ///
    /// [`LayoutError`] when the configuration needs a block that the
    /// layout does not have.
    pub(crate) fn new(
        config: InertialConfig,
        clock: navigate_contract::ClockDomainId,
        correlated: bool,
    ) -> Result<Self, LayoutError> {
        let body = PlanetaryBody::EARTH_WGS84;
        if correlated {
            let config = InertialConfig {
                layout: StateLayout::FULL.with_correlated_blocks(1),
                ..config
            };
            // Boxed at once: each move of the filter by value is a copy of
            // its inline history on the stack.
            return InertialNavigationFilter::<CORRELATED_STATES>::surface_with_layout(
                config, clock, body,
            )
            .map(|filter| Self::Correlated(Box::new(filter)));
        }
        Ok(Self::Full(Box::new(InertialNavigationFilter::new_surface(
            config, clock, body,
        ))))
    }

    /// The layout of the error state.
    #[must_use]
    pub fn layout(&self) -> StateLayout {
        each!(self, f => f.layout())
    }

    /// The configuration of the filter.
    #[must_use]
    pub fn config(&self) -> &InertialConfig {
        each!(self, f => f.config())
    }

    /// The rejection counters of the filter.
    #[must_use]
    pub fn rejections(&self) -> &RejectionCounters {
        each!(self, f => f.rejections())
    }

    /// The frame of the filter ellipsoid.
    #[must_use]
    pub fn frame(&self) -> FrameId {
        each!(self, f => f.frame())
    }

    /// The solution at `now`, as [`InertialNavigationFilter::tick`] gives
    /// it.
    pub fn tick(&mut self, now: MonotonicNanos) -> Option<NavigationSolution> {
        each!(self, f => f.tick(now))
    }

    /// The state at a past time, as [`InertialNavigationFilter::state_at`]
    /// gives it.
    ///
    /// # Errors
    ///
    /// [`PastStateRefusal`] for a time that the history does not hold.
    pub fn state_at(&self, at: MonotonicNanos) -> Result<PastState, PastStateRefusal> {
        each!(self, f => f.state_at(at))
    }

    pub(crate) fn ingest(
        &mut self,
        observation: &Observation,
        now: MonotonicNanos,
    ) -> IngestOutcome {
        each!(self, f => f.ingest(observation, now))
    }

    pub(crate) fn ingest_imu_increment(
        &mut self,
        increment: &ImuIncrement,
        now: MonotonicNanos,
    ) -> IngestOutcome {
        each!(self, f => f.ingest_imu_increment(increment, now))
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "each argument is one required part of the initial state"
    )]
    pub(crate) fn initialize_surface(
        &mut self,
        position: GeodeticPosition,
        velocity: NedVelocity,
        attitude: AttitudeQuaternion,
        pos_cov: SymmetricCov3,
        vel_cov: SymmetricCov3,
        att_cov: SymmetricCov3,
        at: MonotonicNanos,
        composition: SourceComposition,
    ) -> Result<(), RejectionReason> {
        each!(self, f => f.initialize_surface(
            position, velocity, attitude, pos_cov, vel_cov, att_cov, at, composition,
        ))
    }
}
