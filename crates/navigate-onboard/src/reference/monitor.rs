//! The GNSS check of the reference navigator: a reference filter that
//! takes each input but the GNSS values, and the comparison of each GNSS
//! position with its solution.
//!
//! The reference shares the inertial data with the filter of the
//! navigator, and nothing of the receiver. The error of a fix and the
//! error of the reference are thus independent, and the covariance of
//! their separation is the sum of their covariances. After the reference
//! starts again from the solution that includes GNSS, it holds the
//! receiver error of that time; the sum is then larger than the
//! covariance of the separation, and the check finds less.

use nalgebra::{Matrix2, Vector2};
use navigate_contract::{
    GeodeticPosition, MonotonicNanos, NavigationSolution, SensorClass, SolutionQuality, SourceId,
    SymmetricCov3,
};
use navigate_fusion::{FusionConfig, ImuIncrement, IngestOutcome, Observation, ObservationValue};
use navigate_geodesy::BodyEllipsoid;

use super::NavigatorFilter;
use crate::{
    GnssCorrelatedError, GnssFault, GnssIntegrityConfig, GnssIntegrityState, GnssIntegrityStatus,
    GnssSeparation, OnboardError, ReferenceNavigator, SensorInput,
};

/// The acceptance record of a GNSS value that the navigator does not
/// take while GNSS is excluded. A host counts records by this name, so it
/// does not change.
pub(crate) const GNSS_EXCLUDED: &str = "gnss_excluded";

/// One GNSS position, as the check takes it.
struct Fix {
    source: SourceId,
    position: GeodeticPosition,
    /// The white and the time-correlated error on the north and east
    /// axes, m².
    covariance: Matrix2<f64>,
    acquired_at: MonotonicNanos,
    delivered_at: MonotonicNanos,
}

/// The reference filter and the state of the check.
#[derive(Debug, Clone)]
pub(crate) struct GnssMonitor {
    config: GnssIntegrityConfig,
    reference: NavigatorFilter,
    state: GnssIntegrityState,
    since: Option<MonotonicNanos>,
    /// Fixes in a row above the exclusion threshold.
    exceedances: u32,
    agreeing_since: Option<MonotonicNanos>,
    /// The first fix of the present run of GNSS positions that the filter
    /// of the navigator accepted.
    accepting_since: Option<MonotonicNanos>,
    anchored_at: Option<MonotonicNanos>,
    position_fix_at: Option<MonotonicNanos>,
    /// The lowest exclusion bound over the time since the start of a
    /// reference that has only inertial data, m/s.
    detectable_m_s: Option<f64>,
    last: Option<GnssSeparation>,
}

/// The offset of `b` from `a` on the north and east axes at `a`, metres.
fn north_east_m(a: &GeodeticPosition, b: &GeodeticPosition) -> Vector2<f64> {
    use core::f64::consts::{PI, TAU};
    let (meridian_m, prime_vertical_m) = BodyEllipsoid::WGS84.radii_of_curvature_m(a.latitude_rad);
    let longitude = (b.longitude_rad - a.longitude_rad + PI).rem_euclid(TAU) - PI;
    Vector2::new(
        (b.latitude_rad - a.latitude_rad) * (meridian_m + a.altitude_m),
        longitude * (prime_vertical_m + a.altitude_m) * a.latitude_rad.cos(),
    )
}

/// The time from `earlier` to `later`, seconds; zero when `later` is the
/// earlier one.
fn seconds_between(earlier: MonotonicNanos, later: MonotonicNanos) -> f64 {
    later
        .elapsed_since(earlier)
        .map_or(0.0, |d| d.as_nanos() as f64 * 1e-9)
}

/// The north and east block of a NED covariance.
fn horizontal(covariance: &SymmetricCov3) -> Matrix2<f64> {
    let [nn, ne, _, ee, _, _] = covariance.upper_triangle();
    Matrix2::new(nn, ne, ne, ee)
}

/// The largest eigenvalue of a symmetric 2×2 matrix.
fn largest_eigenvalue(m: &Matrix2<f64>) -> f64 {
    let (a, b, c) = (m.m11, m.m12, m.m22);
    0.5 * (a + c) + (0.25 * (a - c) * (a - c) + b * b).sqrt()
}

impl GnssMonitor {
    /// A check with `config` beside the filter `main`. The reference
    /// takes the state of `main` at the next input after `main` has one.
    fn new(config: GnssIntegrityConfig, main: &NavigatorFilter) -> Self {
        Self {
            config,
            reference: main.clone(),
            state: GnssIntegrityState::Consistent,
            since: None,
            exceedances: 0,
            agreeing_since: None,
            accepting_since: None,
            anchored_at: None,
            position_fix_at: None,
            detectable_m_s: None,
            last: None,
        }
    }

    /// Forgets the state of the reference and of the check: the filter of
    /// the navigator has a new state or none.
    pub(crate) fn forget(&mut self) {
        self.reference.reset();
        self.state = GnssIntegrityState::Consistent;
        self.exceedances = 0;
        (self.since, self.agreeing_since, self.anchored_at) = (None, None, None);
        (self.position_fix_at, self.last, self.accepting_since) = (None, None, None);
        self.detectable_m_s = None;
    }

    /// Gives the reference the state of `main` when it has none.
    fn follow(&mut self, main: &NavigatorFilter, now: MonotonicNanos) {
        if self.anchored_at.is_none() && main.is_initialized() {
            self.reference.clone_from(main);
            self.anchored_at = Some(now);
        }
    }

    /// The filter that gives the belief of a terrain frame: the
    /// reference, so that a receiver error does not move the search.
    #[cfg(feature = "terrain-camera")]
    pub(crate) fn belief(&mut self) -> Option<&mut NavigatorFilter> {
        self.anchored_at.map(|_| &mut self.reference)
    }

    /// Gives the reference an observation that the filter of the
    /// navigator was offered, unless a GNSS receiver made it. The result
    /// says whether the reference took it.
    pub(crate) fn mirror(&mut self, observation: &Observation, now: MonotonicNanos) -> bool {
        if self.anchored_at.is_none() || observation.composition.contains(SensorClass::Gnss) {
            return false;
        }
        let accepted = self.reference.ingest(observation, now).is_accepted();
        let position = matches!(
            observation.value,
            ObservationValue::PositionFix { .. }
                | ObservationValue::CartesianPositionFix { .. }
                | ObservationValue::VisualPose { .. }
        );
        if accepted && position {
            self.position_fix_at = Some(observation.stamp.acquired_at);
            self.detectable_m_s = None;
        }
        accepted
    }

    /// Gives the reference an inertial increment that the filter of the
    /// navigator took.
    pub(crate) fn mirror_imu(&mut self, increment: &ImuIncrement, now: MonotonicNanos) {
        if self.anchored_at.is_none() {
            return;
        }
        if let IngestOutcome::Rejected(reason) = self.reference.ingest_imu_increment(increment, now)
        {
            tracing::error!(%reason, "the GNSS reference refused an inertial increment");
        }
    }

    /// The reference solution at the acquisition time of a fix. A fix
    /// older than the history of the reference meets the solution on
    /// delivery, moved back along its velocity.
    fn reference_at(&mut self, fix: &Fix) -> Option<NavigationSolution> {
        let lag_ns = fix
            .delivered_at
            .as_nanos()
            .checked_sub(fix.acquired_at.as_nanos())?;
        if lag_ns == 0 {
            return self.reference.tick(fix.delivered_at);
        }
        if let Ok(past) = self.reference.state_at(fix.acquired_at) {
            return Some(past.solution);
        }
        let mut solution = self.reference.tick(fix.delivered_at)?;
        let lag_s = lag_ns as f64 * 1e-9;
        let (meridian_m, prime_vertical_m) =
            BodyEllipsoid::WGS84.radii_of_curvature_m(solution.position.latitude_rad);
        let (p, v) = (solution.position, solution.velocity);
        solution.position = GeodeticPosition::new(
            p.latitude_rad - v.north_mps * lag_s / (meridian_m + p.altitude_m),
            p.longitude_rad
                - v.east_mps * lag_s / ((prime_vertical_m + p.altitude_m) * p.latitude_rad.cos()),
            p.altitude_m + v.down_mps * lag_s,
        );
        Some(solution)
    }

    /// `fix` against the reference, or `None` without a reference
    /// solution or with a covariance that gives no statistic.
    fn compare(&mut self, fix: &Fix) -> Option<GnssSeparation> {
        self.anchored_at?;
        let reference = self.reference_at(fix)?;
        let d = north_east_m(&reference.position, &fix.position);
        let reference_cov = horizontal(&reference.position_cov);
        let s = reference_cov + fix.covariance;
        let chi2 = d.dot(&(s.try_inverse()? * d));
        let largest = largest_eigenvalue(&s);
        if !(chi2.is_finite() && chi2 >= 0.0 && largest.is_finite() && largest > 0.0) {
            return None;
        }
        Some(GnssSeparation {
            source: fix.source,
            at: fix.acquired_at,
            separation_m: d.norm(),
            chi2,
            caution_bound_m: (self.config.caution_chi2() * largest).sqrt(),
            exclusion_bound_m: (self.config.exclusion_chi2() * largest).sqrt(),
            reference_1sigma_m: largest_eigenvalue(&reference_cov).max(0.0).sqrt(),
        })
    }

    fn enter(&mut self, state: GnssIntegrityState, at: MonotonicNanos) {
        self.state = state;
        self.since = Some(at);
        self.agreeing_since = None;
    }

    /// Judges one GNSS position; the result says whether the navigator
    /// takes the fix. On an exclusion `main` continues from the estimate
    /// of the reference, which holds nothing of the receiver.
    fn judge(&mut self, fix: &Fix, main: &mut NavigatorFilter) -> bool {
        let excluded = self.state == GnssIntegrityState::Excluded;
        let Some(test) = self.compare(fix) else {
            return !excluded;
        };
        self.last = Some(test);
        let within = test.exclusion_bound_m <= self.config.alert_limit_m;
        let calm = test.chi2 <= self.config.caution_chi2();
        if excluded {
            // A reference that says too little cannot say that the
            // receiver is honest again, and neither can one on inertial
            // data alone: its bound grows until it holds any offset.
            let long_enough = self.agreed_long_enough(calm && within, fix.acquired_at);
            let witnessed = self
                .agreeing_since
                .zip(self.position_fix_at)
                .is_some_and(|(since, fixed_at)| fixed_at >= since);
            if long_enough && witnessed {
                self.enter(GnssIntegrityState::Consistent, fix.acquired_at);
                tracing::info!(source = ?fix.source, "GNSS is taken again");
                return true;
            }
            return false;
        }
        if test.chi2 > self.config.exclusion_chi2() {
            self.exceedances = self.exceedances.saturating_add(1);
        } else {
            self.exceedances = 0;
        }
        if self.exceedances >= self.config.exclusion_fixes {
            main.adopt_estimate(&self.reference);
            self.exceedances = 0;
            self.enter(GnssIntegrityState::Excluded, fix.acquired_at);
            tracing::warn!(
                source = ?fix.source,
                separation_m = test.separation_m,
                bound_m = test.exclusion_bound_m,
                "GNSS is excluded: its position left the reference solution"
            );
            return false;
        }
        if self.state == GnssIntegrityState::Consistent {
            if !calm {
                self.enter(GnssIntegrityState::Disagreeing, fix.acquired_at);
            }
        } else if self.agreed_long_enough(calm, fix.acquired_at) {
            // One fix under the threshold among fixes above it is the
            // noise of the receiver, not the end of the disagreement.
            self.enter(GnssIntegrityState::Consistent, fix.acquired_at);
        }
        true
    }

    /// Whether the run of fixes below the caution threshold, which
    /// `agrees` continues or ends, is as long as the configured time.
    fn agreed_long_enough(&mut self, agrees: bool, at: MonotonicNanos) -> bool {
        if !agrees {
            self.agreeing_since = None;
            return false;
        }
        let since = *self.agreeing_since.get_or_insert(at);
        seconds_between(since, at) >= self.config.readmission_s
    }

    /// Takes what the filter of the navigator did with a GNSS position
    /// that the check let through, and starts the reference again from
    /// the estimate of `main` when it says too little: its exclusion
    /// bound is above the alert limit, or it holds its position on
    /// inertial data alone and is past its most sensitive time. It does
    /// so only while the fixes agree and `main` takes them: a filter that
    /// refuses the fixes at its gate is not at the receiver.
    fn offered(&mut self, accepted: bool, at: MonotonicNanos, main: &NavigatorFilter) {
        if !accepted {
            self.accepting_since = None;
            return;
        }
        let since = *self.accepting_since.get_or_insert(at);
        let settled = seconds_between(since, at) >= self.config.readmission_s;
        let Some(bound_m) = self.last.map(|test| test.exclusion_bound_m) else {
            return;
        };
        let coasted_s = self
            .anchored_at
            .map_or(0.0, |from| seconds_between(from, at));
        let mut worn = false;
        if self.position_fix_at.is_none() && coasted_s > 0.0 {
            // The slowest receiver drift that the reference finds in the
            // time since its start.
            let rate = bound_m / coasted_s;
            let lowest = self.detectable_m_s.map_or(rate, |lowest| lowest.min(rate));
            self.detectable_m_s = Some(lowest);
            worn = rate >= self.config.restart_rate_ratio * lowest;
        }
        let useless = bound_m > self.config.alert_limit_m;
        if settled && (useless || worn) && self.state == GnssIntegrityState::Consistent {
            self.reference.adopt_estimate(main);
            self.anchored_at = Some(at);
            self.position_fix_at = None;
            self.detectable_m_s = None;
        }
    }

    /// States a disagreement in `solution`, the solution of the filter of
    /// the navigator at `now`: while the fixes disagree with the
    /// reference, the truth can be at either, so the horizontal
    /// covariance gains the separation of the solution from the
    /// reference, and the quality is not `Good`.
    fn annotate(
        &mut self,
        solution: &mut NavigationSolution,
        now: MonotonicNanos,
        fusion: &FusionConfig,
    ) {
        if self.state != GnssIntegrityState::Disagreeing {
            return;
        }
        let Some(reference) = self.reference.tick(now) else {
            return;
        };
        let d = north_east_m(&reference.position, &solution.position);
        let [nn, ne, nd, ee, ed, dd] = solution.position_cov.upper_triangle();
        let stated = [nn + d.x * d.x, ne + d.x * d.y, nd, ee + d.y * d.y, ed, dd];
        if !stated.iter().all(|value| value.is_finite()) {
            return;
        }
        solution.position_cov = SymmetricCov3::from_upper_triangle(stated);
        let sigma_m = largest_eigenvalue(&horizontal(&solution.position_cov))
            .max(0.0)
            .sqrt();
        solution.integrity.horizontal_1sigma_m = sigma_m;
        let unusable = sigma_m > fusion.degraded_horizontal_1sigma_m
            || solution.integrity.quality == SolutionQuality::Unusable;
        solution.integrity.quality = if unusable {
            SolutionQuality::Unusable
        } else {
            SolutionQuality::Degraded
        };
    }

    fn status(&self) -> GnssIntegrityStatus {
        let consistent = self.state == GnssIntegrityState::Consistent;
        GnssIntegrityStatus {
            state: self.state,
            since: self.since,
            fault: (!consistent).then_some(GnssFault::HorizontalSeparation),
            last: self.last,
            within_alert_limit: self
                .last
                .is_some_and(|test| test.exclusion_bound_m <= self.config.alert_limit_m),
            reference_anchored_at: self.anchored_at,
            reference_position_fix_at: self.position_fix_at,
            inertial_detectable_rate_m_s: self.detectable_m_s,
            agreeing_since: self.agreeing_since,
        }
    }
}

impl ReferenceNavigator {
    /// Starts, changes, or stops the GNSS check.
    pub(super) fn configure_gnss_integrity(
        &mut self,
        config: Option<GnssIntegrityConfig>,
    ) -> Result<(), OnboardError> {
        let Some(config) = config else {
            self.monitor = None;
            return Ok(());
        };
        if !config.is_usable() {
            return Err(OnboardError::Config {
                field: "GNSS integrity configuration",
                detail: format!("{config:?}"),
            });
        }
        match self.monitor.as_deref_mut() {
            Some(monitor) => monitor.config = config,
            None => self.monitor = Some(Box::new(GnssMonitor::new(config, &self.filter))),
        }
        Ok(())
    }

    /// The state of the GNSS check, or `None` without one.
    pub(super) fn gnss_integrity_status(&self) -> Option<GnssIntegrityStatus> {
        self.monitor.as_deref().map(GnssMonitor::status)
    }

    /// Gives the reference of the check its first state.
    pub(super) fn follow_filter(&mut self, now: MonotonicNanos) {
        if let Some(monitor) = self.monitor.as_deref_mut() {
            monitor.follow(&self.filter, now);
        }
    }

    /// The solution of the filter at `now`, with what the check states.
    pub(super) fn checked_solution(&mut self, now: MonotonicNanos) -> Option<NavigationSolution> {
        let mut solution = self.filter.tick(now)?;
        if let Some(monitor) = self.monitor.as_deref_mut() {
            monitor.annotate(&mut solution, now, &self.filter.config().fusion);
        }
        Some(solution)
    }

    /// Whether the navigator takes the GNSS fix of `input`, whose white
    /// position covariance is `white`. Without a check it takes each fix.
    pub(super) fn takes_gnss(
        &mut self,
        input: &SensorInput,
        position: &GeodeticPosition,
        white: &SymmetricCov3,
        correlated: Option<GnssCorrelatedError>,
    ) -> bool {
        let Some(monitor) = self.monitor.as_deref_mut() else {
            return true;
        };
        let slow = correlated.map_or(0.0, |c| c.horizontal_sigma_m * c.horizontal_sigma_m);
        let fix = Fix {
            source: input.sensor,
            position: *position,
            covariance: horizontal(white) + Matrix2::from_diagonal_element(slow),
            acquired_at: input.acquired_at,
            delivered_at: input.delivered_at,
        };
        let was_excluded = monitor.state == GnssIntegrityState::Excluded;
        let taken = monitor.judge(&fix, &mut self.filter);
        if monitor.state == GnssIntegrityState::Excluded && !was_excluded {
            // The refusals were those of the state that the receiver
            // moved; the adopted state has none.
            self.gate_refusals = 0;
        }
        taken
    }

    /// Tells the check what the filter did with the GNSS position of
    /// `input`.
    pub(super) fn gnss_offered(&mut self, input: &SensorInput, accepted: bool) {
        if let Some(monitor) = self.monitor.as_deref_mut() {
            monitor.offered(accepted, input.acquired_at, &self.filter);
        }
    }
}
