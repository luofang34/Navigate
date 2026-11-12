//! The entries of the reference navigator into its GNSS check.

use nalgebra::Matrix2;
use navigate_contract::{
    GeodeticPosition, MonotonicNanos, NavigationSolution, SensorClass, SymmetricCov3,
};
use navigate_fusion::Observation;

use super::{Fix, GNSS_EXCLUDED, GnssMonitor, horizontal};
use crate::{
    AcceptanceRecord, AcceptanceSink, GnssCorrelatedError, GnssIntegrityConfig, GnssIntegrityState,
    GnssIntegrityStatus, OnboardError, Outcome, ReferenceNavigator, SensorInput,
};

impl ReferenceNavigator {
    /// Starts, changes, or stops the GNSS check.
    pub(in crate::reference) fn configure_gnss_integrity(
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
    pub(in crate::reference) fn gnss_integrity_status(&self) -> Option<GnssIntegrityStatus> {
        self.monitor.as_deref().map(GnssMonitor::status)
    }

    /// Gives the reference of the check its first state.
    pub(in crate::reference) fn follow_filter(&mut self, now: MonotonicNanos) {
        if let Some(monitor) = self.monitor.as_deref_mut() {
            monitor.follow(&self.filter, now);
        }
    }

    /// The solution of the filter at `now`, with what the check states.
    pub(in crate::reference) fn checked_solution(
        &mut self,
        now: MonotonicNanos,
    ) -> Option<NavigationSolution> {
        let mut solution = self.filter.tick(now)?;
        if let Some(monitor) = self.monitor.as_deref_mut() {
            monitor.annotate(&mut solution, now, &self.filter.config().fusion);
        }
        Some(solution)
    }

    /// Whether the navigator takes the GNSS fix of `input`, whose white
    /// position covariance is `white`. Without a check it takes each fix.
    pub(in crate::reference) fn takes_gnss(
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

    /// Offers an observation that a sensor made itself. While GNSS is
    /// excluded, an observation of the GNSS class is of a receiver in the
    /// same signal: it gets a record and is not offered. The check judges
    /// the fixes of a GNSS receiver only, so such an observation is never
    /// the cause of an exclusion.
    pub(in crate::reference) fn offer_unless_excluded(
        &mut self,
        input: &SensorInput,
        observation: &Observation,
        sink: &mut dyn AcceptanceSink,
    ) {
        let excluded = self.monitor.as_deref().is_some_and(|monitor| {
            monitor.state == GnssIntegrityState::Excluded
                && observation.composition.contains(SensorClass::Gnss)
        });
        if excluded {
            sink.record(AcceptanceRecord {
                sensor: input.sensor,
                acquired_at: input.acquired_at,
                measurement: "observation",
                outcome: Outcome::Unused(GNSS_EXCLUDED),
                context: None,
            });
        } else {
            self.offer(input, "observation", observation, None, sink);
        }
    }

    /// Tells the check what the filter did with the GNSS position of
    /// `input`.
    pub(in crate::reference) fn gnss_offered(&mut self, input: &SensorInput, accepted: bool) {
        if let Some(monitor) = self.monitor.as_deref_mut() {
            monitor.offered(accepted, input.acquired_at, &self.filter);
        }
    }
}
