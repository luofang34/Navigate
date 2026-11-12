//! The GNSS fixes of the reference navigator once it has a state: a
//! position with its declared slow error, and a Doppler velocity.

use navigate_contract::{SensorClass, SourceComposition, SymmetricCov3};
use navigate_fusion::{Observation, ObservationValue};

use super::monitor::GNSS_EXCLUDED;
use crate::{
    AcceptanceRecord, AcceptanceSink, ErrorBudget, Outcome, ReferenceNavigator, SensorData,
    SensorInput,
};

impl ReferenceNavigator {
    pub(super) fn gnss(
        &mut self,
        input: &SensorInput,
        budget: ErrorBudget,
        sink: &mut dyn AcceptanceSink,
    ) {
        let SensorData::GnssFix {
            position,
            velocity,
            position_cov,
            velocity_cov,
        } = &input.data
        else {
            return;
        };
        let (h, v, s, correlated) = match budget {
            ErrorBudget::Gnss {
                horizontal_sigma_m,
                vertical_sigma_m,
                velocity_sigma_m_s,
                correlated,
            } => (
                horizontal_sigma_m,
                vertical_sigma_m,
                velocity_sigma_m_s,
                correlated,
            ),
            _ => (f64::NAN, f64::NAN, f64::NAN, None),
        };
        let position_cov =
            position_cov.unwrap_or(SymmetricCov3::from_diagonal(h * h, h * h, v * v));
        if !self.takes_gnss(input, position, &position_cov, correlated) {
            // The Doppler velocity is of the same receiver: it goes with
            // the position.
            let measurements = ["gnss_position", "gnss_velocity"];
            let count = if velocity.is_some() { 2 } else { 1 };
            for measurement in measurements.into_iter().take(count) {
                sink.record(AcceptanceRecord {
                    sensor: input.sensor,
                    acquired_at: input.acquired_at,
                    measurement,
                    outcome: Outcome::Unused(GNSS_EXCLUDED),
                    context: None,
                });
            }
            return;
        }
        let value = ObservationValue::PositionFix {
            position: *position,
            covariance: position_cov,
        };
        let stamp = self.stamp(input.sensor, input.acquired_at);
        let observation = Observation::new(stamp, value, SourceComposition::of(SensorClass::Gnss));
        // The velocity is from Doppler and stays white.
        let observation = match correlated {
            Some(c) => observation.with_correlated_error(c.filter_error()),
            None => observation,
        };
        let gated = self.offer(input, "gnss_position", &observation, None, sink);
        self.gnss_offered(input, !gated);
        // A run of positions outside the gate says that the state is far
        // from the receiver: the alignment is suspect.
        self.gate_refusals = if gated {
            self.gate_refusals.saturating_add(1)
        } else {
            0
        };
        if let Some(velocity) = velocity {
            let covariance =
                velocity_cov.unwrap_or(SymmetricCov3::from_diagonal(s * s, s * s, s * s));
            let value = ObservationValue::VelocityFix {
                velocity: *velocity,
                covariance,
            };
            self.offer_value(input, "gnss_velocity", value, SensorClass::Gnss, sink);
        }
    }
}
