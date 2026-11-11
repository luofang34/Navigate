//! The reference navigator before and at its first state: what it does
//! with an input while it aligns, and how a state is set and forgotten.

use navigate_contract::{SensorClass, SourceComposition, SymmetricCov3};
use navigate_fusion::ImuBiasPrior;

use super::{measurement, names};
use crate::{
    AcceptanceRecord, AcceptanceSink, Alignment, AlignmentConfig, AlignmentRecord, AlignmentStatus,
    ErrorBudget, GnssCorrelatedError, OnboardError, Outcome, ReferenceNavigator, SensorData,
    SensorInput,
};

impl ReferenceNavigator {
    /// Takes an input before the first state, whose data is of the kind
    /// of its sensor. The alignment uses what it can; each input of
    /// another sensor than the inertial unit gets a record that the
    /// alignment took it, whether it used it or not. A star exposure mark
    /// holds no prior, because there is no state to give one.
    pub(in crate::reference) fn aligning(
        &mut self,
        input: &SensorInput,
        budget: ErrorBudget,
        sink: &mut dyn AcceptanceSink,
    ) -> Result<(), OnboardError> {
        let at = input.acquired_at;
        match &input.data {
            SensorData::ImuIncrement {
                dt_s,
                delta_theta_rad,
                delta_velocity_m_s,
            } => {
                let solved =
                    self.aligner
                        .increment(at, *dt_s, *delta_theta_rad, *delta_velocity_m_s);
                return solved.map_or(Ok(()), |s| {
                    self.set_state(&s.alignment, s.record, Some(s.biases))
                });
            }
            SensorData::StarExposureMid => {
                if let Some(channel) = self.stars.get_mut(&input.sensor.get()) {
                    channel.hold_prior(None);
                }
                return Ok(());
            }
            SensorData::GnssFix {
                position,
                velocity,
                position_cov,
                velocity_cov,
            } => {
                let (declared, declared_speed, slow) = match budget {
                    ErrorBudget::Gnss {
                        horizontal_sigma_m: h,
                        vertical_sigma_m: v,
                        velocity_sigma_m_s,
                        correlated,
                    } => (
                        Some(SymmetricCov3::from_diagonal(h * h, h * h, v * v)),
                        Some(velocity_sigma_m_s),
                        correlated,
                    ),
                    _ => (None, None, None),
                };
                // The largest axis of a stated velocity covariance, or the
                // declared one-sigma of the receiver.
                let speed_sigma = velocity_cov
                    .map(|c| {
                        let (n, e, d) = c.diagonal();
                        n.max(e).max(d).sqrt()
                    })
                    .or(declared_speed);
                let velocity = velocity.zip(speed_sigma);
                let covariance = position_cov.or(declared).map(|c| with_slow_error(c, slow));
                self.aligner.gnss(at, *position, velocity, covariance);
            }
            SensorData::MagneticField { field_body_t } => {
                self.aligner.magnetic(at, *field_body_t);
            }
            SensorData::SurveyedPosition {
                position,
                horizontal_sigma_m,
                vertical_sigma_m,
            } => self
                .aligner
                .surveyed(at, *position, (*horizontal_sigma_m, *vertical_sigma_m)),
            _ => {}
        }
        sink.record(AcceptanceRecord {
            sensor: input.sensor,
            acquired_at: input.acquired_at,
            measurement: measurement(&input.data),
            outcome: Outcome::Unused(names::ALIGNMENT),
            context: None,
        });
        Ok(())
    }

    /// Sets the state of the filter, with the IMU biases of a
    /// self-alignment or none for a state that the host gave, and what the
    /// navigator states about it. What the navigator kept about an earlier
    /// state goes.
    pub(in crate::reference) fn set_state(
        &mut self,
        a: &Alignment,
        record: AlignmentRecord,
        biases: Option<ImuBiasPrior>,
    ) -> Result<(), OnboardError> {
        let refused = |reason| OnboardError::Refused {
            stage: "alignment",
            reason,
        };
        self.filter.set_imu_bias_prior(biases).map_err(refused)?;
        self.filter
            .initialize_surface(
                a.position,
                a.velocity,
                a.attitude,
                a.position_cov,
                a.velocity_cov,
                a.attitude_cov,
                a.at,
                SourceComposition::of(SensorClass::Inertial),
            )
            .map_err(refused)?;
        self.forget_earlier_state();
        self.alignment = Some(record);
        Ok(())
    }

    /// Sets the state from the true state of a simulation, for an
    /// evaluation that needs a navigator with a known start. A flight has
    /// no such value: the entry exists only with the feature `evaluation`
    /// and in the tests of this crate, and it is not part of the
    /// [`crate::Navigator`] trait.
    ///
    /// # Errors
    ///
    /// Refuses a state that the filter does not accept.
    #[cfg(any(test, feature = "evaluation"))]
    pub fn align_from_truth(&mut self, truth: &Alignment) -> Result<(), OnboardError> {
        let record = AlignmentRecord {
            source: crate::AlignmentSource::Truth,
            ..super::transfer_record(truth)
        };
        self.set_state(truth, record, None)
    }

    /// Forgets the state: the navigator aligns again from its sensors.
    pub(in crate::reference) fn forget_state(&mut self) {
        self.filter.reset();
        self.forget_earlier_state();
    }

    fn forget_earlier_state(&mut self) {
        self.alignment = None;
        self.gate_refusals = 0;
        self.aligner.restart();
        for channel in self.stars.values_mut() {
            channel.forget_state();
        }
        // The cells of the terrain means say that a map error is in the
        // state of the filter. A new state holds none.
        #[cfg(feature = "terrain-camera")]
        self.terrain.forget_fixes();
    }

    /// Where the navigator is on the way to its first state.
    pub(in crate::reference) fn alignment_now(&self) -> AlignmentStatus {
        match self.alignment {
            Some(record) => AlignmentStatus::Aligned {
                record,
                suspect: self.gate_refusals >= self.aligner.config.suspect_after,
            },
            None => {
                let (phase, waiting_for) = self.aligner.waiting();
                AlignmentStatus::Aligning { phase, waiting_for }
            }
        }
    }

    /// Sets how the navigator aligns itself.
    ///
    /// # Errors
    ///
    /// Refuses a value that is not finite and positive.
    pub fn set_alignment_config(&mut self, config: AlignmentConfig) -> Result<(), OnboardError> {
        if !config.is_usable() {
            return Err(OnboardError::Config {
                field: "alignment configuration",
                detail: format!("{config:?}"),
            });
        }
        self.aligner.config = config;
        Ok(())
    }

    /// Records an input of a sensor that the navigator uses only to align.
    pub(in crate::reference) fn alignment_only(input: &SensorInput, sink: &mut dyn AcceptanceSink) {
        sink.record(AcceptanceRecord {
            sensor: input.sensor,
            acquired_at: input.acquired_at,
            measurement: measurement(&input.data),
            outcome: Outcome::Unused(names::ALIGNMENT_ONLY),
            context: None,
        });
    }
}

/// `covariance` with the time-correlated part of a GNSS error added. A
/// start takes one fix, and that fix holds the slow error in full.
fn with_slow_error(covariance: SymmetricCov3, slow: Option<GnssCorrelatedError>) -> SymmetricCov3 {
    let Some(slow) = slow else {
        return covariance;
    };
    let [xx, xy, xz, yy, yz, zz] = covariance.upper_triangle();
    let h = slow.horizontal_sigma_m * slow.horizontal_sigma_m;
    let v = slow.vertical_sigma_m * slow.vertical_sigma_m;
    SymmetricCov3::from_upper_triangle([xx + h, xy, xz, yy + h, yz, zz + v])
}
