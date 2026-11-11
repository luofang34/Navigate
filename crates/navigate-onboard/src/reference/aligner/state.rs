//! The reference navigator before and at its first state: what it does
//! with an input while it aligns, and how a state is set and forgotten.

use navigate_contract::{SensorClass, SourceComposition, SymmetricCov3};

use super::{measurement, names};
use crate::{
    AcceptanceRecord, AcceptanceSink, Alignment, AlignmentConfig, AlignmentRecord, AlignmentStatus,
    ErrorBudget, OnboardError, Outcome, ReferenceNavigator, SensorData, SensorInput,
};

impl ReferenceNavigator {
    /// Takes an input before the first state: the alignment uses what it
    /// can, and each aiding input gets a record that says so.
    pub(in crate::reference) fn aligning(
        &mut self,
        input: &SensorInput,
        budget: ErrorBudget,
        sink: &mut dyn AcceptanceSink,
    ) -> Result<(), OnboardError> {
        match &input.data {
            SensorData::ImuIncrement {
                dt_s,
                delta_theta_rad,
                delta_velocity_m_s,
            } => {
                let solved = self.aligner.increment(
                    input.acquired_at,
                    *dt_s,
                    *delta_theta_rad,
                    *delta_velocity_m_s,
                );
                return solved.map_or(Ok(()), |s| self.set_state(&s.alignment, s.record));
            }
            SensorData::GnssFix {
                position,
                velocity,
                position_cov,
                ..
            } => {
                let declared = match budget {
                    ErrorBudget::Gnss {
                        horizontal_sigma_m: h,
                        vertical_sigma_m: v,
                        ..
                    } => Some(SymmetricCov3::from_diagonal(h * h, h * h, v * v)),
                    _ => None,
                };
                self.aligner
                    .gnss(*position, *velocity, position_cov.or(declared));
            }
            SensorData::MagneticField { field_body_t } => self.aligner.magnetic(*field_body_t),
            SensorData::SurveyedPosition {
                position,
                horizontal_sigma_m,
                vertical_sigma_m,
            } => self
                .aligner
                .surveyed(*position, (*horizontal_sigma_m, *vertical_sigma_m)),
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

    /// Sets the state of the filter and what the navigator states about
    /// it. What the navigator kept about an earlier state goes.
    pub(in crate::reference) fn set_state(
        &mut self,
        a: &Alignment,
        record: AlignmentRecord,
    ) -> Result<(), OnboardError> {
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
            .map_err(|reason| OnboardError::Refused {
                stage: "alignment",
                reason,
            })?;
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
        self.set_state(truth, record)
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
