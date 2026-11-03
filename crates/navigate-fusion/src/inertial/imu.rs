//! One inertial measurement unit sample (ADR-0014).

use navigate_contract::{FrameId, ObservationStamp, SourceComposition};

/// Upper limit of the sample interval in seconds that admission accepts.
const MAX_INTERVAL_S: f64 = 1.0;

/// One sample of an inertial measurement unit: the mean specific force
/// and the mean angular rate over the interval that ends at
/// `stamp.acquired_at`.
///
/// The filter holds both rates constant over the interval. The filter
/// propagates from its state time to `stamp.acquired_at`, so a sample
/// that follows an observation inside its interval propagates only the
/// remaining part of the interval.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct ImuSample {
    /// Identity, ordering, and the end time of the interval.
    pub stamp: ObservationStamp,
    /// Mean specific force in m/s², in the body axes.
    pub specific_force_body_mps2: [f64; 3],
    /// Mean angular rate relative to inertial space in rad/s, in the body
    /// axes.
    pub angular_rate_body_rad_s: [f64; 3],
    /// Length of the interval in seconds.
    pub dt_s: f64,
    /// Declared provenance. It normally holds `SensorClass::Inertial`.
    pub composition: SourceComposition,
    /// Declared navigation frame of the stream. The filter refuses a
    /// frame that is not its active frame.
    pub frame: Option<FrameId>,
}

impl ImuSample {
    /// Builds a sample from its parts without a declared frame.
    #[must_use]
    pub const fn new(
        stamp: ObservationStamp,
        specific_force_body_mps2: [f64; 3],
        angular_rate_body_rad_s: [f64; 3],
        dt_s: f64,
        composition: SourceComposition,
    ) -> Self {
        Self {
            stamp,
            specific_force_body_mps2,
            angular_rate_body_rad_s,
            dt_s,
            composition,
            frame: None,
        }
    }

    /// Builds a sample from its parts with a declared frame.
    #[must_use]
    pub const fn in_frame(
        stamp: ObservationStamp,
        specific_force_body_mps2: [f64; 3],
        angular_rate_body_rad_s: [f64; 3],
        dt_s: f64,
        composition: SourceComposition,
        frame: FrameId,
    ) -> Self {
        Self::new(
            stamp,
            specific_force_body_mps2,
            angular_rate_body_rad_s,
            dt_s,
            composition,
        )
        .with_frame(frame)
    }

    /// This sample with `frame` declared.
    #[must_use]
    pub const fn with_frame(mut self, frame: FrameId) -> Self {
        self.frame = Some(frame);
        self
    }

    /// Whether every rate is finite and the interval is finite, positive,
    /// and at most one second.
    #[must_use]
    pub fn is_plausible(&self) -> bool {
        self.specific_force_body_mps2.iter().all(|f| f.is_finite())
            && self.angular_rate_body_rad_s.iter().all(|w| w.is_finite())
            && self.dt_s.is_finite()
            && self.dt_s > 0.0
            && self.dt_s <= MAX_INTERVAL_S
    }
}
