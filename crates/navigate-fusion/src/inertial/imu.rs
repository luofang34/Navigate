//! One inertial measurement unit sample, as mean rates or as
//! integrated increments (ADR-0014, ADR-0017).

use navigate_contract::{FrameId, ObservationStamp, SourceComposition};

/// Upper limit of the sample interval in seconds that admission accepts.
const MAX_INTERVAL_S: f64 = 1.0;

/// Whether every value is finite and the interval is finite, positive,
/// and at most [`MAX_INTERVAL_S`].
fn plausible(values: [&[f64; 3]; 2], dt_s: f64) -> bool {
    values.iter().all(|v| v.iter().all(|c| c.is_finite()))
        && dt_s.is_finite()
        && dt_s > 0.0
        && dt_s <= MAX_INTERVAL_S
}

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
        plausible(
            [
                &self.specific_force_body_mps2,
                &self.angular_rate_body_rad_s,
            ],
            self.dt_s,
        )
    }
}

/// One output of an inertial measurement unit that integrates its
/// sensors: the rotation and the velocity increments over the interval
/// that ends at `stamp.acquired_at`.
///
/// The unit applies the coning correction to `delta_theta_body_rad` and
/// the sculling correction to `delta_velocity_body_mps`. The filter
/// applies the rotation correction of the velocity:
/// `Δv_rot = Δṽ + ½·(Δθ̃ × Δṽ)`, with `Δθ̃ = Δθ − b_g·Δt` and
/// `Δṽ = Δv − b_a·Δt`. Thus `Δv_rot` is in the body axes at the start
/// of the interval. A delayed observation inside the interval splits it.
/// Each part then uses the mean rates `Δv/Δt` and `Δθ/Δt`, as an
/// [`ImuSample`] does.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct ImuIncrement {
    /// Identity, ordering, and the end time of the interval.
    pub stamp: ObservationStamp,
    /// Declared provenance. It normally holds `SensorClass::Inertial`.
    pub composition: SourceComposition,
    /// Length of the interval in seconds.
    pub dt_s: f64,
    /// Rotation increment relative to inertial space in radians, in the
    /// body axes, with the coning correction.
    pub delta_theta_body_rad: [f64; 3],
    /// Specific-force velocity increment in m/s, in the body axes, with
    /// the sculling correction and without the rotation correction.
    pub delta_velocity_body_mps: [f64; 3],
    /// Declared navigation frame of the stream. The filter refuses a
    /// frame that is not its active frame.
    pub frame: Option<FrameId>,
}

impl ImuIncrement {
    /// Builds an increment from its parts without a declared frame.
    #[must_use]
    pub const fn new(
        stamp: ObservationStamp,
        composition: SourceComposition,
        dt_s: f64,
        delta_theta_body_rad: [f64; 3],
        delta_velocity_body_mps: [f64; 3],
    ) -> Self {
        Self {
            stamp,
            composition,
            dt_s,
            delta_theta_body_rad,
            delta_velocity_body_mps,
            frame: None,
        }
    }

    /// This increment with `frame` declared.
    #[must_use]
    pub const fn with_frame(mut self, frame: FrameId) -> Self {
        self.frame = Some(frame);
        self
    }

    /// Whether every increment is finite and the interval is finite,
    /// positive, and at most one second.
    #[must_use]
    pub fn is_plausible(&self) -> bool {
        plausible(
            [&self.delta_theta_body_rad, &self.delta_velocity_body_mps],
            self.dt_s,
        )
    }
}
