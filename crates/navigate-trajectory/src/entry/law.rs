//! The vehicle values and the gains of the drag tracking law.

use crate::error::TrajectoryError;
use crate::state::StateVector;

/// The vehicle values and the gains of the entry law.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct EntryLaw {
    /// The lift-to-drag ratio of the vehicle at its entry angle of
    /// attack. It has no unit.
    pub lift_to_drag: f64,
    /// The density scale height `H` of the atmosphere in meters.
    pub density_scale_height_m: f64,
    /// The natural frequency `ω` of the drag error in rad/s.
    pub natural_frequency_rad_per_s: f64,
    /// The damping ratio `ζ` of the drag error. It has no unit.
    pub damping_ratio: f64,
    /// Below this sensed drag in m/s², the law does not track and
    /// commands zero bank, so that the full lift points up.
    pub min_tracking_drag_mps2: f64,
    /// The half width in radians of the heading-error deadband of the
    /// bank reversal.
    pub heading_deadband_rad: f64,
}

impl EntryLaw {
    /// The default natural frequency of the drag error in rad/s.
    pub const DEFAULT_NATURAL_FREQUENCY_RAD_PER_S: f64 = 0.05;
    /// The default damping ratio of the drag error.
    pub const DEFAULT_DAMPING_RATIO: f64 = 0.7;
    /// The default tracking threshold in m/s², approximately 0.05 of
    /// the standard gravity of the Earth.
    pub const DEFAULT_MIN_TRACKING_DRAG_MPS2: f64 = 0.5;
    /// The default heading deadband in radians, 10°.
    pub const DEFAULT_HEADING_DEADBAND_RAD: f64 = 10.0 * core::f64::consts::PI / 180.0;

    /// A law for a vehicle with the lift-to-drag ratio `lift_to_drag` in
    /// an atmosphere with the density scale height
    /// `density_scale_height_m`, with the default gains, threshold, and
    /// deadband.
    #[must_use]
    pub const fn new(lift_to_drag: f64, density_scale_height_m: f64) -> Self {
        Self {
            lift_to_drag,
            density_scale_height_m,
            natural_frequency_rad_per_s: Self::DEFAULT_NATURAL_FREQUENCY_RAD_PER_S,
            damping_ratio: Self::DEFAULT_DAMPING_RATIO,
            min_tracking_drag_mps2: Self::DEFAULT_MIN_TRACKING_DRAG_MPS2,
            heading_deadband_rad: Self::DEFAULT_HEADING_DEADBAND_RAD,
        }
    }

    /// Sets the natural frequency in rad/s and the damping ratio of the
    /// drag error.
    #[must_use]
    pub const fn with_gains(
        mut self,
        natural_frequency_rad_per_s: f64,
        damping_ratio: f64,
    ) -> Self {
        self.natural_frequency_rad_per_s = natural_frequency_rad_per_s;
        self.damping_ratio = damping_ratio;
        self
    }

    /// Sets the tracking threshold in m/s².
    #[must_use]
    pub const fn with_min_tracking_drag_mps2(mut self, min_tracking_drag_mps2: f64) -> Self {
        self.min_tracking_drag_mps2 = min_tracking_drag_mps2;
        self
    }

    /// Sets the half width of the heading deadband in radians.
    #[must_use]
    pub const fn with_heading_deadband_rad(mut self, heading_deadband_rad: f64) -> Self {
        self.heading_deadband_rad = heading_deadband_rad;
        self
    }

    /// Checks that every value is finite and positive. The damping
    /// ratio and the threshold can also be zero.
    pub(super) fn validate(&self) -> Result<(), TrajectoryError> {
        let checks = [
            (self.lift_to_drag > 0.0, "lift_to_drag"),
            (self.density_scale_height_m > 0.0, "density_scale_height_m"),
            (
                self.natural_frequency_rad_per_s > 0.0,
                "natural_frequency_rad_per_s",
            ),
            (self.damping_ratio >= 0.0, "damping_ratio"),
            (self.min_tracking_drag_mps2 >= 0.0, "min_tracking_drag_mps2"),
            (self.heading_deadband_rad > 0.0, "heading_deadband_rad"),
        ];
        let finite = [
            self.lift_to_drag,
            self.density_scale_height_m,
            self.natural_frequency_rad_per_s,
            self.damping_ratio,
            self.min_tracking_drag_mps2,
            self.heading_deadband_rad,
        ];
        for ((ok, name), value) in checks.into_iter().zip(finite) {
            if !(ok && value.is_finite()) {
                return Err(TrajectoryError::InvalidParameter { name });
            }
        }
        Ok(())
    }

    /// The commanded `cos(bank)` in `[-1, 1]` for the vehicle state, the
    /// gravitational parameter `gm`, the sensed drag, and the reference
    /// drag and rate.
    ///
    /// The sensed drag rate is `Ḋ = D·(−ḣ/H − 2D/v)`, because the density
    /// changes as `exp(−h/H)` and the drag slows the vehicle at the rate
    /// `D`.
    pub(super) fn cos_bank(
        &self,
        vehicle: &StateVector,
        gm: f64,
        drag: f64,
        reference_drag: f64,
        reference_rate: f64,
    ) -> f64 {
        let r = vehicle.position_m.norm();
        let v = vehicle.velocity_mps.norm();
        let h_dot = vehicle.position_m.dot(&vehicle.velocity_mps) / r;
        let scale = self.density_scale_height_m;
        let drag_rate = if v > 0.0 {
            drag * (-h_dot / scale - 2.0 * drag / v)
        } else {
            0.0
        };
        let omega = self.natural_frequency_rad_per_s;
        let reference_sq = reference_drag * reference_drag;
        let f1 = omega * omega * scale / reference_sq;
        let f2 = 2.0 * self.damping_ratio * omega * scale / reference_sq;
        let equilibrium = (gm / (r * r) - v * v / r) / reference_drag;
        let vertical =
            equilibrium + f1 * (drag - reference_drag) + f2 * (drag_rate - reference_rate);
        (vertical / self.lift_to_drag).clamp(-1.0, 1.0)
    }
}
