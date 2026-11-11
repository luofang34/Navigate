//! A start that an operator or a host core declares: where the vehicle is,
//! how high, where it heads, and how fast, each with a one-sigma, at the
//! stamp of the input. The navigator builds the attitude and the
//! covariance from its own inertial data and this statement; it needs no
//! GNSS.

use navigate_data::VerticalDatum;
use navigate_fusion::ImuBiasPrior;

/// What a speed is measured against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpeedKind {
    /// Speed through the air, along the heading. The ground velocity
    /// differs from it by the wind.
    Airspeed,
    /// Speed over the ground, along the track.
    GroundSpeed,
}

/// A declared height and its reference.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DeclaredHeight {
    /// Height in metres above `datum`.
    pub height_m: f64,
    /// The reference of the height. A height above a geoid takes the
    /// bound of an unknown undulation as part of its error; a height of
    /// unknown reference gives no state.
    pub datum: VerticalDatum,
    /// One-sigma of the height above its datum, metres.
    pub sigma_m: f64,
}

/// A declared speed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DeclaredSpeed {
    /// What the speed is measured against.
    pub kind: SpeedKind,
    /// Speed, m/s.
    pub speed_m_s: f64,
    /// One-sigma of the speed, m/s.
    pub sigma_m_s: f64,
}

/// The start that an operator declares, valid at the stamp of its input.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DeclaredStart {
    /// Latitude, radians.
    pub latitude_rad: f64,
    /// Longitude, radians.
    pub longitude_rad: f64,
    /// Horizontal one-sigma of the position, each axis, metres.
    pub horizontal_sigma_m: f64,
    /// The height.
    pub height: DeclaredHeight,
    /// True heading, radians clockwise from north.
    pub true_heading_rad: f64,
    /// One-sigma of the heading, radians.
    pub heading_sigma_rad: f64,
    /// The speed.
    pub speed: DeclaredSpeed,
    /// The IMU biases at the start, with their covariances, in the body
    /// axes: for example the gyroscope turn-on bias that a ground
    /// calibration measured before an in-air start. `None` starts the
    /// biases at zero.
    pub imu_bias_prior: Option<ImuBiasPrior>,
}

impl DeclaredStart {
    /// Whether each value is finite and each one-sigma positive.
    #[must_use]
    pub fn is_usable(&self) -> bool {
        let values = [
            self.latitude_rad,
            self.longitude_rad,
            self.height.height_m,
            self.true_heading_rad,
            self.speed.speed_m_s,
        ];
        let sigmas = [
            self.horizontal_sigma_m,
            self.height.sigma_m,
            self.heading_sigma_rad,
            self.speed.sigma_m_s,
        ];
        values.iter().all(|v| v.is_finite())
            && sigmas.iter().all(|v| v.is_finite() && *v > 0.0)
            && self.speed.speed_m_s >= 0.0
    }
}
