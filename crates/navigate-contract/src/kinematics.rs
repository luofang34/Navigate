//! Kinematic value types in canonical units: radians, meters, meters per
//! second. Frames are explicit in the type names; nothing here converts —
//! conversions live in `navigate-geodesy`.

/// A WGS84 geodetic position. Altitude is height above the ellipsoid.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct GeodeticPosition {
    /// Latitude in radians, positive north, in `[-π/2, π/2]`.
    pub latitude_rad: f64,
    /// Longitude in radians, positive east, in `[-π, π]`.
    pub longitude_rad: f64,
    /// Height above the WGS84 ellipsoid in meters.
    pub altitude_m: f64,
}

impl GeodeticPosition {
    /// Builds a position from its parts. Range checking is the
    /// consumer's admission concern; [`Self::is_plausible`] is the
    /// shared predicate.
    #[must_use]
    pub const fn new(latitude_rad: f64, longitude_rad: f64, altitude_m: f64) -> Self {
        Self {
            latitude_rad,
            longitude_rad,
            altitude_m,
        }
    }

    /// Whether all components are finite and within geodetic range.
    #[must_use]
    pub fn is_plausible(&self) -> bool {
        self.latitude_rad.is_finite()
            && self.longitude_rad.is_finite()
            && self.altitude_m.is_finite()
            && self.latitude_rad.abs() <= core::f64::consts::FRAC_PI_2
            && self.longitude_rad.abs() <= core::f64::consts::PI
    }
}

/// A velocity in the local-level North-East-Down frame.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct NedVelocity {
    /// North component in meters per second.
    pub north_mps: f64,
    /// East component in meters per second.
    pub east_mps: f64,
    /// Down component in meters per second (positive descending).
    pub down_mps: f64,
}

impl NedVelocity {
    /// Builds a velocity from its parts.
    #[must_use]
    pub const fn new(north_mps: f64, east_mps: f64, down_mps: f64) -> Self {
        Self {
            north_mps,
            east_mps,
            down_mps,
        }
    }

    /// Whether all components are finite.
    #[must_use]
    pub fn is_finite(&self) -> bool {
        self.north_mps.is_finite() && self.east_mps.is_finite() && self.down_mps.is_finite()
    }
}

/// A unit quaternion attitude (body relative to local-level NED),
/// scalar-first.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct AttitudeQuaternion {
    /// Scalar component.
    pub w: f64,
    /// First vector component.
    pub x: f64,
    /// Second vector component.
    pub y: f64,
    /// Third vector component.
    pub z: f64,
}

impl AttitudeQuaternion {
    /// Builds a quaternion from its parts; the caller upholds unit norm.
    #[must_use]
    pub const fn new(w: f64, x: f64, y: f64, z: f64) -> Self {
        Self { w, x, y, z }
    }
}

/// A symmetric 3×3 covariance stored as its upper triangle
/// `[xx, xy, xz, yy, yz, zz]`. Axis meaning is the carrying field's
/// (NED for position and velocity covariances).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SymmetricCov3 {
    upper: [f64; 6],
}

impl SymmetricCov3 {
    /// Builds a covariance from its upper triangle
    /// `[xx, xy, xz, yy, yz, zz]`.
    #[must_use]
    pub const fn from_upper_triangle(upper: [f64; 6]) -> Self {
        Self { upper }
    }

    /// A diagonal covariance from per-axis variances.
    #[must_use]
    pub const fn from_diagonal(xx: f64, yy: f64, zz: f64) -> Self {
        Self {
            upper: [xx, 0.0, 0.0, yy, 0.0, zz],
        }
    }

    /// The upper triangle `[xx, xy, xz, yy, yz, zz]`.
    #[must_use]
    pub const fn upper_triangle(&self) -> [f64; 6] {
        self.upper
    }

    /// Diagonal variances `(xx, yy, zz)`.
    #[must_use]
    pub const fn diagonal(&self) -> (f64, f64, f64) {
        (self.upper[0], self.upper[3], self.upper[5])
    }

    /// Whether every element is finite and every diagonal variance is
    /// non-negative. This is a plausibility screen, not a positive-
    /// semidefiniteness proof; the filter's admission gate performs the
    /// full check.
    #[must_use]
    pub fn is_plausible(&self) -> bool {
        let finite = self.upper.iter().all(|v| v.is_finite());
        let (xx, yy, zz) = self.diagonal();
        finite && xx >= 0.0 && yy >= 0.0 && zz >= 0.0
    }
}
