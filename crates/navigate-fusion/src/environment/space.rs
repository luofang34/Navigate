//! Gravity in a Cartesian frame centered on one body (ADR-0014).

use nalgebra::{Matrix3, Vector3};
use navigate_contract::FrameId;

use super::{GravityField, MAX_ABS_J2, centrifugal, centrifugal_gradient};

/// Below this distance from the center in meters, the central gravity
/// and its gradient are zero. The point-mass field is singular there.
const MIN_CENTRAL_DISTANCE_M: f64 = 1.0;

/// A third body as a point mass at a known position in the frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PointMassBody {
    /// NAIF code of the body.
    pub naif_id: i32,
    /// Gravitational parameter `μ` in m³/s².
    pub gm_m3_per_s2: f64,
    /// Position of the body in the frame in meters.
    pub position_in_frame_m: [f64; 3],
}

impl PointMassBody {
    fn position(&self) -> Vector3<f64> {
        Vector3::from(self.position_in_frame_m)
    }

    /// Whether `μ` is finite and positive and the position is finite and
    /// not at the frame center.
    #[must_use]
    pub fn is_plausible(&self) -> bool {
        let r = self.position();
        self.gm_m3_per_s2.is_finite()
            && self.gm_m3_per_s2 > 0.0
            && r.iter().all(|c| c.is_finite())
            && r.norm() >= MIN_CENTRAL_DISTANCE_M
    }

    /// The tidal acceleration `μ·(d/|d|³ − r_k/|r_k|³)` with `d = r_k − r`.
    /// The second term is the acceleration of the frame center, so the
    /// acceleration is zero at the center.
    pub(crate) fn tidal_acceleration(&self, r: &Vector3<f64>) -> Vector3<f64> {
        let body = self.position();
        let d = body - r;
        let d_norm = d.norm();
        let body_norm = body.norm();
        if d_norm < MIN_CENTRAL_DISTANCE_M || body_norm < MIN_CENTRAL_DISTANCE_M {
            return Vector3::zeros();
        }
        self.gm_m3_per_s2 * (d / d_norm.powi(3) - body / body_norm.powi(3))
    }

    /// The tidal gradient `μ/|d|³·(3·d·dᵀ/|d|² − I)`.
    pub(crate) fn tidal_gradient(&self, r: &Vector3<f64>) -> Matrix3<f64> {
        let d = self.position() - r;
        let d_norm = d.norm();
        if d_norm < MIN_CENTRAL_DISTANCE_M {
            return Matrix3::zeros();
        }
        let unit = d / d_norm;
        self.gm_m3_per_s2 / d_norm.powi(3) * (3.0 * unit * unit.transpose() - Matrix3::identity())
    }
}

/// The point mass and the `J2` zonal harmonic of the central body about
/// a pole in the frame axes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CentralBodyGravity {
    /// Gravitational parameter `μ` in m³/s².
    pub gm_m3_per_s2: f64,
    /// Unnormalized `J2` zonal harmonic. It has no unit.
    pub j2: f64,
    /// Reference radius `a` of `J2` in meters.
    pub equatorial_radius_m: f64,
    /// Unit pole of the body in the frame axes.
    pub pole_unit_in_frame: [f64; 3],
}

impl CentralBodyGravity {
    /// Whether every constant is finite, `μ` and `a` are positive, `J2`
    /// is small, and the pole has unit norm.
    #[must_use]
    pub fn is_plausible(&self) -> bool {
        let pole_norm = Vector3::from(self.pole_unit_in_frame).norm();
        self.gm_m3_per_s2.is_finite()
            && self.gm_m3_per_s2 > 0.0
            && self.j2.is_finite()
            && self.j2.abs() < MAX_ABS_J2
            && self.equatorial_radius_m.is_finite()
            && self.equatorial_radius_m > 0.0
            && (pole_norm - 1.0).abs() <= 1e-9
    }

    /// The coordinate-free gravity with `k` the pole, `s = r·k`, and
    /// `u = s/|r|`:
    /// `g = −μ/|r|³·(r + 1.5·J2·(a/|r|)²·((1 − 5u²)·r + 2·s·k))`.
    pub(crate) fn acceleration(&self, r: &Vector3<f64>) -> Vector3<f64> {
        let r_norm = r.norm();
        if r_norm < MIN_CENTRAL_DISTANCE_M {
            return Vector3::zeros();
        }
        let k = Vector3::from(self.pole_unit_in_frame);
        let s = r.dot(&k);
        let u = s / r_norm;
        let ratio = self.equatorial_radius_m / r_norm;
        let zonal = 1.5 * self.j2 * ratio * ratio * ((1.0 - 5.0 * u * u) * r + 2.0 * s * k);
        -self.gm_m3_per_s2 / r_norm.powi(3) * (r + zonal)
    }

    /// The exact gradient `∂g/∂r` of [`Self::acceleration`].
    pub(crate) fn gradient(&self, r: &Vector3<f64>) -> Matrix3<f64> {
        let r_norm = r.norm();
        if r_norm < MIN_CENTRAL_DISTANCE_M {
            return Matrix3::zeros();
        }
        let identity = Matrix3::identity();
        let mu = self.gm_m3_per_s2;
        let r2 = r_norm * r_norm;
        let point_mass = -mu / r_norm.powi(3) * (identity - 3.0 * r * r.transpose() / r2);
        let k = Vector3::from(self.pole_unit_in_frame);
        let s = r.dot(&k);
        let u2 = s * s / r2;
        // g₂ = −c·w/|r|⁵ with c = 1.5·μ·J2·a² and
        // w = (1 − 5u²)·r + 2·s·k.
        let c = 1.5 * mu * self.j2 * self.equatorial_radius_m * self.equatorial_radius_m;
        let w = (1.0 - 5.0 * u2) * r + 2.0 * s * k;
        let dw = (1.0 - 5.0 * u2) * identity - 10.0 * s / r2 * r * k.transpose()
            + 10.0 * s * s / (r2 * r2) * r * r.transpose()
            + 2.0 * k * k.transpose();
        let r5 = r2 * r2 * r_norm;
        let zonal = -c * (dw / r5 - 5.0 * w * r.transpose() / (r5 * r2));
        point_mass + zonal
    }
}

/// The gravity model of a [`SpaceEnvironment`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum SpaceGravityModel {
    /// No gravity, for a test or a short coast far from every body.
    Zero,
    /// The central body alone.
    CentralBody(CentralBodyGravity),
    /// The central body and the tidal terms of third bodies.
    MultiBody {
        /// The central body at the frame center.
        central: CentralBodyGravity,
        /// The third bodies at their positions at the current time.
        third_bodies: Vec<PointMassBody>,
    },
}

impl SpaceGravityModel {
    fn is_plausible(&self) -> bool {
        match self {
            Self::Zero => true,
            Self::CentralBody(central) => central.is_plausible(),
            Self::MultiBody {
                central,
                third_bodies,
            } => central.is_plausible() && third_bodies.iter().all(PointMassBody::is_plausible),
        }
    }

    /// The gravity at `r` and its gradient.
    pub(crate) fn evaluate(&self, r: &Vector3<f64>) -> (Vector3<f64>, Matrix3<f64>) {
        match self {
            Self::Zero => (Vector3::zeros(), Matrix3::zeros()),
            Self::CentralBody(central) => (central.acceleration(r), central.gradient(r)),
            Self::MultiBody {
                central,
                third_bodies,
            } => third_bodies.iter().fold(
                (central.acceleration(r), central.gradient(r)),
                |(g, gradient), body| {
                    (
                        g + body.tidal_acceleration(r),
                        gradient + body.tidal_gradient(r),
                    )
                },
            ),
        }
    }
}

/// A Cartesian navigation frame centered on one body (ADR-0014).
#[derive(Debug, Clone, PartialEq)]
pub struct SpaceEnvironment {
    /// The frame of the state, the observations, and the solutions.
    pub frame: FrameId,
    /// Rotation vector of the frame axes relative to inertial space, in
    /// rad/s in the frame axes. Zero for an inertial frame.
    pub frame_rotation_rad_per_s: [f64; 3],
    /// The gravity model.
    pub gravity: SpaceGravityModel,
}

impl SpaceEnvironment {
    /// An environment in a frame with inertial axes.
    #[must_use]
    pub const fn inertial(frame: FrameId, gravity: SpaceGravityModel) -> Self {
        Self {
            frame,
            frame_rotation_rad_per_s: [0.0; 3],
            gravity,
        }
    }

    /// Whether the rotation is finite and every gravity constant is
    /// plausible.
    #[must_use]
    pub fn is_plausible(&self) -> bool {
        self.frame_rotation_rad_per_s.iter().all(|w| w.is_finite()) && self.gravity.is_plausible()
    }
}

impl GravityField for SpaceEnvironment {
    /// The gravity plus the centrifugal term of a rotating frame.
    fn gravity(&self, p: &Vector3<f64>) -> (Vector3<f64>, Matrix3<f64>) {
        let omega = self.frame_rotation();
        let (g, gradient) = self.gravity.evaluate(p);
        (
            g + centrifugal(&omega, p),
            gradient + centrifugal_gradient(&omega),
        )
    }

    fn frame_rotation(&self) -> Vector3<f64> {
        Vector3::from(self.frame_rotation_rad_per_s)
    }
}
