//! Scalar magnetic anomaly against a pre-linearized map (ADR-0012).
//!
//! The source adapter reads the anomaly map and its gradient at a
//! linearization point. The model uses the first-order expansion
//! `h(p) = map_t_at_lin + g · (p − p_lin)`. All magnetic values are SI:
//! tesla, tesla per meter, and tesla². The gradient arrives in the NED
//! axes at the linearization point. The model rotates it into the plane
//! axes, because the two sets of axes differ away from the plane origin.

use nalgebra::{Matrix3, SMatrix, SVector, Vector3};
use navigate_contract::GeodeticPosition;
use navigate_geodesy::LocalTangentPlane;

use crate::config::FusionConfig;
use crate::kalman::{Mat6, MeasurementModel, Vec6, horizontal_1sigma_m, rotation_between};
use crate::rejection::RejectionReason;

/// The map side of one magnetic measurement: the map value and gradient
/// at a linearization point, and the radius where the linear model is
/// valid.
#[derive(Debug, Clone, Copy)]
pub(crate) struct MapLinearization {
    pub(crate) linearization_point: GeodeticPosition,
    pub(crate) map_t_at_lin: f64,
    pub(crate) gradient_ned_t_per_m: [f64; 3],
    pub(crate) valid_radius_m: f64,
}

/// A magnetic anomaly measurement in plane axes.
#[derive(Debug, Clone, Copy)]
pub(crate) struct MagneticAnomalyModel {
    measured_t: f64,
    map_t_at_lin: f64,
    g_plane: Vector3<f64>,
    p_lin_plane: Vector3<f64>,
    valid_radius_m: f64,
    variance_t2: f64,
}

impl MagneticAnomalyModel {
    /// Expresses the map linearization in the axes of `plane`. `None`
    /// when the linearization point cannot anchor a tangent plane.
    pub(crate) fn new(
        plane: &LocalTangentPlane,
        map: &MapLinearization,
        measured_t: f64,
        variance_t2: f64,
    ) -> Option<Self> {
        let local_lin = LocalTangentPlane::new(map.linearization_point, plane.ellipsoid()).ok()?;
        let offset = plane.to_ned(&map.linearization_point);
        let [g_n, g_e, g_d] = map.gradient_ned_t_per_m;
        Some(Self {
            measured_t,
            map_t_at_lin: map.map_t_at_lin,
            g_plane: rotation_between(&local_lin, plane) * Vector3::new(g_n, g_e, g_d),
            p_lin_plane: Vector3::new(offset.north_m, offset.east_m, offset.down_m),
            valid_radius_m: map.valid_radius_m,
            variance_t2,
        })
    }

    fn offset_from_lin(&self, x: &Vec6) -> Vector3<f64> {
        Vector3::new(x[0], x[1], x[2]) - self.p_lin_plane
    }
}

/// Refuses a state that can be outside the valid radius. The margin is
/// `linearization_sigma_multiplier` horizontal 1-sigma values. A
/// negative or non-finite multiplier gives an infinite margin, so the
/// check fails closed.
pub(crate) fn check_linearization_region(
    offset_m: f64,
    horizontal_1sigma_m: f64,
    valid_radius_m: f64,
    config: &FusionConfig,
) -> Result<(), RejectionReason> {
    let k = config.linearization_sigma_multiplier;
    let margin_m = if k.is_finite() && k >= 0.0 {
        k * horizontal_1sigma_m
    } else {
        f64::INFINITY
    };
    let distance_plus_ksigma_m = offset_m + margin_m;
    if distance_plus_ksigma_m.is_finite() && distance_plus_ksigma_m <= valid_radius_m {
        return Ok(());
    }
    Err(RejectionReason::LinearizationRegionExceeded {
        distance_plus_ksigma_m,
        valid_radius_m,
    })
}

/// Refuses a gradient that gives too little information. The ratio
/// `η = gᵀ P_pos g / R` compares the predicted signal variance with the
/// noise variance. It has no unit, so it does not depend on the magnetic
/// scale. A negative or non-finite floor fails closed.
pub(crate) fn check_gradient_information(
    g_plane: &Vector3<f64>,
    p_pos: &Matrix3<f64>,
    variance_t2: f64,
    config: &FusionConfig,
) -> Result<(), RejectionReason> {
    let information_ratio = g_plane.dot(&(p_pos * g_plane)) / variance_t2;
    let floor = config.min_information_ratio;
    let floor_valid = floor.is_finite() && floor >= 0.0;
    if floor_valid && information_ratio.is_finite() && information_ratio >= floor {
        return Ok(());
    }
    Err(RejectionReason::UninformativeGradient {
        information_ratio,
        floor,
    })
}

impl MeasurementModel<1> for MagneticAnomalyModel {
    /// Checks the linearization region first, then the information ratio.
    fn admissible(&self, x: &Vec6, p: &Mat6, config: &FusionConfig) -> Result<(), RejectionReason> {
        check_linearization_region(
            self.offset_from_lin(x).norm(),
            horizontal_1sigma_m(p),
            self.valid_radius_m,
            config,
        )?;
        let p_pos = p.fixed_view::<3, 3>(0, 0).into_owned();
        check_gradient_information(&self.g_plane, &p_pos, self.variance_t2, config)
    }

    fn innovation(&self, x: &Vec6) -> SVector<f64, 1> {
        let predicted = self.map_t_at_lin + self.g_plane.dot(&self.offset_from_lin(x));
        SVector::<f64, 1>::new(self.measured_t - predicted)
    }

    fn jacobian(&self, _x: &Vec6) -> Option<SMatrix<f64, 1, 6>> {
        let g = self.g_plane;
        Some(SMatrix::<f64, 1, 6>::new(g.x, g.y, g.z, 0.0, 0.0, 0.0))
    }

    fn noise(&self) -> SMatrix<f64, 1, 1> {
        SMatrix::<f64, 1, 1>::new(self.variance_t2)
    }
}
