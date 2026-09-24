//! Measurement models of the six-state core (ADR-0008).

use nalgebra::{Matrix3, SMatrix, SVector, Vector3};

use super::{MeasurementModel, Vec6};

/// Range below which the range Jacobian direction is undefined, in meters.
const MIN_RANGE_M: f64 = 1.0;

/// A direct measurement of the position or the velocity block.
#[derive(Debug, Clone, Copy)]
pub(crate) struct LinearBlock {
    offset: usize,
    z: Vector3<f64>,
    r: Matrix3<f64>,
}

impl LinearBlock {
    /// Position rows: `H = [I₃ 0₃]`.
    pub(crate) const fn position(z: Vector3<f64>, r: Matrix3<f64>) -> Self {
        Self { offset: 0, z, r }
    }

    /// Velocity rows: `H = [0₃ I₃]`.
    pub(crate) const fn velocity(z: Vector3<f64>, r: Matrix3<f64>) -> Self {
        Self { offset: 3, z, r }
    }

    fn h(&self) -> SMatrix<f64, 3, 6> {
        let mut h = SMatrix::<f64, 3, 6>::zeros();
        for i in 0..3 {
            h[(i, i + self.offset)] = 1.0;
        }
        h
    }
}

impl MeasurementModel<3> for LinearBlock {
    fn innovation(&self, x: &Vec6) -> SVector<f64, 3> {
        self.z - self.h() * x
    }

    fn jacobian(&self, _x: &Vec6) -> Option<SMatrix<f64, 3, 6>> {
        Some(self.h())
    }

    fn noise(&self) -> SMatrix<f64, 3, 3> {
        self.r
    }
}

/// Slant range to a transmitter at a known local NED position, such as a DME.
///
/// `h(x) = |p − s|`. The local plane is a tangent plane of the WGS84
/// ellipsoid, so the Euclidean distance in it is the slant range.
#[derive(Debug, Clone, Copy)]
pub(crate) struct RangeModel {
    station_ned: Vector3<f64>,
    range_m: f64,
    variance_m2: f64,
}

impl RangeModel {
    pub(crate) const fn new(station_ned: Vector3<f64>, range_m: f64, variance_m2: f64) -> Self {
        Self {
            station_ned,
            range_m,
            variance_m2,
        }
    }

    fn line_of_sight(&self, x: &Vec6) -> Vector3<f64> {
        Vector3::new(x[0], x[1], x[2]) - self.station_ned
    }
}

impl MeasurementModel<1> for RangeModel {
    fn innovation(&self, x: &Vec6) -> SVector<f64, 1> {
        SVector::<f64, 1>::new(self.range_m - self.line_of_sight(x).norm())
    }

    fn jacobian(&self, x: &Vec6) -> Option<SMatrix<f64, 1, 6>> {
        let los = self.line_of_sight(x);
        let range = los.norm();
        if range < MIN_RANGE_M {
            return None;
        }
        let unit = los / range;
        Some(SMatrix::<f64, 1, 6>::new(
            unit.x, unit.y, unit.z, 0.0, 0.0, 0.0,
        ))
    }

    fn noise(&self) -> SMatrix<f64, 1, 1> {
        SMatrix::<f64, 1, 1>::new(self.variance_m2)
    }
}

#[cfg(test)]
mod tests;
