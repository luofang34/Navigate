//! One-way and two-way radiometric range and range rate (ADR-0018).
//!
//! Each model is a scalar in the Cartesian frame of the regime. With `u`
//! the unit vector from a station or an emitter to the vehicle at the
//! distance `d`, a range has the row `uᵀ` on the position error. A range
//! rate `u·w` with the relative velocity `w` has the row `uᵀ` on the
//! velocity error and `(w − u·(u·w))ᵀ/d` on the position error. A
//! one-way model adds the receiver clock bias or drift with the row `1`.
//! A two-way model sums the uplink and the downlink and has no clock
//! term.

use nalgebra::{Const, DimMin, SMatrix, Vector3};

use super::{MIN_RANGE_M, scalar};
use crate::float_math::ln;
use crate::inertial::clock::SPEED_OF_LIGHT_MPS;
use crate::inertial::config::InertialConfig;
use crate::inertial::estimate::{InertialCore, Linearized, Mode};
use crate::inertial::nominal::{POS, VEL};
use crate::observation::ObservationValue;
use crate::rejection::RejectionReason;

/// One leg from a station or an emitter to the vehicle.
struct Leg {
    unit: Vector3<f64>,
    distance: f64,
}

impl Leg {
    /// The leg from `station` to the vehicle at `p`. A vehicle closer
    /// than [`MIN_RANGE_M`] has no direction.
    fn new(p: &Vector3<f64>, station: &[f64; 3]) -> Result<Self, RejectionReason> {
        let offset = p - Vector3::from(*station);
        let distance = offset.norm();
        if distance < MIN_RANGE_M {
            return Err(RejectionReason::ImplausibleCovariance);
        }
        Ok(Self {
            unit: offset / distance,
            distance,
        })
    }

    /// The range rate of the relative velocity `w`, with its rows on the
    /// position error and on the velocity error.
    fn rate(&self, w: &Vector3<f64>) -> (f64, Vector3<f64>, Vector3<f64>) {
        let rate = self.unit.dot(w);
        let position_row = (w - self.unit * rate) / self.distance;
        (rate, position_row, self.unit)
    }
}

/// The general-relativistic delay in meters of one leg between `a` and
/// `b` past a body with the gravitational parameter `gm` at `body`:
/// `2μ/c²·ln((r₁ + r₂ + r₁₂)/(r₁ + r₂ − r₁₂))`. The second value is the
/// gradient of the delay with respect to `a`. A leg through the body
/// center has no finite delay.
pub(crate) fn shapiro_delay(
    a: &Vector3<f64>,
    b: &Vector3<f64>,
    gm: f64,
    body: &Vector3<f64>,
) -> Result<(f64, Vector3<f64>), RejectionReason> {
    let (to_a, to_b, chord) = (a - body, b - body, a - b);
    let (r1, r2, r12) = (to_a.norm(), to_b.norm(), chord.norm());
    let (sum, difference) = (r1 + r2 + r12, r1 + r2 - r12);
    if !(difference > 0.0 && r1 > 0.0 && r12 > 0.0) {
        return Err(RejectionReason::NonFiniteValue);
    }
    let k = 2.0 * gm / (SPEED_OF_LIGHT_MPS * SPEED_OF_LIGHT_MPS);
    let (d_r1, d_r12) = (to_a / r1, chord / r12);
    let gradient = k * ((d_r1 + d_r12) / sum - (d_r1 - d_r12) / difference);
    Ok((k * ln(sum / difference), gradient))
}

/// Applies a radiometric value to `core`.
pub(super) fn apply<const N: usize>(
    core: &mut InertialCore<N>,
    value: &ObservationValue,
    config: &InertialConfig,
    mode: Mode,
) -> Result<f64, RejectionReason>
where
    Const<N>: DimMin<Const<N>, Output = Const<N>>,
{
    let model = linearize(core, value, config)?;
    core.update(&model, config.fusion.scalar_gate_chi2, mode)
}

/// The radiometric value linearized at the nominal state of `core`.
pub(crate) fn linearize<const N: usize>(
    core: &InertialCore<N>,
    value: &ObservationValue,
    config: &InertialConfig,
) -> Result<Linearized<1, N>, RejectionReason>
where
    Const<N>: DimMin<Const<N>, Output = Const<N>>,
{
    let clock = core.layout.clock().filter(|_| config.clock_model.is_some());
    let model = match value {
        ObservationValue::Pseudorange { .. } | ObservationValue::PseudorangeRate { .. }
            if clock.is_none() =>
        {
            return Err(RejectionReason::ClockStateDisabled { kind: value.kind() });
        }
        ObservationValue::Pseudorange {
            emitter_position_m,
            pseudorange_m,
            sagnac_and_media_correction_m,
            variance_m2,
        } => {
            let leg = Leg::new(&core.nominal.p, emitter_position_m)?;
            let predicted =
                leg.distance + core.nominal.clock_bias_m + sagnac_and_media_correction_m;
            let mut h = SMatrix::<f64, 1, N>::zeros();
            h.fixed_view_mut::<1, 3>(0, POS)
                .copy_from(&leg.unit.transpose());
            if let Some(index) = clock {
                h[index] = 1.0;
            }
            scalar(pseudorange_m - predicted, h, *variance_m2)
        }
        ObservationValue::PseudorangeRate {
            emitter_position_m,
            emitter_velocity_mps,
            pseudorange_rate_mps,
            variance_m2_per_s2,
        } => {
            let leg = Leg::new(&core.nominal.p, emitter_position_m)?;
            let w = core.nominal.v - Vector3::from(*emitter_velocity_mps);
            let (rate, position_row, velocity_row) = leg.rate(&w);
            let mut h = rate_row(&position_row, &velocity_row);
            if let Some(index) = clock {
                h[index + 1] = 1.0;
            }
            let predicted = rate + core.nominal.clock_drift_mps;
            scalar(pseudorange_rate_mps - predicted, h, *variance_m2_per_s2)
        }
        ObservationValue::TwoWayRange { .. } => two_way_range(core, value)?,
        ObservationValue::TwoWayRangeRate { .. } => two_way_range_rate(core, value)?,
        _ => return Err(RejectionReason::UnsupportedMeasurement { kind: value.kind() }),
    };
    Ok(model)
}

/// A row with `position_row` on the position error and `velocity_row`
/// on the velocity error.
fn rate_row<const N: usize>(
    position_row: &Vector3<f64>,
    velocity_row: &Vector3<f64>,
) -> SMatrix<f64, 1, N> {
    let mut h = SMatrix::<f64, 1, N>::zeros();
    h.fixed_view_mut::<1, 3>(0, POS)
        .copy_from(&position_row.transpose());
    h.fixed_view_mut::<1, 3>(0, VEL)
        .copy_from(&velocity_row.transpose());
    h
}

/// The two-way range: the uplink and the downlink distances, the delay
/// of each leg past the body of `gravitation_gm_m3_s2`, and the
/// turnaround delay.
fn two_way_range<const N: usize>(
    core: &InertialCore<N>,
    value: &ObservationValue,
) -> Result<Linearized<1, N>, RejectionReason>
where
    Const<N>: DimMin<Const<N>, Output = Const<N>>,
{
    let ObservationValue::TwoWayRange {
        tx_station_pos_m,
        rx_station_pos_m,
        gravitation_gm_m3_s2,
        round_trip_range_m,
        turnaround_delay_m,
        variance_m2,
    } = value
    else {
        return Err(RejectionReason::UnsupportedMeasurement { kind: value.kind() });
    };
    let p = core.nominal.p;
    let up = Leg::new(&p, tx_station_pos_m)?;
    let down = Leg::new(&p, rx_station_pos_m)?;
    let mut predicted = up.distance + down.distance + turnaround_delay_m;
    let mut row = up.unit + down.unit;
    if let Some((gm, body)) = gravitation_gm_m3_s2 {
        let body = Vector3::from(*body);
        for station in [tx_station_pos_m, rx_station_pos_m] {
            let (delay, gradient) = shapiro_delay(&p, &Vector3::from(*station), *gm, &body)?;
            predicted += delay;
            row += gradient;
        }
    }
    let mut h = SMatrix::<f64, 1, N>::zeros();
    h.fixed_view_mut::<1, 3>(0, POS).copy_from(&row.transpose());
    Ok(scalar(round_trip_range_m - predicted, h, *variance_m2))
}

/// The two-way range rate: the sum of the uplink and the downlink range
/// rates.
fn two_way_range_rate<const N: usize>(
    core: &InertialCore<N>,
    value: &ObservationValue,
) -> Result<Linearized<1, N>, RejectionReason>
where
    Const<N>: DimMin<Const<N>, Output = Const<N>>,
{
    let ObservationValue::TwoWayRangeRate {
        tx_station_pos_m,
        tx_station_vel_mps,
        rx_station_pos_m,
        rx_station_vel_mps,
        round_trip_range_rate_mps,
        variance_m2_per_s2,
    } = value
    else {
        return Err(RejectionReason::UnsupportedMeasurement { kind: value.kind() });
    };
    let (p, v) = (core.nominal.p, core.nominal.v);
    let up = Leg::new(&p, tx_station_pos_m)?;
    let down = Leg::new(&p, rx_station_pos_m)?;
    let (rate_up, p_up, v_up) = up.rate(&(v - Vector3::from(*tx_station_vel_mps)));
    let (rate_down, p_down, v_down) = down.rate(&(v - Vector3::from(*rx_station_vel_mps)));
    let h = rate_row(&(p_up + p_down), &(v_up + v_down));
    let residual = round_trip_range_rate_mps - (rate_up + rate_down);
    Ok(scalar(residual, h, *variance_m2_per_s2))
}
