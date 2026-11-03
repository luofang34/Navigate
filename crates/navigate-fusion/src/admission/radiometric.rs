//! Plausibility of the radiometric and line-of-sight values
//! (ADR-0018).

use super::{check_scalar_variance, is_unit_vector, require};
use crate::observation::{LineOfSightReference, ObservationValue};
use crate::rejection::RejectionReason;

fn finite(values: &[f64]) -> bool {
    values.iter().all(|v| v.is_finite())
}

/// The plausibility of a radiometric or line-of-sight value. A value of
/// another kind is refused by name.
pub(super) fn check(value: &ObservationValue) -> Result<(), RejectionReason> {
    match value {
        ObservationValue::Pseudorange {
            emitter_position_m,
            pseudorange_m,
            sagnac_and_media_correction_m,
            variance_m2,
        } => {
            require(
                finite(emitter_position_m)
                    && pseudorange_m.is_finite()
                    && sagnac_and_media_correction_m.is_finite(),
            )?;
            check_scalar_variance(*variance_m2)
        }
        ObservationValue::PseudorangeRate {
            emitter_position_m,
            emitter_velocity_mps,
            pseudorange_rate_mps,
            variance_m2_per_s2,
        } => {
            require(
                finite(emitter_position_m)
                    && finite(emitter_velocity_mps)
                    && pseudorange_rate_mps.is_finite(),
            )?;
            check_scalar_variance(*variance_m2_per_s2)
        }
        ObservationValue::TwoWayRange { .. } => check_two_way_range(value),
        ObservationValue::TwoWayRangeRate {
            tx_station_pos_m,
            tx_station_vel_mps,
            rx_station_pos_m,
            rx_station_vel_mps,
            round_trip_range_rate_mps,
            variance_m2_per_s2,
        } => {
            require(
                finite(tx_station_pos_m)
                    && finite(tx_station_vel_mps)
                    && finite(rx_station_pos_m)
                    && finite(rx_station_vel_mps)
                    && round_trip_range_rate_mps.is_finite(),
            )?;
            check_scalar_variance(*variance_m2_per_s2)
        }
        ObservationValue::LineOfSight {
            measured_body_unit,
            reference,
            angular_variance_rad2,
        } => {
            let reference_plausible = match reference {
                LineOfSightReference::InertialDirection {
                    direction_in_frame_unit,
                } => is_unit_vector(direction_in_frame_unit),
                LineOfSightReference::TargetPosition {
                    position_in_frame_m,
                } => finite(position_in_frame_m),
            };
            require(is_unit_vector(measured_body_unit) && reference_plausible)?;
            check_scalar_variance(*angular_variance_rad2)
        }
        _ => Err(RejectionReason::UnsupportedMeasurement { kind: value.kind() }),
    }
}

fn check_two_way_range(value: &ObservationValue) -> Result<(), RejectionReason> {
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
    let body = gravitation_gm_m3_s2
        .as_ref()
        .is_none_or(|(gm, position)| gm.is_finite() && *gm > 0.0 && finite(position));
    require(
        finite(tx_station_pos_m)
            && finite(rx_station_pos_m)
            && body
            && round_trip_range_m.is_finite()
            && *round_trip_range_m >= 0.0
            && turnaround_delay_m.is_finite(),
    )?;
    check_scalar_variance(*variance_m2)
}
