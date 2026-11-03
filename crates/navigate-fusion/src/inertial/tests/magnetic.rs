//! The magnetic map bias belongs to one source and one map revision.

use navigate_geodesy::{LocalTangentPlane, NedOffset};

use super::*;
use crate::filter::FogmBias;
use crate::inertial::nominal::MAG_BIAS;
use crate::observation::{Observation, ObservationValue};
use crate::rejection::RejectionReason;

const GRADIENT: [f64; 3] = [3e-11, -2e-11, 0.0];
const MAP_T: f64 = 5e-8;
const SIGMA_T: f64 = 1e-8;

fn config() -> InertialConfig {
    InertialConfig {
        mag_bias: FogmBias::new(3_600.0, SIGMA_T),
        ..InertialConfig::default()
    }
}

/// A measurement at rest at the origin with the map bias `bias_t`.
fn anomaly(sequence: u32, at_ms: u64, revision: u64, bias_t: f64) -> Observation {
    let plane = LocalTangentPlane::new(earth_origin()).expect("plausible");
    let value = ObservationValue::MagneticAnomaly {
        measured_t: MAP_T + bias_t,
        map_t_at_lin: MAP_T,
        gradient_ned_t_per_m: GRADIENT,
        linearization_point: plane.from_ned(&NedOffset::new(0.0, 0.0, 0.0)),
        valid_radius_m: 2_000.0,
        variance_t2: 1e-18,
    };
    Observation::new(
        stamp(7, sequence, at_ms),
        value,
        SourceComposition::of(SensorClass::MagneticAnomaly),
    )
    .with_map_revision(revision)
}

#[test]
fn a_new_map_revision_starts_a_new_bias_and_a_constant_revision_converges() {
    let mut filter = surface_filter(config(), PlanetaryBody::earth_wgs84(), earth_origin());
    let rates = rest_rates(&filter);
    let first_bias = 4e-9;
    let second_bias = -6e-9;
    let propagate = |filter: &mut InertialNavigationFilter, k: u32| {
        let at_ms = u64::from(k) * 1_000;
        assert_accepted(filter.ingest_imu(&imu(k, at_ms, rates, 1.0), t(at_ms)));
    };
    let measure = |filter: &mut InertialNavigationFilter, k: u32, revision: u64, bias: f64| {
        let at_ms = u64::from(k) * 1_000;
        filter.ingest(&anomaly(k, at_ms, revision, bias), t(at_ms))
    };
    let step = |filter: &mut InertialNavigationFilter, k: u32, revision: u64, bias: f64| {
        propagate(filter, k);
        measure(filter, k, revision, bias)
    };
    for k in 1..=30 {
        assert_accepted(step(&mut filter, k, 1, first_bias));
    }
    let converged = core(&filter).nominal.b_mag;
    let converged_variance = core(&filter).p[(MAG_BIAS, MAG_BIAS)];
    assert!(
        (converged - first_bias).abs() < 0.1 * first_bias,
        "{converged}"
    );
    assert!(converged_variance < 0.05 * SIGMA_T * SIGMA_T);
    // An outlier from the new revision fails the gate. The refusal does
    // not reset the bias or the active revision.
    propagate(&mut filter, 31);
    let before_outlier = core(&filter).clone();
    let outlier = measure(&mut filter, 31, 2, 5e-7);
    assert!(matches!(
        outlier,
        IngestOutcome::Rejected(RejectionReason::InnovationGate { .. })
    ));
    assert_eq!(core(&filter), &before_outlier);
    assert_eq!(core(&filter).mag_key.map(|key| key.1), Some(Some(1)));
    // The first admitted value of the new revision starts from zero with
    // the stationary variance, so it moves toward the new bias at once.
    assert_accepted(step(&mut filter, 32, 2, second_bias));
    let after_switch = core(&filter).nominal.b_mag;
    assert!(after_switch < 0.5 * second_bias, "{after_switch}");
    assert!(core(&filter).p[(MAG_BIAS, MAG_BIAS)] > converged_variance);
    for k in 33..=60 {
        assert_accepted(step(&mut filter, k, 2, second_bias));
    }
    let reconverged = core(&filter).nominal.b_mag;
    assert!(
        (reconverged - second_bias).abs() < 0.1 * second_bias.abs(),
        "{reconverged}"
    );
    // The position stays at the origin: the bias, not the position,
    // absorbed the offset.
    assert!(
        core(&filter).nominal.p.norm() < 1.0,
        "{}",
        core(&filter).nominal.p
    );
}
