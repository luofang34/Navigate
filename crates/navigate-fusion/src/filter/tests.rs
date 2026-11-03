use nalgebra::{Matrix3, SMatrix, SVector, Vector3};

use super::{
    ConstantVelocityPropagator, FogmBias, KalmanState, LinearBlock, Mat6, MeasurementModel,
    Propagator, Vec6, horizontal_1sigma_m, is_positive_definite, vertical_1sigma_m,
};
use crate::config::FusionConfig;

/// A direct measurement of a one-state bias.
struct BiasReading {
    z: f64,
    variance: f64,
}

impl MeasurementModel<1, 1> for BiasReading {
    fn innovation(&self, x: &SVector<f64, 1>) -> SVector<f64, 1> {
        SVector::<f64, 1>::new(self.z - x[0])
    }

    fn jacobian(&self, _x: &SVector<f64, 1>) -> Option<SMatrix<f64, 1, 1>> {
        Some(SMatrix::<f64, 1, 1>::identity())
    }

    fn noise(&self) -> SMatrix<f64, 1, 1> {
        SMatrix::<f64, 1, 1>::new(self.variance)
    }
}

#[test]
fn propagation_inflates_position_uncertainty() {
    let state = KalmanState::from_position_fix(&Matrix3::from_diagonal_element(4.0), 1.0);
    let advanced = state.propagated(2.0, 0.5);
    // P_pp' = P_pp + dt²·P_vv + q·dt³/3 = 4 + 4 + 4/3.
    assert!((advanced.p[(0, 0)] - (4.0 + 4.0 + 4.0 / 3.0)).abs() < 1e-9);
    assert!(advanced.p[(0, 0)] > state.p[(0, 0)]);
}

#[test]
fn update_moves_the_state_toward_the_measurement() {
    let state = KalmanState::from_position_fix(&Matrix3::from_diagonal_element(100.0), 10.0);
    let z = Vector3::new(10.0, 0.0, 0.0);
    let r = Matrix3::identity();
    let prepared = state
        .prepare_update(&LinearBlock::position(z, r))
        .expect("invertible innovation covariance");
    let updated = state.apply_update(&prepared);
    assert!((updated.x[0] - 10.0 * 100.0 / 101.0).abs() < 1e-9);
    assert!(updated.p[(0, 0)] < state.p[(0, 0)]);
    // Joseph form keeps the covariance symmetric.
    assert!((updated.p - updated.p.transpose()).norm() < 1e-12);
}

#[test]
fn definiteness_screen_refuses_impossible_and_singular_covariances() {
    let overcorrelated = Matrix3::new(25.0, 100.0, 0.0, 100.0, 25.0, 0.0, 0.0, 0.0, 25.0);
    assert!(!is_positive_definite(&overcorrelated));
    assert!(is_positive_definite(&Matrix3::identity()));
    assert!(!is_positive_definite(&Matrix3::from_diagonal_element(-1.0)));
    // Zero and rank-deficient covariances claim a direction of exact
    // certainty; the definite screen refuses them.
    assert!(!is_positive_definite(&Matrix3::zeros()));
    assert!(!is_positive_definite(&Matrix3::new(
        1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0
    )));
}

#[test]
fn singular_innovation_covariance_is_reported_not_inverted() {
    let degenerate = KalmanState {
        x: Vec6::zeros(),
        p: Mat6::zeros(),
    };
    let prepared =
        degenerate.prepare_update(&LinearBlock::position(Vector3::zeros(), Matrix3::zeros()));
    assert!(prepared.is_none());
}

#[test]
fn sigma_extraction_reads_the_horizontal_block() {
    let mut p = Mat6::zeros();
    p[(0, 0)] = 9.0;
    p[(1, 1)] = 4.0;
    p[(2, 2)] = 16.0;
    assert!((horizontal_1sigma_m(&p) - 3.0).abs() < 1e-12);
    assert!((vertical_1sigma_m(&p) - 4.0).abs() < 1e-12);
    // Correlation rotates the ellipse: the major axis exceeds both
    // diagonal sigmas.
    p[(0, 1)] = 5.0;
    p[(1, 0)] = 5.0;
    assert!(horizontal_1sigma_m(&p) > 3.0);
}

#[test]
fn the_six_state_shortcut_delegates_to_the_constant_velocity_propagator() {
    let state = KalmanState::from_position_fix(&Matrix3::from_diagonal_element(4.0), 1.0);
    let direct = ConstantVelocityPropagator { accel_psd: 0.5 }.propagate(&state, 2.0);
    assert_eq!(state.propagated(2.0, 0.5), direct);
    assert_eq!(state.propagated(0.0, 0.5), state);
    assert_eq!(state.propagated(-1.0, 0.5), state);
}

#[test]
fn the_core_updates_a_one_state_fogm_bias() {
    let bias = FogmBias::new(60.0, 3e-9);
    let mut state = KalmanState::<1> {
        x: SVector::<f64, 1>::new(0.0),
        p: SMatrix::<f64, 1, 1>::new(9e-18),
    };
    let reading = BiasReading {
        z: 2e-9,
        variance: 1e-18,
    };
    assert!(
        reading
            .admissible(&state.x, &state.p, &FusionConfig::default())
            .is_ok()
    );
    for _ in 0..50 {
        let (phi, _) = bias.propagate(1.0, 1.0);
        let (mean, q) = bias.propagate(state.x[0], 1.0);
        state = KalmanState {
            x: SVector::<f64, 1>::new(mean),
            p: SMatrix::<f64, 1, 1>::new(phi * phi * state.p[(0, 0)] + q),
        };
        let prepared = state.prepare_update(&reading).expect("positive variance");
        state = state.apply_update(&prepared);
    }
    // The bias settles near the readings, and the variance stays below
    // the reading variance and above zero.
    assert!((state.x[0] - 2e-9).abs() < 2e-10, "{}", state.x[0]);
    assert!(state.p[(0, 0)] > 0.0 && state.p[(0, 0)] < 1e-18);
}
