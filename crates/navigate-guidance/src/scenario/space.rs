//! Space scenarios: solutions in Cartesian frames, the time of a step,
//! and a fourth-order Runge–Kutta propagator.

use nalgebra::Vector3;
use navigate_contract::{
    CartesianPosition, CartesianSolution, CartesianVelocity, FaultDetection, FrameId,
    MonotonicNanos, Redundancy, SensorClass, SolutionQuality, SolutionStamp, SourceComposition,
    SourceEpoch, SpaceIntegrityAssessment, SpaceUncertainty, SymmetricCov3, WrappingSequence,
};
use navigate_trajectory::StateVector;

use super::CLOCK;

/// A `Good` inertial-only solution of `state` in `frame`, stamped
/// `solved_at` on [`CLOCK`], with the one-sigma `sigma_m` on every
/// principal axis.
pub(crate) fn space_solution(
    frame: FrameId,
    state: &StateVector,
    solved_at: MonotonicNanos,
    sigma_m: f64,
) -> CartesianSolution {
    let composition = SourceComposition::of(SensorClass::Inertial);
    let (p, v) = (state.position_m, state.velocity_mps);
    CartesianSolution::new(
        SolutionStamp::new(
            SourceEpoch::new(1),
            WrappingSequence::new(1),
            solved_at,
            CLOCK,
        ),
        frame,
        CartesianPosition::new(p.x, p.y, p.z),
        CartesianVelocity::new(v.x, v.y, v.z),
        SymmetricCov3::from_diagonal(sigma_m * sigma_m, sigma_m * sigma_m, sigma_m * sigma_m),
        SymmetricCov3::from_diagonal(0.01, 0.01, 0.01),
        SpaceIntegrityAssessment::new(
            SolutionQuality::Good,
            composition,
            Redundancy::None,
            SpaceUncertainty::PrincipalAxes {
                axes_1sigma_m: [sigma_m; 3],
                max_axis_1sigma_m: sigma_m,
            },
            FaultDetection::Unavailable,
        ),
        composition,
    )
}

/// The monotonic time `millis` milliseconds after the origin.
pub(crate) fn at_millis(millis: u64) -> MonotonicNanos {
    MonotonicNanos::from_nanos(millis * 1_000_000)
}

/// The point-mass gravity of `gm` at `p`.
pub(crate) fn point_mass(gm: f64, p: &Vector3<f64>) -> Vector3<f64> {
    -p * (gm / p.norm().powi(3))
}

/// One fourth-order Runge–Kutta step of the acceleration `acc(p, v)`.
pub(crate) fn rk4(
    state: &StateVector,
    dt: f64,
    acc: impl Fn(&Vector3<f64>, &Vector3<f64>) -> Vector3<f64>,
) -> StateVector {
    let (p, v) = (state.position_m, state.velocity_mps);
    let h = dt / 2.0;
    let k1 = (v, acc(&p, &v));
    let k2 = (v + k1.1 * h, acc(&(p + k1.0 * h), &(v + k1.1 * h)));
    let k3 = (v + k2.1 * h, acc(&(p + k2.0 * h), &(v + k2.1 * h)));
    let k4 = (v + k3.1 * dt, acc(&(p + k3.0 * dt), &(v + k3.1 * dt)));
    StateVector::new(
        p + (k1.0 + k2.0 * 2.0 + k3.0 * 2.0 + k4.0) * (dt / 6.0),
        v + (k1.1 + k2.1 * 2.0 + k3.1 * 2.0 + k4.1) * (dt / 6.0),
    )
}
