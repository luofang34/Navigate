use super::*;
use navigate_contract::{
    FaultDetection, GeodeticPosition, IntegrityAssessment, NedVelocity, Redundancy, SensorClass,
    SolutionQuality, SolutionStamp, SourceComposition, SourceEpoch, WrappingSequence,
};

type TestResult = Result<(), Box<dyn std::error::Error>>;

const CLOCK: ClockDomainId = ClockDomainId::new(4);

fn solution(north_mps: f64) -> NavigationSolution {
    NavigationSolution::new(
        SolutionStamp::new(
            SourceEpoch::new(1),
            WrappingSequence::new(1),
            MonotonicNanos::from_nanos(1_000_000_000),
            CLOCK,
        ),
        GeodeticPosition::new(47.0_f64.to_radians(), 8.0_f64.to_radians(), 900.0),
        NedVelocity::new(north_mps, 0.0, 0.0),
        SymmetricCov3::from_diagonal(100.0, 400.0, 25.0),
        SymmetricCov3::from_diagonal(4.0, 4.0, 1.0),
        IntegrityAssessment::new(
            SolutionQuality::Good,
            SourceComposition::of(SensorClass::Gnss),
            Redundancy::None,
            10.0,
            5.0,
            FaultDetection::Unavailable,
        ),
        SourceComposition::of(SensorClass::Gnss),
    )
}

fn capture(at_s: f64) -> FrameCapture {
    FrameCapture {
        at: MonotonicNanos::from_nanos((at_s * 1e9) as u64),
        clock: CLOCK,
        time_error_bound: DurationNanos::from_nanos(0),
        orientation: UnitQuaternion::identity(),
        attitude_sigma_rad: 0.02,
    }
}

fn frame() -> Result<LocalFrame, navigate_visual::VisualError> {
    LocalFrame::anchor_mercator(47.0, 8.0)
}

#[test]
fn the_prior_moves_with_velocity_and_grows_with_time() -> TestResult {
    let at_solution = search_prior(&solution(50.0), frame()?, capture(1.0), budget())?;
    let later = search_prior(&solution(50.0), frame()?, capture(3.0), budget())?;
    let moved = later.center.position - at_solution.center.position;
    assert!((moved.y - 100.0).abs() < 1e-6, "north 50 m/s for 2 s");
    assert!(moved.x.abs() < 1e-6);
    // NED north variance 100 becomes frame y; east variance 400 becomes frame x.
    assert!((at_solution.position_covariance_m2[(1, 1)] - 100.0).abs() < 1e-9);
    assert!((at_solution.position_covariance_m2[(0, 0)] - 400.0).abs() < 1e-9);
    assert!(later.position_covariance_m2[(1, 1)] > (10.0_f64 + 2.0 * 2.0).powi(2));
    Ok(())
}

#[test]
fn a_foreign_clock_is_refused() -> TestResult {
    let mut c = capture(1.0);
    c.clock = ClockDomainId::new(99);
    assert!(matches!(
        search_prior(&solution(0.0), frame()?, c, budget()),
        Err(VisualFusionError::ClockDomainMismatch)
    ));
    Ok(())
}

fn budget() -> PropagationBudget {
    PropagationBudget {
        maximum_age: DurationNanos::from_millis(3000),
        solution_time_error_bound: DurationNanos::from_nanos(0),
        acceleration_rms_bound_mps2: Some(2.0),
    }
}

#[test]
fn unknown_correlation_is_bounded_for_both_projection_directions() {
    let a = Matrix3::new(4.0, 0.0, 0.0, 1.0, 3.0, 0.0, 0.5, -0.3, 2.0);
    let b = Matrix3::new(1.0, 0.5, 0.0, 0.0, 2.0, 0.0, -0.2, 0.0, 1.0);
    for dt in [-2.0, -0.1, 0.0, 0.1, 2.0] {
        let bound = correlated_sum(a * a.transpose(), b * b.transpose() * dt * dt);
        for rho in [-1.0, -0.5, 0.0, 0.5, 1.0] {
            let actual = a * a.transpose()
                + b * b.transpose() * dt * dt
                + (a * b.transpose() + b * a.transpose()) * rho * dt;
            assert!(
                (bound - actual)
                    .symmetric_eigen()
                    .eigenvalues
                    .iter()
                    .all(|v| *v >= -1e-10)
            );
        }
    }
}

#[test]
fn timestamp_uncertainty_widens_search_without_moving_capture_time() -> TestResult {
    let exact = search_prior(&solution(50.0), frame()?, capture(1.0), budget())?;
    let mut delayed = capture(1.0);
    delayed.time_error_bound = DurationNanos::from_millis(100);
    let uncertain = search_prior(&solution(50.0), frame()?, delayed, budget())?;
    assert_eq!(exact.center.position, uncertain.center.position);
    assert!(uncertain.position_covariance_m2[(1, 1)] >= 225.0);
    assert!(
        (uncertain.position_covariance_m2 - exact.position_covariance_m2)
            .symmetric_eigen()
            .eigenvalues
            .min()
            >= 0.0
    );
    Ok(())
}

#[test]
fn projection_refuses_unknown_process_error_excess_age_and_invalid_inputs() -> TestResult {
    let mut unknown = budget();
    unknown.acceleration_rms_bound_mps2 = None;
    assert!(search_prior(&solution(0.0), frame()?, capture(1.0), unknown).is_err());
    assert!(matches!(
        search_prior(&solution(0.0), frame()?, capture(5.0), budget()),
        Err(VisualFusionError::ProjectionWindow { .. })
    ));
    let mut sample = solution(0.0);
    sample.velocity_cov = SymmetricCov3::from_diagonal(-1.0, 1.0, 1.0);
    assert!(search_prior(&sample, frame()?, capture(1.0), budget()).is_err());
    sample = solution(f64::NAN);
    assert!(search_prior(&sample, frame()?, capture(1.0), budget()).is_err());
    let mut uncertain = capture(1.0);
    uncertain.time_error_bound = DurationNanos::from_nanos(u64::MAX);
    let mut policy = budget();
    policy.solution_time_error_bound = DurationNanos::from_nanos(1);
    assert!(search_prior(&solution(0.0), frame()?, uncertain, policy).is_err());
    Ok(())
}
