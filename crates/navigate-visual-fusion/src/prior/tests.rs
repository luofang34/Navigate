#![allow(clippy::expect_used)]
use super::*;
use navigate_contract::{
    FaultDetection, GeodeticPosition, IntegrityAssessment, NedVelocity, Redundancy, SensorClass,
    SolutionQuality, SolutionStamp, SourceComposition, SourceEpoch, WrappingSequence,
};

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
        orientation: UnitQuaternion::identity(),
        attitude_sigma_rad: 0.02,
    }
}

fn frame() -> LocalFrame {
    LocalFrame::anchor_mercator(47.0, 8.0).expect("anchor")
}

#[test]
fn the_prior_moves_with_velocity_and_grows_with_time() {
    let at_solution = search_prior(&solution(50.0), frame(), capture(1.0)).expect("prior");
    let later = search_prior(&solution(50.0), frame(), capture(3.0)).expect("prior");
    let moved = later.center.position - at_solution.center.position;
    assert!((moved.y - 100.0).abs() < 1e-6, "north 50 m/s for 2 s");
    assert!(moved.x.abs() < 1e-6);
    // NED north variance 100 becomes frame y; east variance 400 becomes frame x.
    assert!((at_solution.position_covariance_m2[(1, 1)] - 100.0).abs() < 1e-9);
    assert!((at_solution.position_covariance_m2[(0, 0)] - 400.0).abs() < 1e-9);
    assert!((later.position_covariance_m2[(1, 1)] - (100.0 + 4.0 * 4.0)).abs() < 1e-9);
}

#[test]
fn a_foreign_clock_is_refused() {
    let mut c = capture(1.0);
    c.clock = ClockDomainId::new(99);
    assert!(matches!(
        search_prior(&solution(0.0), frame(), c),
        Err(VisualFusionError::ClockDomainMismatch)
    ));
}
