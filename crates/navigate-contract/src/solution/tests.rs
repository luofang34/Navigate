use super::{CartesianSolution, NavigationSolution};
use crate::{
    AttitudeQuaternion, CartesianPosition, CartesianVelocity, ClockDomainId, Ellipsoid,
    FaultDetection, FrameId, GeodeticPosition, IntegrityAssessment, MonotonicNanos, NedVelocity,
    Redundancy, SensorClass, SolutionQuality, SolutionStamp, SourceComposition, SourceEpoch,
    SpaceIntegrityAssessment, SpaceUncertainty, SymmetricCov3, WrappingSequence,
};

fn stamp() -> SolutionStamp {
    SolutionStamp::new(
        SourceEpoch::new(0),
        WrappingSequence::new(0),
        MonotonicNanos::from_nanos(1),
        ClockDomainId::new(0),
    )
}

fn integrity() -> IntegrityAssessment {
    IntegrityAssessment::new(
        SolutionQuality::Good,
        SourceComposition::of(SensorClass::Gnss),
        Redundancy::None,
        1.0,
        1.0,
        FaultDetection::Unavailable,
    )
}

fn space_integrity() -> SpaceIntegrityAssessment {
    SpaceIntegrityAssessment::new(
        SolutionQuality::Good,
        SourceComposition::of(SensorClass::Inertial),
        Redundancy::None,
        SpaceUncertainty::PrincipalAxes {
            axes_1sigma_m: [10.0, 10.0, 10.0],
            max_axis_1sigma_m: 10.0,
        },
        FaultDetection::Unavailable,
    )
}

fn solution_on(ellipsoid: Ellipsoid) -> NavigationSolution {
    NavigationSolution::new(
        stamp(),
        GeodeticPosition::new(0.1, 0.2, 30.0),
        NedVelocity::new(1.0, 2.0, 0.0),
        SymmetricCov3::from_diagonal(1.0, 1.0, 1.0),
        SymmetricCov3::from_diagonal(0.1, 0.1, 0.1),
        integrity(),
        SourceComposition::of(SensorClass::Gnss),
        ellipsoid,
    )
}

fn solution() -> NavigationSolution {
    solution_on(Ellipsoid::WGS84)
}

#[test]
fn a_new_solution_is_on_the_named_ellipsoid_without_attitude() {
    for ellipsoid in [Ellipsoid::WGS84, Ellipsoid::MOON_PA, Ellipsoid::MARS_IAU] {
        let solution = solution_on(ellipsoid);
        assert_eq!(solution.ellipsoid, ellipsoid);
        assert_eq!(solution.attitude, None);
        assert_eq!(solution.attitude_cov, None);
    }
}

#[test]
fn builders_set_the_ellipsoid_and_the_attitude_covariance() {
    let cov = SymmetricCov3::from_diagonal(1e-4, 1e-4, 4e-4);
    let solution = solution()
        .with_ellipsoid(Ellipsoid::MOON_ME)
        .with_attitude(AttitudeQuaternion::new(1.0, 0.0, 0.0, 0.0))
        .with_attitude_cov(cov);
    assert_eq!(solution.ellipsoid, Ellipsoid::MOON_ME);
    assert_eq!(solution.ellipsoid.frame, FrameId::MOON_ME);
    assert_eq!(solution.attitude_cov, Some(cov));
    assert!(solution.attitude.is_some());
}

#[test]
fn a_cartesian_solution_names_its_frame() {
    let solution = CartesianSolution::new(
        stamp(),
        FrameId::MOON_J2000,
        CartesianPosition::new(1_837_400.0, 0.0, 0.0),
        CartesianVelocity::new(0.0, 1_633.0, 0.0),
        SymmetricCov3::from_diagonal(100.0, 100.0, 100.0),
        SymmetricCov3::from_diagonal(0.01, 0.01, 0.01),
        space_integrity(),
        SourceComposition::of(SensorClass::Inertial),
    );
    assert_eq!(solution.frame, FrameId::MOON_J2000);
    assert_eq!(solution.integrity.uncertainty.max_axis_1sigma_m(), 10.0);
    assert_eq!(solution.integrity.protection_radius_m, None);
    assert_eq!(
        solution
            .integrity
            .with_protection_radius(42.0)
            .protection_radius_m,
        Some(42.0)
    );
    assert!(solution.frame.is_inertial());
    assert!(solution.position.is_finite());
    assert!(solution.velocity.is_finite());
    assert_eq!(solution.attitude, None);
    let cov = SymmetricCov3::from_diagonal(1e-6, 1e-6, 1e-6);
    let with_attitude = solution
        .with_attitude(AttitudeQuaternion::new(1.0, 0.0, 0.0, 0.0))
        .with_attitude_cov(cov);
    assert_eq!(with_attitude.attitude_cov, Some(cov));
}

#[test]
fn cartesian_values_report_non_finite_components() {
    assert!(!CartesianPosition::new(f64::NAN, 0.0, 0.0).is_finite());
    assert!(!CartesianPosition::new(0.0, 0.0, f64::INFINITY).is_finite());
    assert!(!CartesianVelocity::new(0.0, f64::NAN, 0.0).is_finite());
    assert!(CartesianVelocity::new(0.0, 0.0, 0.0).is_finite());
}

#[test]
fn every_space_uncertainty_form_states_its_largest_axis() {
    let rtn = SpaceUncertainty::OrbitalRtn {
        radial_1sigma_m: 3.0,
        along_track_1sigma_m: 12.0,
        cross_track_1sigma_m: 5.0,
        max_axis_1sigma_m: 12.5,
    };
    assert_eq!(rtn.max_axis_1sigma_m(), 12.5);
    let axes = SpaceUncertainty::PrincipalAxes {
        axes_1sigma_m: [9.0, 4.0, 1.0],
        max_axis_1sigma_m: 9.0,
    };
    assert_eq!(axes.max_axis_1sigma_m(), 9.0);
}
