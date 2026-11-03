use navigate_contract::{
    ClockDomainId, Ellipsoid, FaultDetection, GeodeticPosition, IntegrityAssessment,
    MonotonicNanos, NavigationSolution, NedVelocity, Redundancy, SensorClass, SolutionQuality,
    SolutionStamp, SourceComposition, SourceEpoch, SymmetricCov3, WrappingSequence,
};
use navigate_geodesy::{GeoidUndulation, ZeroUndulation};

use super::OrthometricTerrain;
use crate::{
    AlertSeverity, ClearanceConfig, ClearanceUnavailable, OrthometricElevation, TerrainElevation,
    assess_clearance,
};

const CLOCK: ClockDomainId = ClockDomainId::new(0);

/// A digital elevation model of Mars at 100 m above the areoid in the
/// northern hemisphere, with no coverage in the south.
struct MarsDem;

impl OrthometricElevation for MarsDem {
    fn orthometric_height_m(&self, position: &GeodeticPosition) -> Option<f64> {
        (position.latitude_rad >= 0.0).then_some(100.0)
    }

    fn ellipsoid(&self) -> Ellipsoid {
        Ellipsoid::MARS_IAU
    }
}

/// An areoid 25 m above the Mars ellipsoid, with coverage only west of
/// the prime meridian and at it.
struct Areoid;

impl GeoidUndulation for Areoid {
    fn ellipsoid(&self) -> Ellipsoid {
        Ellipsoid::MARS_IAU
    }

    fn undulation_m(&self, _latitude_rad: f64, longitude_rad: f64) -> Option<f64> {
        (longitude_rad <= 0.0).then_some(25.0)
    }
}

fn mars_solution(latitude_rad: f64, longitude_rad: f64, altitude_m: f64) -> NavigationSolution {
    let composition = SourceComposition::of(SensorClass::Inertial);
    NavigationSolution::new(
        SolutionStamp::new(
            SourceEpoch::new(0),
            WrappingSequence::new(0),
            MonotonicNanos::from_nanos(0),
            CLOCK,
        ),
        GeodeticPosition::new(latitude_rad, longitude_rad, altitude_m),
        NedVelocity::new(0.0, 0.0, 0.0),
        SymmetricCov3::from_diagonal(1.0, 1.0, 4.0),
        SymmetricCov3::from_diagonal(0.1, 0.1, 0.1),
        IntegrityAssessment::new(
            SolutionQuality::Good,
            composition,
            Redundancy::None,
            1.0,
            2.0,
            FaultDetection::Unavailable,
        ),
        composition,
        Ellipsoid::MARS_IAU,
    )
}

fn assess(
    solution: &NavigationSolution,
    terrain: &dyn TerrainElevation,
) -> Result<crate::ClearanceAssessment, ClearanceUnavailable> {
    let now = MonotonicNanos::from_nanos(0);
    assess_clearance(
        solution,
        now,
        CLOCK,
        Some(terrain),
        &ClearanceConfig::default(),
    )
}

#[test]
fn the_undulation_turns_an_orthometric_height_into_an_ellipsoidal_height() {
    let terrain = OrthometricTerrain::new(&MarsDem, &Areoid);
    let position = GeodeticPosition::new(0.3, -0.2, 0.0);
    assert_eq!(terrain.elevation_m(&position), Some(125.0));
    assert_eq!(TerrainElevation::ellipsoid(&terrain), Ellipsoid::MARS_IAU);
    // An ellipsoidal altitude of 200 m is 75 m above the terrain, not
    // 100 m: the areoid lifts the terrain by 25 m.
    let assessment = assess(&mars_solution(0.3, -0.2, 200.0), &terrain).expect("assessed");
    assert!((assessment.height_above_terrain_m - 75.0).abs() < 1e-9);
    assert_eq!(assessment.severity, Some(AlertSeverity::Caution));
}

#[test]
fn a_missing_undulation_fails_closed() {
    let terrain = OrthometricTerrain::new(&MarsDem, &Areoid);
    // The DEM covers the position, and the areoid does not.
    assert_eq!(
        terrain.elevation_m(&GeodeticPosition::new(0.3, 0.2, 0.0)),
        None
    );
    assert_eq!(
        assess(&mars_solution(0.3, 0.2, 200.0), &terrain),
        Err(ClearanceUnavailable::NoCoverage)
    );
    // The areoid covers the position, and the DEM does not.
    assert_eq!(
        assess(&mars_solution(-0.3, -0.2, 200.0), &terrain),
        Err(ClearanceUnavailable::NoCoverage)
    );
}

#[test]
fn an_undulation_model_on_another_ellipsoid_fails_closed() {
    let earth_geoid = ZeroUndulation(Ellipsoid::WGS84);
    let terrain = OrthometricTerrain::new(&MarsDem, &earth_geoid);
    assert_eq!(
        terrain.elevation_m(&GeodeticPosition::new(0.3, -0.2, 0.0)),
        None
    );
    assert_eq!(
        assess(&mars_solution(0.3, -0.2, 200.0), &terrain),
        Err(ClearanceUnavailable::NoCoverage)
    );
}
