//! The stated covariance of a terrain fix against its error, through the
//! real matcher: the demo camera (400 by 300, 67° across, matched at 320
//! by 240) over heights, view angles, textures, and the ways in which a
//! camera frame differs from the map imagery. The map elevation is exact
//! and its imagery states a millimetre, or the registration error it has,
//! so the stated covariance is that of the match.

mod mismatch;

use std::sync::Arc;

use nalgebra::{Matrix3, UnitQuaternion, Vector3};
use navigate_camera::{CameraMount, ImagerModel, Intrinsics};
use navigate_contract::{AttitudeQuaternion, GeodeticPosition, NedVelocity, SymmetricCov3};
use navigate_visual::CameraModel;

use super::flight::{CAMERA, CLOCK, Flight, RATE, at, suite};
use super::world::{CENTRE, GROUND_M, TODAY, Texture, World, WorldSpec, frame_at};
use crate::{
    Alignment, Navigator, ReferenceNavigator, SensorData, SensorInput, TerrainConfig,
    TerrainMapRequest,
};
use mismatch::Mismatch;

/// The camera of the demo: 400 by 300 with a field of 67° across.
fn camera() -> CameraModel {
    CameraModel {
        width: 400,
        height: 300,
        fx: 300.0,
        fy: 300.0,
        cx: 199.5,
        cy: 149.5,
    }
}

fn imager() -> ImagerModel {
    let k = camera();
    ImagerModel::pinhole(Intrinsics {
        width: k.width,
        height: k.height,
        fx: k.fx,
        fy: k.fy,
        cx: k.cx,
        cy: k.cy,
    })
}

fn world(texture: Texture, relief_m: f64, shift_m: [f64; 2], registration_mm: u32) -> World {
    World::build(&WorldSpec {
        accuracy_mm: Some((registration_mm, 1)),
        texture,
        relief_m,
        radius_m: 6_000.0,
        imagery_shift_m: shift_m,
        ..WorldSpec::default()
    })
}

fn mount(off_deg: f64) -> CameraMount {
    CameraMount::rotated_forward(0.0, -(90.0 - off_deg).to_radians(), 0.0)
}

fn hover(k: u32, set: u32, agl_m: f64) -> Flight {
    let shift = f64::from(set) * 500.0;
    let north = f64::from(k % 3) * 1_000.0 - 1_000.0 + shift;
    let east = f64::from(k / 3) * 1_000.0 - 1_000.0 + shift;
    Flight {
        start: [
            CENTRE[0] + north / 111_100.0,
            CENTRE[1] + east / 83_000.0,
            GROUND_M + agl_m,
        ],
        speed_m_s: 0.0,
    }
}

/// One fix: the error, north, east and down, and the stated covariance.
struct Measured {
    error: Vector3<f64>,
    total: Matrix3<f64>,
}

fn matrix(c: &SymmetricCov3) -> Matrix3<f64> {
    let [a, b, c2, d, e, f] = c.upper_triangle();
    Matrix3::new(a, b, c2, b, d, e, c2, e, f)
}

fn measure(
    (map, truth): (&World, &World),
    flight: &Flight,
    mount: CameraMount,
    mismatch: Mismatch,
) -> Option<Measured> {
    let mut navigator = ReferenceNavigator::new(suite(imager(), mount), CLOCK, None).unwrap();
    let [lat, lon, h] = flight.start;
    navigator
        .align(&Alignment {
            position: GeodeticPosition::new(
                lat.to_radians() + 8.0 / 6_364_000.0,
                lon.to_radians() - 6.0 / 4_760_000.0,
                h + 3.0,
            ),
            velocity: NedVelocity::new(0.0, 0.0, 0.0),
            attitude: AttitudeQuaternion::new(1.0, 0.0, 0.0, 0.0),
            position_cov: SymmetricCov3::from_diagonal(400.0, 400.0, 400.0),
            velocity_cov: SymmetricCov3::from_diagonal(0.01, 0.01, 0.01),
            attitude_cov: SymmetricCov3::from_diagonal(1e-6, 1e-6, 1e-5),
            at: at(0),
            imu_bias_prior: None,
        })
        .unwrap();
    let request = TerrainMapRequest {
        config: TerrainConfig::default(),
        ..TerrainMapRequest::new(map.package, TODAY)
    };
    navigator
        .load_terrain_map(Arc::clone(&map.store), &request, None)
        .ok()?;
    navigator.set_terrain_match_inspection(true);
    let mut sink = Vec::new();
    for step in 1..=RATE {
        navigator.ingest(&flight.increment(step), &mut sink).ok()?;
    }
    let level = UnitQuaternion::identity();
    let rendered = frame_at(truth, &camera(), flight.body(RATE), &level, &mount, RATE);
    let seed = (flight.start[0] * 1e6) as u64 ^ (flight.start[1].abs() * 1e6) as u64;
    let luma = mismatch.apply(&rendered, 400, 300, seed);
    let frame = SensorInput {
        sensor: CAMERA,
        acquired_at: at(RATE),
        delivered_at: at(RATE),
        data: SensorData::TerrainFrame {
            width: 400,
            height: 300,
            luma,
        },
    };
    navigator.ingest(&frame, &mut sink).ok()?;
    let record = navigator.last_terrain_match(CAMERA)?;
    let fix = record.fix()?;
    let [lat, lon, h] = flight.body(RATE);
    let p = fix.position;
    let error = Vector3::new(
        (p.latitude_rad - lat.to_radians()) * 6_364_000.0,
        (p.longitude_rad - lon.to_radians()) * 6_378_137.0 * lat.to_radians().cos(),
        -(p.altitude_m - h),
    );
    Some(Measured {
        error,
        total: matrix(&fix.covariance_ned),
    })
}

/// One case of the grid: the map and the ground the camera sees, and the
/// camera.
struct Case {
    name: &'static str,
    map: (Texture, [f64; 2], u32),
    ground: Texture,
    relief_m: f64,
    camera: Mismatch,
}

const CASES: &[Case] = &[
    Case {
        name: "farm-exact",
        map: (Texture::Farmland, [0.0, 0.0], 1),
        ground: Texture::Farmland,
        relief_m: 80.0,
        camera: Mismatch::EXACT,
    },
    Case {
        name: "farm-photo",
        map: (Texture::Farmland, [0.0, 0.0], 1),
        ground: Texture::Farmland,
        relief_m: 80.0,
        camera: Mismatch::PHOTOMETRIC,
    },
    Case {
        name: "farm-optics",
        map: (Texture::Farmland, [0.0, 0.0], 1),
        ground: Texture::Farmland,
        relief_m: 80.0,
        camera: Mismatch::OPTICS,
    },
    Case {
        name: "farm-all",
        map: (Texture::Farmland, [0.0, 0.0], 1),
        ground: Texture::Farmland,
        relief_m: 80.0,
        camera: Mismatch::ALL,
    },
    Case {
        name: "farm-season",
        map: (Texture::Farmland, [0.0, 0.0], 1),
        ground: Texture::FarmlandSeason,
        relief_m: 80.0,
        camera: Mismatch::ALL,
    },
    Case {
        name: "farm-reg",
        map: (Texture::Farmland, [1.5, -1.0], 2_000),
        ground: Texture::Farmland,
        relief_m: 80.0,
        camera: Mismatch::ALL,
    },
    Case {
        name: "town-exact",
        map: (Texture::TownIrregular, [0.0, 0.0], 1),
        ground: Texture::TownIrregular,
        relief_m: 30.0,
        camera: Mismatch::EXACT,
    },
    Case {
        name: "town-all",
        map: (Texture::TownIrregular, [0.0, 0.0], 1),
        ground: Texture::TownIrregular,
        relief_m: 30.0,
        camera: Mismatch::ALL,
    },
    Case {
        name: "low-exact",
        map: (Texture::LowModerate, [0.0, 0.0], 1),
        ground: Texture::LowModerate,
        relief_m: 80.0,
        camera: Mismatch::EXACT,
    },
    Case {
        name: "low-all",
        map: (Texture::LowModerate, [0.0, 0.0], 1),
        ground: Texture::LowModerate,
        relief_m: 80.0,
        camera: Mismatch::ALL,
    },
    Case {
        name: "town-grid-exact",
        map: (Texture::Town, [0.0, 0.0], 1),
        ground: Texture::Town,
        relief_m: 30.0,
        camera: Mismatch::EXACT,
    },
    Case {
        name: "faint-all",
        map: (Texture::Low, [0.0, 0.0], 1),
        ground: Texture::Low,
        relief_m: 80.0,
        camera: Mismatch::ALL,
    },
];

/// The 0.1 % and 99.9 % points of `χ²(3n)/n`, by Wilson–Hilferty.
fn band(n: usize) -> (f64, f64) {
    let k = (3 * n) as f64;
    let a = 2.0 / (9.0 * k);
    let at = |z: f64| k * (1.0 - a + z * a.sqrt()).powi(3) / n as f64;
    (at(-3.09), at(3.09))
}

/// The mean normalized error of the fixes of `case` at each height of
/// `heights`, view of `offs`, and position set of `sets`, with the count
/// of fixes: groups of fewer than four fixes are left out.
fn groups(
    case: &Case,
    heights: &[f64],
    offs: &[f64],
    sets: &[u32],
) -> Vec<((f64, f64, u32), usize, f64)> {
    let (texture, shift, registration) = case.map;
    let map = world(texture, case.relief_m, shift, registration);
    let exact = (case.ground, [0.0, 0.0], 1) == case.map;
    let ground = (!exact).then(|| world(case.ground, case.relief_m, [0.0, 0.0], 1));
    let ground = ground.as_ref().unwrap_or(&map);
    let mut out = Vec::new();
    for &agl_m in heights {
        for &off_deg in offs {
            for &set in sets {
                let nees: Vec<f64> = (0..9)
                    .filter_map(|k| {
                        let flight = hover(k, set, agl_m);
                        let f = measure((&map, ground), &flight, mount(off_deg), case.camera)?;
                        Some((f.error.transpose() * f.total.try_inverse()? * f.error)[(0, 0)])
                    })
                    .collect();
                if nees.len() >= 4 {
                    let mean = nees.iter().sum::<f64>() / nees.len() as f64;
                    out.push(((agl_m, off_deg, set), nees.len(), mean));
                }
            }
        }
    }
    out
}

/// Checks each group of `case`: the fixes are never over-confident, and
/// their sigma is not more than twice too large past the low edge of the
/// band.
fn check(case: &Case, heights: &[f64], offs: &[f64], sets: &[u32]) -> usize {
    let groups = groups(case, heights, offs, sets);
    for ((agl_m, off_deg, set), n, mean) in &groups {
        let (low, high) = band(*n);
        tracing::info!(
            case = case.name,
            agl_m,
            off_deg,
            set,
            n,
            mean,
            low,
            high,
            "terrain fix NEES"
        );
        let at = format!(
            "{} {agl_m} m {off_deg}° set {set}: {mean:.2} of {n}",
            case.name
        );
        assert!(*mean <= high, "over-confident: {at} > {high:.2}");
        assert!(*mean >= low / 4.0, "too cautious: {at} < {:.2}", low / 4.0);
    }
    groups.len()
}

/// The cases of the release test: the exact map and the full camera
/// mismatch over fields, low texture with the full mismatch, and a town.
const RELEASE: [usize; 4] = [0, 3, 9, 6];

/// From 1000 m and 1760 m, at the nadir and 30°: the fixes over fields,
/// with the camera as the map shows it and with each mismatch at once,
/// over low texture with that mismatch, and over a town, state their
/// error, neither over-confident nor more than about twice too cautious.
#[test]
#[cfg_attr(debug_assertions, ignore = "needs an optimized build")]
fn a_terrain_fix_states_its_error_under_camera_mismatch() {
    super::flight::show_events();
    let mut checked = 0;
    for index in RELEASE {
        checked += check(&CASES[index], &[1_000.0, 1_760.0], &[0.0, 30.0], &[0]);
    }
    assert!(checked >= 10, "{checked}");
}

/// Every case of [`CASES`] from 400 m, 1000 m and 1760 m at 0°, 30° and
/// 45°, on both sets of positions: the grid that calibrates the floor and
/// the residual correlation of the terrain match.
#[test]
#[ignore = "the full calibration grid: about ten minutes in an optimized build"]
fn the_full_grid_states_the_error_of_a_terrain_fix() {
    super::flight::show_events();
    let mut checked = 0;
    for case in CASES {
        checked += check(
            case,
            &[400.0, 1_000.0, 1_760.0],
            &[0.0, 30.0, 45.0],
            &[0, 1],
        );
    }
    assert!(checked >= 80, "{checked}");
}
