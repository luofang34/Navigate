//! Frames whose appearance differs from the map: another exposure and
//! light, the optics of a camera, another season, and faint ground. A
//! match either fixes with an error that its covariance holds or refuses.

use std::sync::Arc;

use nalgebra::{Matrix2, UnitQuaternion, Vector2};
use navigate_camera::CameraMount;
use navigate_contract::{AttitudeQuaternion, GeodeticPosition, NedVelocity, SymmetricCov3};

use super::flight::{CAMERA, CLOCK, Flight, RATE, at, show_events, suite};
use super::geometry::mismatch::Mismatch;
use super::geometry::{camera, imager};
use super::world::{CENTRE, GROUND_M, TODAY, Texture, World, WorldSpec, frame_at};
use crate::{
    Alignment, Navigator, Outcome, ReferenceNavigator, SensorData, SensorInput, TerrainMapRequest,
};

/// Squared Mahalanobis length of a two-dimensional error that a
/// consistent covariance exceeds once in a thousand.
const NEES_999: f64 = 13.82;

fn world(texture: Texture) -> World {
    World::build(&WorldSpec {
        radius_m: 6_000.0,
        texture,
        ..WorldSpec::default()
    })
}

/// The outcome of one frame from `agl_m` above the ground at `start`,
/// with the map of `map` and the ground of `truth` seen through
/// `mismatch`, and for a fix its horizontal error and NEES under the
/// covariance it stated.
fn one(
    (map, truth, mismatch): (&World, &World, Mismatch),
    start: [f64; 3],
    seed: u64,
) -> (Outcome, Option<(f64, f64)>) {
    let hover = Flight {
        start,
        speed_m_s: 0.0,
    };
    let mount = CameraMount::nadir();
    let mut navigator = ReferenceNavigator::new(suite(imager(), mount), CLOCK, None).unwrap();
    let [lat, lon, h] = start;
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
    let request = TerrainMapRequest::new(map.package, TODAY);
    navigator
        .load_terrain_map(Arc::clone(&map.store), &request, None)
        .unwrap();
    navigator.set_terrain_match_inspection(true);
    let mut sink = Vec::new();
    for step in 1..=RATE {
        navigator.ingest(&hover.increment(step), &mut sink).unwrap();
    }
    let level = UnitQuaternion::identity();
    let rendered = frame_at(truth, &camera(), hover.body(RATE), &level, &mount, RATE);
    let input = SensorInput {
        sensor: CAMERA,
        acquired_at: at(RATE),
        delivered_at: at(RATE),
        data: SensorData::TerrainFrame {
            width: 400,
            height: 300,
            luma: mismatch.apply(&rendered, 400, 300, seed),
        },
    };
    navigator.ingest(&input, &mut sink).unwrap();
    let outcome = super::fixes::terrain(&sink)
        .last()
        .map(|r| r.0.clone())
        .unwrap();
    let fix = navigator
        .last_terrain_match(CAMERA)
        .and_then(|record| record.fix());
    let measured = fix.map(|fix| {
        let [lat, lon, _] = hover.body(RATE);
        let p = fix.position;
        let e = Vector2::new(
            (p.latitude_rad - lat.to_radians()) * 6_364_000.0,
            (p.longitude_rad - lon.to_radians()) * 6_378_137.0 * lat.to_radians().cos(),
        );
        let [nn, ne, _, ee, _, _] = fix.covariance_ned.upper_triangle();
        let nees = Matrix2::new(nn, ne, ne, ee)
            .try_inverse()
            .map_or(f64::INFINITY, |inverse| (e.transpose() * inverse * e).x);
        (e.norm(), nees)
    });
    (outcome, measured)
}

/// From 400 m and 1760 m, at eight places each: frames under another
/// exposure and light, under that and the optics of a camera, in another
/// season, and over faint ground. No accepted fix has an error past what
/// its covariance holds; another season is refused or fixed honestly; a
/// changed exposure still fixes, the matcher comparing the images with
/// their local contrast normalised; and ground of about two counts of
/// block contrast is refused as faint before any match.
#[test]
#[cfg_attr(debug_assertions, ignore = "needs an optimized build")]
fn a_frame_unlike_the_map_fixes_honestly_or_not_at_all() {
    show_events();
    let farm = world(Texture::Farmland);
    let season = world(Texture::FarmlandSeason);
    let faint = world(Texture::Low);
    let moderate = world(Texture::LowModerate);
    // The fewest fixes of the eight places from 400 m and from 1760 m.
    let cases = [
        ("photometric", (&farm, &farm, Mismatch::PHOTOMETRIC), [6, 6]),
        ("all", (&farm, &farm, Mismatch::ALL), [6, 6]),
        ("season", (&farm, &season, Mismatch::EXACT), [0, 0]),
        ("season all", (&farm, &season, Mismatch::ALL), [0, 0]),
        ("faint", (&faint, &faint, Mismatch::EXACT), [0, 0]),
        (
            "moderate all",
            (&moderate, &moderate, Mismatch::ALL),
            [0, 6],
        ),
    ];
    let (mut inconsistent, mut scarce, mut matched_faint) = (Vec::new(), Vec::new(), Vec::new());
    for (height, agl_m) in [400.0, 1_760.0].into_iter().enumerate() {
        for (name, case, fewest) in cases {
            let (mut fixes, mut neeses, mut worst_m) = (0, Vec::new(), 0.0_f64);
            for k in 0..8_u32 {
                let start = [
                    CENTRE[0] + (f64::from(k % 4) - 1.5) * 700.0 / 111_100.0,
                    CENTRE[1] + (f64::from(k / 4) - 0.5) * 700.0 / 83_000.0,
                    GROUND_M + agl_m,
                ];
                let (outcome, measured) = one(case, start, u64::from(k) + 3);
                if name == "faint" && outcome != Outcome::Unused("faint_frame") {
                    matched_faint.push((agl_m, k, outcome.clone()));
                }
                if outcome != Outcome::Accepted {
                    continue;
                }
                if let Some((error_m, nees)) = measured {
                    fixes += 1;
                    neeses.push(nees);
                    worst_m = worst_m.max(error_m);
                    if nees > NEES_999 {
                        inconsistent.push((name, agl_m, error_m, nees));
                    }
                }
            }
            let mean = neeses.iter().sum::<f64>() / neeses.len().max(1) as f64;
            tracing::info!(name, agl_m, fixes, mean_nees = mean, worst_m, "appearance");
            if fixes < fewest.get(height).copied().unwrap_or(0) {
                scarce.push((name, agl_m, fixes));
            }
        }
    }
    assert!(inconsistent.is_empty(), "{inconsistent:?}");
    assert!(scarce.is_empty(), "{scarce:?}");
    assert!(matched_faint.is_empty(), "{matched_faint:?}");
}
