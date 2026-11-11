//! EXPLORATION: the stated covariance of a terrain fix against its error,
//! through the real matcher, over heights, view angles and textures.

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

fn world(texture: Texture, relief_m: f64) -> World {
    World::build(&WorldSpec {
        accuracy_mm: Some((1, 1)),
        texture,
        relief_m,
        radius_m: 6_000.0,
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

/// One fix: the error, north east down, the stated covariance, the
/// external part, the frame model one-sigma, and the inliers, the spatial
/// support and the reprojection RMS of the match.
struct Measured {
    error: Vector3<f64>,
    total: Matrix3<f64>,
    external: Matrix3<f64>,
    model_m: f64,
    quality: (usize, usize, f64),
}

fn matrix(c: &SymmetricCov3) -> Matrix3<f64> {
    let [a, b, c2, d, e, f] = c.upper_triangle();
    Matrix3::new(a, b, c2, b, d, e, c2, e, f)
}

fn measure(map: &World, flight: &Flight, mount: CameraMount) -> Option<Measured> {
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
    let luma = frame_at(map, &camera(), flight.body(RATE), &level, &mount, RATE);
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
    let solved = record.solved()?;
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
        external: matrix(&fix.external_ned),
        model_m: fix.frame_model_m,
        quality: (
            solved.inliers,
            solved.spatial_support,
            solved.reprojection_rms_px,
        ),
    })
}

#[test]
#[cfg_attr(debug_assertions, ignore = "needs an optimized build")]
fn explore_geometry_covariance() {
    super::flight::show_events();
    let only = std::env::var("TEXTURE").unwrap_or_default();
    let worlds = [
        ("farmland", Texture::Farmland, 80.0),
        ("town", Texture::Town, 30.0),
        ("hills", Texture::Farmland, 300.0),
        ("low", Texture::Low, 80.0),
        ("town2", Texture::TownIrregular, 30.0),
        ("lowmod", Texture::LowModerate, 80.0),
    ];
    for (name, texture, relief) in worlds {
        if !only.is_empty() && only != name {
            continue;
        }
        let map = world(texture, relief);
        for agl_m in [400.0, 800.0, 1_200.0, 1_600.0, 2_000.0] {
            for off_deg in [0.0, 30.0, 45.0] {
                let mut fixes = Vec::new();
                for set in 0..2 {
                    for k in 0..9 {
                        fixes.extend(measure(&map, &hover(k, set, agl_m), mount(off_deg)));
                    }
                }
                let n = fixes.len() as f64;
                if fixes.is_empty() {
                    tracing::warn!("GEOM {name} agl={agl_m} off={off_deg} n=0");
                    continue;
                }
                let nees = fixes
                    .iter()
                    .filter_map(|f| {
                        Some((f.error.transpose() * f.total.try_inverse()? * f.error)[(0, 0)])
                    })
                    .sum::<f64>()
                    / n;
                let unfloored = fixes
                    .iter()
                    .filter_map(|f| {
                        let (_, support, rms) = f.quality;
                        let n = support as f64;
                        let raw = rms * rms * n / (2.0 * n - 6.0);
                        let g = (f.total - f.external - Matrix3::identity() * f.model_m.powi(2))
                            * (raw / raw.max(1.0));
                        let c = g + f.external + Matrix3::identity() * f.model_m.powi(2);
                        Some((f.error.transpose() * c.try_inverse()? * f.error)[(0, 0)])
                    })
                    .sum::<f64>()
                    / n;
                let rms_h = (fixes
                    .iter()
                    .map(|f| f.error.xy().norm_squared() / 2.0)
                    .sum::<f64>()
                    / n)
                    .sqrt();
                let rms_v = (fixes.iter().map(|f| f.error.z * f.error.z).sum::<f64>() / n).sqrt();
                let geometry =
                    |f: &Measured| f.total - f.external - Matrix3::identity() * f.model_m.powi(2);
                let sig_h = (fixes
                    .iter()
                    .map(|f| (geometry(f)[(0, 0)] + geometry(f)[(1, 1)]) / 2.0)
                    .sum::<f64>()
                    / n)
                    .sqrt();
                let sig_v = (fixes.iter().map(|f| geometry(f)[(2, 2)]).sum::<f64>() / n).sqrt();
                let model = fixes.iter().map(|f| f.model_m).sum::<f64>() / n;
                let ext = (fixes.iter().map(|f| f.external.trace() / 3.0).sum::<f64>() / n).sqrt();
                let inl = fixes.iter().map(|f| f.quality.0 as f64).sum::<f64>() / n;
                let sup = fixes.iter().map(|f| f.quality.1 as f64).sum::<f64>() / n;
                let rpx = fixes.iter().map(|f| f.quality.2).sum::<f64>() / n;
                tracing::warn!(
                    "GEOM {name} agl={agl_m} off={off_deg} n={n} nees={nees:.2} unfloored={unfloored:.2} err_h={rms_h:.2} err_v={rms_v:.2} geom_h={sig_h:.2} geom_v={sig_v:.2} model={model:.2} ext={ext:.2} inliers={inl:.0} support={sup:.0} rms_px={rpx:.2}"
                );
            }
        }
    }
}
