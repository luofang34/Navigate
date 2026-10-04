use super::*;
use crate::CameraPose;

fn camera() -> CameraModel {
    CameraModel {
        width: 640,
        height: 360,
        fx: 459.3,
        fy: 459.3,
        cx: 319.5,
        cy: 179.5,
    }
}

fn pose(position: [f64; 3], roll: f64, pitch: f64, yaw: f64) -> CameraPose {
    CameraPose {
        position: Vector3::from(position),
        orientation: UnitQuaternion::from_euler_angles(roll, pitch, yaw),
    }
}

/// Ground points `z = relief(x, y)` seen by both cameras, as matches.
fn matches(a: &CameraPose, b: &CameraPose, relief: impl Fn(f64, f64) -> f64) -> Vec<PixelMatch> {
    let c = camera();
    let mut out = Vec::new();
    for row in 0..12 {
        for col in 0..20 {
            let pixel = Vector2::new(16.0 + 31.0 * f64::from(col), 12.0 + 30.0 * f64::from(row));
            let ray = a.orientation
                * Vector3::new((pixel.x - c.cx) / c.fx, (c.cy - pixel.y) / c.fy, -1.0);
            if ray.z >= -1e-3 {
                continue;
            }
            // Intersect the relief surface by fixed-point iteration from the plane z = 0.
            let mut depth = -a.position.z / ray.z;
            for _ in 0..20 {
                let p = a.position + ray * depth;
                depth = (relief(p.x, p.y) - a.position.z) / ray.z;
            }
            let world = a.position + ray * depth;
            if let Some(q) = c.project(b, world)
                && q.x > 1.0
                && q.y > 1.0
                && q.x < 638.0
                && q.y < 358.0
            {
                out.push(PixelMatch {
                    reference: pixel,
                    query: q,
                });
            }
        }
    }
    out
}

fn truth(a: &CameraPose, b: &CameraPose) -> (UnitQuaternion<f64>, Vector3<f64>, Vector3<f64>) {
    let d = a.position.z;
    let rotation = a.orientation.inverse() * b.orientation;
    let translation = a.orientation.inverse() * (b.position - a.position) / d;
    let normal = a.orientation.inverse() * Vector3::z();
    (rotation, translation, normal)
}

#[test]
fn recovers_rotation_translation_and_ground_normal_at_any_tilt() {
    for (tilt, yaw) in [(0.05, 0.3), (0.35, -1.2), (0.7, 2.5)] {
        let a = pose([10.0, -5.0, 120.0], tilt, 0.1 * tilt, yaw);
        let b = pose(
            [13.0, -3.0, 121.0],
            tilt + 0.01,
            0.1 * tilt - 0.004,
            yaw + 0.02,
        );
        let found = plane_motion(
            &camera(),
            &matches(&a, &b, |_, _| 0.0),
            &PlaneMotionConfig::default(),
        )
        .unwrap_or_else(|e| panic!("tilt {tilt}: {e}"));
        let (rotation, translation, normal) = truth(&a, &b);
        let matching = found
            .solutions
            .iter()
            .find(|s| s.normal.angle(&normal) < 1e-6);
        let solution =
            matching.unwrap_or_else(|| panic!("tilt {tilt}: {found:?} lacks {normal:?}"));
        assert!(solution.rotation.angle_to(&rotation) < 1e-6, "tilt {tilt}");
        assert!(
            (solution.translation_per_distance - translation).norm() < 1e-6,
            "tilt {tilt}"
        );
        // Near nadir, visibility leaves one solution; a strongly tilted view
        // moving along its axis may keep both.
        if tilt < 0.5 {
            assert!(found.unique().is_some(), "tilt {tilt}: {found:?}");
        }
        assert!(found.normal_sigma_rad.is_finite() && found.normal_sigma_rad > 0.0);
    }
}

#[test]
fn small_parallax_and_pure_rotation_do_not_give_a_normal() {
    let a = pose([0.0, 0.0, 120.0], 0.05, 0.0, 0.0);
    let rotated = pose([0.0, 0.0, 120.0], 0.06, 0.01, 0.02);
    assert!(
        plane_motion(
            &camera(),
            &matches(&a, &rotated, |_, _| 0.0),
            &PlaneMotionConfig::default()
        )
        .is_err()
    );
    let crawl = pose([0.1, 0.0, 120.0], 0.05, 0.0, 0.0);
    assert!(
        plane_motion(
            &camera(),
            &matches(&a, &crawl, |_, _| 0.0),
            &PlaneMotionConfig::default()
        )
        .is_err()
    );
}

#[test]
fn strong_relief_is_not_reported_as_a_plane() {
    let a = pose([0.0, 0.0, 60.0], 0.1, 0.0, 0.0);
    let b = pose([4.0, 1.0, 60.0], 0.1, 0.0, 0.0);
    // Rolling hills with 20 m amplitude under a 60 m camera: no plane dominates.
    let hills = |x: f64, y: f64| 20.0 * (x / 15.0).sin() * (y / 15.0).cos();
    let result = plane_motion(
        &camera(),
        &matches(&a, &b, hills),
        &PlaneMotionConfig::default(),
    );
    assert!(result.is_err(), "{result:?}");
}

#[test]
fn outliers_do_not_move_the_normal() {
    let a = pose([0.0, 0.0, 100.0], 0.2, -0.05, 0.4);
    let b = pose([3.0, 2.0, 100.5], 0.205, -0.05, 0.41);
    let mut pairs = matches(&a, &b, |_, _| 0.0);
    let count = pairs.len() / 4;
    for (i, pair) in pairs.iter_mut().take(count).enumerate() {
        pair.query += Vector2::new(17.0 + i as f64 % 7.0, -11.0);
    }
    let found = plane_motion(&camera(), &pairs, &PlaneMotionConfig::default()).unwrap();
    let solution = found.unique().expect("one visible solution");
    assert!(solution.normal.angle(&truth(&a, &b).2) < 1e-6);
    assert!(found.inliers <= pairs.len() - count);
}

fn gaussian(seed: &mut u64) -> f64 {
    let mut uniform = || {
        *seed = seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((*seed >> 11) as f64 + 0.5) / (1_u64 << 53) as f64
    };
    let (a, b) = (uniform(), uniform());
    (-2.0 * a.ln()).sqrt() * (std::f64::consts::TAU * b).cos()
}

#[test]
fn the_normal_error_matches_its_declared_sigma_under_pixel_noise() {
    let mut seed = 99_u64;
    let mut ratios = Vec::new();
    for case in 0..120 {
        let u = |seed: &mut u64| gaussian(seed).abs().min(3.0) / 3.0;
        let tilt = 1.0 * u(&mut seed);
        let yaw = case as f64 * 0.7;
        let height = 60.0 + 200.0 * u(&mut seed);
        let a = pose([0.0, 0.0, height], tilt, 0.2 * tilt, yaw);
        let forward = a.orientation * Vector3::new(0.0, 0.0, -1.0);
        let lateral = Vector3::new(yaw.cos(), yaw.sin(), 0.0);
        let share = u(&mut seed);
        let step =
            (lateral * (1.0 - share) + forward * share) * height * (0.02 + 0.08 * u(&mut seed));
        let b = CameraPose {
            position: a.position + step,
            orientation: a.orientation * UnitQuaternion::from_euler_angles(0.01, -0.008, 0.012),
        };
        let mut pairs = matches(&a, &b, |_, _| 0.0);
        for pair in &mut pairs {
            pair.reference += Vector2::new(gaussian(&mut seed), gaussian(&mut seed)) * 0.3;
            pair.query += Vector2::new(gaussian(&mut seed), gaussian(&mut seed)) * 0.3;
        }
        let Ok(found) = plane_motion(&camera(), &pairs, &PlaneMotionConfig::default()) else {
            continue;
        };
        let normal = truth(&a, &b).2;
        let error = found
            .solutions
            .iter()
            .map(|s| s.normal.angle(&normal))
            .fold(f64::INFINITY, f64::min);
        ratios.push(error / found.normal_sigma_rad);
    }
    ratios.sort_by(f64::total_cmp);
    assert!(
        ratios.len() >= 80,
        "most noisy cases are accepted: {}",
        ratios.len()
    );
    let p90 = ratios[ratios.len() * 9 / 10];
    assert!(
        p90 <= 3.0,
        "p90 of error / sigma is {p90:.2}; median {:.2}",
        ratios[ratios.len() / 2]
    );
}

#[test]
fn motion_along_the_normal_gives_one_solution() {
    for climb in [5.0, -5.0] {
        let a = pose([0.0, 0.0, 100.0], 0.0, 0.0, 0.4);
        let b = pose([0.0, 0.0, 100.0 + climb], 0.0, 0.0, 0.4);
        let found = plane_motion(
            &camera(),
            &matches(&a, &b, |_, _| 0.0),
            &PlaneMotionConfig::default(),
        )
        .unwrap();
        assert!(found.unique().is_some(), "climb {climb}: {found:?}");
        assert!(found.solutions[0].normal.angle(&truth(&a, &b).2) < 1e-6);
    }
}
