use super::*;

struct HoldoutScene {
    world: Vec<[f64; 3]>,
    pixels: Vec<[f64; 2]>,
    k: Matrix3<f64>,
    size: [u32; 2],
    true_eye: [f64; 3],
    true_rot: Matrix3<f64>,
}

fn synthetic_holdout_scene(scale_m: f64, outlier_every_five: usize) -> HoldoutScene {
    let size = [960, 720];
    let k = Matrix3::new(780.0, 0.0, 479.5, 0.0, 780.0, 359.5, 0.0, 0.0, 1.0);
    let true_eye = [0.21 * scale_m, -0.14 * scale_m, 0.92 * scale_m];
    let look_down = Matrix3::new(1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0);
    let tilt = Rotation3::from_euler_angles(0.16, -0.11, 0.35).into_inner();
    let true_rot = tilt * look_down;

    let mut world = Vec::with_capacity(80);
    let mut pixels = Vec::with_capacity(80);
    for gy in 0..8 {
        for gx in 0..10 {
            let u = 48.0 + f64::from(gx) * 96.0;
            let v = 45.0 + f64::from(gy) * 90.0;
            let depth = scale_m * (0.82 + 0.14 * (f64::from(gx) * 0.7 + f64::from(gy) * 0.9).sin());
            let ray_cam = Vector3::new(
                (u - k[(0, 2)]) * depth / k[(0, 0)],
                (v - k[(1, 2)]) * depth / k[(1, 1)],
                depth,
            );
            let pt = Vector3::from(true_eye) + true_rot.transpose() * ray_cam;
            world.push([pt.x, pt.y, pt.z]);
            pixels.push([u, v]);
        }
    }

    let mut outliers = 0usize;
    for (idx, px) in pixels.iter_mut().enumerate() {
        if (idx * 3 + 1) % 5 < outlier_every_five {
            let mag = 120.0 + (outliers as f64) * 6.0;
            let ang = 0.4 + (outliers as f64) * 0.23;
            px[0] += mag * ang.cos();
            px[1] += mag * ang.sin();
            outliers += 1;
        }
    }
    HoldoutScene {
        world,
        pixels,
        k,
        size,
        true_eye,
        true_rot,
    }
}

#[test]
fn l1_gnc_and_hybrid_accept_accurate_pose_under_40_percent_outliers() {
    let s = synthetic_holdout_scene(1_000.0, 2);
    let init_eye = [
        s.true_eye[0] + 65.0,
        s.true_eye[1] - 45.0,
        s.true_eye[2] + 30.0,
    ];
    let init_rot = Rotation3::from_euler_angles(0.06, -0.05, 0.07).into_inner() * s.true_rot;

    let cand = fit_pose_l1_gnc(&s.world, &s.pixels, &s.k, s.size, (init_eye, init_rot))
        .expect("L1-GNC must produce a pose candidate");
    let repeat = fit_pose_l1_gnc(&s.world, &s.pixels, &s.k, s.size, (init_eye, init_rot))
        .expect("repeated L1-GNC call must match");
    assert_eq!(cand, repeat);
    assert!(cand.score.geometry_accepted, "score: {:?}", cand.score);

    let eye_err = (Vector3::from(cand.eye) - Vector3::from(s.true_eye)).norm();
    let rel_rot = cand.rotation * s.true_rot.transpose();
    let trace_cos = ((rel_rot.trace() - 1.0) * 0.5).clamp(-1.0, 1.0);
    let rot_err_deg = trace_cos.acos().to_degrees();
    assert!(eye_err < 0.05, "eye error {eye_err} m >= 0.05 m");
    assert!(
        rot_err_deg < 0.01,
        "rotation error {rot_err_deg} deg >= 0.01 deg"
    );
    assert_eq!(cand.score.inliers, 48);

    let hybrid = fit_pose_hybrid(
        &s.world,
        &s.pixels,
        &s.k,
        s.size,
        Some((init_eye, init_rot)),
    )
    .expect("hybrid solver must accept deterministic L1-GNC candidate");
    assert_eq!(hybrid, cand);

    let near = synthetic_holdout_scene(10.0, 1);
    let fallback = fit_pose_hybrid(&near.world, &near.pixels, &near.k, near.size, None)
        .expect("hybrid solver without prior must fall back to RANSAC");
    assert!(fallback.score.geometry_accepted);
}

#[test]
fn enforces_displacement_guard_and_minimum_pair_count() {
    let s = synthetic_holdout_scene(1_000.0, 2);
    assert!(
        fit_pose_l1_gnc(
            &s.world[..15],
            &s.pixels[..15],
            &s.k,
            s.size,
            (s.true_eye, s.true_rot)
        )
        .is_none()
    );

    let far_eye = [s.true_eye[0] + 5_500.0, s.true_eye[1], s.true_eye[2]];
    assert!(fit_pose_l1_gnc(&s.world, &s.pixels, &s.k, s.size, (far_eye, s.true_rot)).is_none());
}

#[test]
fn hybrid_recovers_low_agl_coplanar_nadir_under_large_shift_and_yaw_offset() {
    let size = [800, 450];
    let k = Matrix3::new(533.0, 0.0, 400.0, 0.0, 533.0, 225.0, 0.0, 0.0, 1.0);
    let true_eye = [42.0, -31.0, 56.0];
    let look_down = Matrix3::new(1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0);
    let yaw_68 = 68.0_f64.to_radians();
    let g_true = Matrix3::new(
        yaw_68.cos(),
        -yaw_68.sin(),
        0.0,
        yaw_68.sin(),
        yaw_68.cos(),
        0.0,
        0.0,
        0.0,
        1.0,
    );
    let true_rot = look_down * g_true.transpose();

    let mut world = Vec::new();
    let mut pixels = Vec::new();
    for gy in 0..7 {
        for gx in 0..9 {
            let u = 48.0 + f64::from(gx) * 88.0;
            let v = 36.0 + f64::from(gy) * 63.0;
            let ray_cam = Vector3::new((u - 400.0) / 533.0, (v - 225.0) / 533.0, 1.0);
            let ray_w = true_rot.transpose() * ray_cam;
            let t = -true_eye[2] / ray_w.z;
            let pt = Vector3::from(true_eye) + ray_w * t;
            world.push([pt.x, pt.y, 0.0]);
            pixels.push([u, v]);
        }
    }
    for (idx, px) in pixels.iter_mut().enumerate() {
        if idx % 3 == 0 {
            px[0] = (px[0] + 210.0) % 780.0;
            px[1] = (px[1] + 155.0) % 430.0;
        }
    }

    let seed_eye = [0.0, 0.0, 72.0];
    let yaw_30 = 30.0_f64.to_radians();
    let g_seed = Matrix3::new(
        yaw_30.cos(),
        -yaw_30.sin(),
        0.0,
        yaw_30.sin(),
        yaw_30.cos(),
        0.0,
        0.0,
        0.0,
        1.0,
    );
    let seed_rot = look_down * g_seed.transpose();

    let cand = fit_pose_hybrid(&world, &pixels, &k, size, Some((seed_eye, seed_rot)))
        .expect("planar low-AGL nadir with 38 deg yaw offset must solve");
    assert!(cand.score.geometry_accepted, "score: {:?}", cand.score);
    let err_m = (Vector3::from(cand.eye) - Vector3::from(true_eye)).norm();
    assert!(err_m < 0.25, "expected < 0.25 m error, got {err_m}");
}

#[test]
fn score_pose_candidate_adapts_to_sparse_holdout_and_low_agl_relief_parallax() {
    let size = [800, 450];
    let k = Matrix3::new(577.0, 0.0, 400.0, 0.0, 577.0, 225.0, 0.0, 0.0, 1.0);
    let eye = [0.0, 0.0, 98.0];
    let rot = Matrix3::new(1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0);

    let mut world = Vec::new();
    let mut pixels = Vec::new();
    for gy in 0..5 {
        for gx in 0..6 {
            let u = 80.0 + f64::from(gx) * 120.0;
            let v = 50.0 + f64::from(gy) * 85.0;
            let x = (u - 400.0) * 98.0 / 577.0;
            let y = -(v - 225.0) * 98.0 / 577.0;
            world.push([x, y, 0.0]);
            // Inject ~9.0 px unmodeled suburban roof relief parallax at 98 m AGL.
            pixels.push([u + 6.5, v - 6.2]);
        }
    }
    // Case A: 30 points with only 3 in holdout (27 train + 3 hold = 30 inliers).
    let mut held_sparse = vec![false; world.len()];
    held_sparse[0] = true;
    held_sparse[10] = true;
    held_sparse[20] = true;
    let c_sparse = score_pose_candidate(
        &world,
        &pixels,
        &held_sparse,
        (eye, rot),
        &k,
        size,
        scaled_inlier_px(size),
    );
    assert!(c_sparse.score.geometry_accepted, "{:?}", c_sparse.score);

    // Case B: 27 inliers split as 18 train + 9 holdout out of 30 points (3 outliers).
    let mut pixels_b = pixels.clone();
    pixels_b[27] = [790.0, 440.0];
    pixels_b[28] = [10.0, 10.0];
    pixels_b[29] = [790.0, 10.0];
    let mut held_12 = vec![false; world.len()];
    for slot in held_12.iter_mut().take(9) {
        *slot = true;
    }
    held_12[27] = true;
    held_12[28] = true;
    held_12[29] = true;
    let c_18_9 = score_pose_candidate(
        &world,
        &pixels_b,
        &held_12,
        (eye, rot),
        &k,
        size,
        scaled_inlier_px(size),
    );
    assert_eq!(c_18_9.score.train_inliers, 18);
    assert_eq!(c_18_9.score.inliers, 27);
    assert!(c_18_9.score.geometry_accepted, "{:?}", c_18_9.score);
}
