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
    for gy in 0..10 {
        for gx in 0..12 {
            let u = 44.0 + f64::from(gx) * 64.0;
            let v = 32.0 + f64::from(gy) * 38.0;
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
    let rot = Matrix3::new(1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0);

    // 1. Standard-altitude regime (500 m AGL, relief_ratio == 1.0): 30 points with sparse holdout pass.
    let high_eye = [0.0, 0.0, 500.0];
    let mut high_world = Vec::new();
    let mut high_pixels = Vec::new();
    for gy in 0..5 {
        for gx in 0..6 {
            let u = 80.0 + f64::from(gx) * 120.0;
            let v = 50.0 + f64::from(gy) * 85.0;
            let x = (u - 400.0) * 500.0 / 577.0;
            let y = -(v - 225.0) * 500.0 / 577.0;
            high_world.push([x, y, 0.0]);
            high_pixels.push([u + 2.5, v - 2.0]);
        }
    }
    let mut held_sparse = vec![false; high_world.len()];
    held_sparse[0] = true;
    held_sparse[10] = true;
    held_sparse[20] = true;
    let c_sparse = score_pose_candidate(
        &high_world,
        &high_pixels,
        &held_sparse,
        (high_eye, rot),
        &k,
        size,
        scaled_inlier_px(size),
    );
    assert!(c_sparse.score.geometry_accepted, "{:?}", c_sparse.score);

    // 2. Low-AGL relief regime (98 m AGL -> eff_inlier_px = 12 px, relief_ratio > 1.5):
    //    31 random dark-noise correspondences MUST be rejected, while 60 real relief points pass.
    let low_eye = [0.0, 0.0, 98.0];
    let mut low_world = Vec::new();
    let mut low_pixels = Vec::new();
    for gy in 0..6 {
        for gx in 0..10 {
            let u = 50.0 + f64::from(gx) * 75.0;
            let v = 40.0 + f64::from(gy) * 72.0;
            let x = (u - 400.0) * 98.0 / 577.0;
            let y = -(v - 225.0) * 98.0 / 577.0;
            low_world.push([x, y, 0.0]);
            low_pixels.push([u + 6.5, v - 6.2]);
        }
    }
    let held_31 = crate::holdout_pnp::spatial_holdout(&low_pixels[..31], size);
    let c_dark_31 = score_pose_candidate(
        &low_world[..31],
        &low_pixels[..31],
        &held_31,
        (low_eye, rot),
        &k,
        size,
        scaled_inlier_px(size),
    );
    assert!(
        !c_dark_31.score.geometry_accepted,
        "31 low-AGL 12 px correspondences must be rejected to prevent dark-woods false positives: {:?}",
        c_dark_31.score
    );

    let held_60 = crate::holdout_pnp::spatial_holdout(&low_pixels, size);
    let c_real_60 = score_pose_candidate(
        &low_world,
        &low_pixels,
        &held_60,
        (low_eye, rot),
        &k,
        size,
        scaled_inlier_px(size),
    );
    assert!(
        c_real_60.score.geometry_accepted,
        "60 low-AGL correspondences with >= 12 holdout inliers and >= 10 cells must pass: {:?}",
        c_real_60.score
    );

    // 3. Sparse low-AGL views (< 60 pairs, hold_count < 12) must not hit an impossible min_hold = 12 gate:
    //    (a) DJI_0029 f-00013 regime: 42 pairs, 33 inliers, hold_count = 7, hold_inliers = 6, hold_median ~ 9.3 px.
    let mut dji29_world = Vec::with_capacity(42);
    let mut dji29_pixels = Vec::with_capacity(42);
    let mut dji29_held = vec![false; 42];
    for idx in 0..42_u32 {
        let u = 45.0 + f64::from(idx % 7) * 112.0;
        let v = 35.0 + f64::from(idx / 7) * 70.0;
        let x = (u - 400.0) * 98.0 / 577.0;
        let y = -(v - 225.0) * 98.0 / 577.0;
        dji29_world.push([x, y, 0.0]);
        let is_inlier = idx < 33;
        let err_px = if is_inlier {
            if idx < 7 { 9.3 } else { 7.4 }
        } else {
            48.0
        };
        dji29_pixels.push([u + err_px, v]);
    }
    for &h_idx in &[0_usize, 1, 2, 3, 4, 5, 35] {
        dji29_held[h_idx] = true;
    }
    let c_dji29 = score_pose_candidate(
        &dji29_world,
        &dji29_pixels,
        &dji29_held,
        (low_eye, rot),
        &k,
        size,
        scaled_inlier_px(size),
    );
    assert_eq!(c_dji29.score.holdout_count, 7);
    assert_eq!(c_dji29.score.inliers, 33);
    assert!(
        c_dji29.score.geometry_accepted,
        "DJI_0029 sparse 42-pair / 7-holdout lock must pass: {:?}",
        c_dji29.score
    );

    //    Also verify the same 42-pair / 33-inlier lock at 640x360 (base_inlier_px = 5.0, relief_ratio = 2.4).
    let size_640 = [640, 360];
    let k_640 = Matrix3::new(462.0, 0.0, 320.0, 0.0, 462.0, 180.0, 0.0, 0.0, 1.0);
    let dji29_px_640: Vec<[f64; 2]> = dji29_pixels
        .iter()
        .map(|&[u, v]| [u * 0.8, v * 0.8])
        .collect();
    let c_dji29_640 = score_pose_candidate(
        &dji29_world,
        &dji29_px_640,
        &dji29_held,
        (low_eye, rot),
        &k_640,
        size_640,
        scaled_inlier_px(size_640),
    );
    assert!(
        c_dji29_640.score.geometry_accepted,
        "DJI_0029 sparse 33-inlier lock must also pass at 640x360: {:?}",
        c_dji29_640.score
    );

    //    (b) Tight low-AGL pose with 28 sub-pixel / <6.25 px inliers and hold_count = 6 passes via tight gate.
    let mut tight_pixels = Vec::with_capacity(28);
    for (u_pt, &[x, y, _]) in low_world.iter().take(28).enumerate() {
        let u = 400.0 + x * 577.0 / 98.0;
        let v = 225.0 - y * 577.0 / 98.0;
        let jitter = if u_pt % 2 == 0 { 1.8 } else { -1.6 };
        tight_pixels.push([u + jitter, v - jitter * 0.5]);
    }
    let held_28 = crate::holdout_pnp::spatial_holdout(&tight_pixels, size);
    let c_tight_28 = score_pose_candidate(
        &low_world[..28],
        &tight_pixels,
        &held_28,
        (low_eye, rot),
        &k,
        size,
        scaled_inlier_px(size),
    );
    assert!(
        c_tight_28.score.geometry_accepted,
        "28 tight low-AGL inliers must pass standard tight gate without 12-holdout penalty: {:?}",
        c_tight_28.score
    );

    //    (c) DJI_0030 f-00001 regime: 34 pairs, 26 inliers, train_rms ~ 5.6 px, hold_count = 3, hold_inliers = 2.
    let mut dji30_world = Vec::with_capacity(34);
    let mut dji30_pixels = Vec::with_capacity(34);
    let mut dji30_held = vec![false; 34];
    for idx in 0..34_u32 {
        let u = 55.0 + f64::from(idx % 6) * 130.0;
        let v = 35.0 + f64::from(idx / 6) * 72.0;
        let x = (u - 400.0) * 98.0 / 577.0;
        let y = -(v - 225.0) * 98.0 / 577.0;
        dji30_world.push([x, y, 0.0]);
        let err_px = if idx < 2 {
            9.1
        } else if idx < 26 {
            5.5
        } else {
            52.0
        };
        dji30_pixels.push([u + err_px, v]);
    }
    for &h_idx in &[0_usize, 1, 30] {
        dji30_held[h_idx] = true;
    }
    let c_dji30 = score_pose_candidate(
        &dji30_world,
        &dji30_pixels,
        &dji30_held,
        (low_eye, rot),
        &k,
        size,
        scaled_inlier_px(size),
    );
    assert_eq!(c_dji30.score.inliers, 26);
    assert!(
        c_dji30.score.geometry_accepted,
        "DJI_0030 sparse 34-pair / 26-inlier relief lock with tight train RMS must pass: {:?}",
        c_dji30.score
    );
}
