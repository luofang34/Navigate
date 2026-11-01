use super::*;

#[test]
fn verify_physical_plausibility_agl_bounds() {
    let cfg = PhysicalPlausibilityConfig::default();
    assert!(verify_physical_plausibility(505.0, 500.0, 0.99, 1e-4, 1.0, &cfg).is_ok());
    assert!(matches!(
        verify_physical_plausibility(504.0, 500.0, 0.99, 1e-4, 1.0, &cfg),
        Err(PlausibilityError::SubSurfaceClearance { .. })
    ));
}

#[test]
fn stratospheric_and_orbital_clearance_has_no_ceiling() {
    let cfg = PhysicalPlausibilityConfig::default();
    assert!(verify_physical_plausibility(25_000.0, 100.0, 0.99, 1e-4, 1.0, &cfg).is_ok());
    assert!(verify_physical_plausibility(400_000.0, 0.0, 0.99, 1e-4, 1.0, &cfg).is_ok());
}

#[test]
fn cheirality_and_fisher_rejections() {
    let cfg = PhysicalPlausibilityConfig::default();
    assert!(matches!(
        verify_physical_plausibility(600.0, 500.0, 0.95, 1e-4, 1.0, &cfg),
        Err(PlausibilityError::LowCheiralityFraction { .. })
    ));
    assert!(matches!(
        verify_physical_plausibility(600.0, 500.0, 0.99, 1e-7, 1.0, &cfg),
        Err(PlausibilityError::IllConditionedFisher { .. })
    ));
    assert!(matches!(
        verify_physical_plausibility(600.0, 500.0, 0.99, 1e-4, 5.5, &cfg),
        Err(PlausibilityError::ExcessiveReprojectionError { .. })
    ));
}

#[test]
fn nan_and_inf_metrics_rejection() {
    let cfg = PhysicalPlausibilityConfig::default();
    assert_eq!(
        verify_physical_plausibility(500.0, 0.0, f64::NAN, 1e-4, 1.0, &cfg),
        Err(PlausibilityError::NonFinitePose)
    );
    assert_eq!(
        verify_physical_plausibility(500.0, 0.0, 0.99, f64::NAN, 1.0, &cfg),
        Err(PlausibilityError::NonFinitePose)
    );
    assert_eq!(
        verify_physical_plausibility(500.0, 0.0, 0.99, 1e-4, f64::INFINITY, &cfg),
        Err(PlausibilityError::NonFinitePose)
    );
}

#[test]
fn untrusted_prior_bypasses_bounding() {
    let cand_eye = [10000.0, 0.0, 500.0];
    assert!(verify_prior_bounding(cand_eye, None, false, 3.0));
}

struct FlatDemTerrain {
    dem_m: f64,
}

impl CorridorTerrain for FlatDemTerrain {
    fn geodetic(&self, local_enu: [f64; 3]) -> [f64; 3] {
        [39.5, -75.5, local_enu[2]]
    }

    fn local(&self, _lat_deg: f64, _lon_deg: f64, alt_m: f64) -> [f64; 3] {
        [0.0, 0.0, alt_m]
    }

    fn sample_dem(&self, _lat_deg: f64, _lon_deg: f64) -> Option<f64> {
        Some(self.dem_m)
    }

    fn camera_rotation(
        &self,
        _lat_deg: f64,
        _lon_deg: f64,
        heading_deg: f64,
        down_deg: f64,
        roll_deg: f64,
    ) -> Matrix3<f64> {
        local_camera_rotation(heading_deg, down_deg, roll_deg)
    }
}

fn build_synthetic_candidate_at_depth(
    depth_base: f64,
    inlier_err_px: f64,
    flip_inlier_depth: bool,
) -> PoseCandidate {
    let eye = [0.0, 0.0, 500.0];
    let rot = local_camera_rotation(0.0, 35.0, 0.0);
    let rot_inv = rot.transpose();
    let k = Matrix3::new(600.0, 0.0, 399.5, 0.0, 600.0, 299.5, 0.0, 0.0, 1.0);
    let size = [800_u32, 600_u32];

    let mut points = Vec::new();
    let mut pixels = Vec::new();
    let mut inliers = Vec::new();

    // 24 geometric inliers across a 6 x 4 grid with varied depths so Fisher rank is 6.
    for row in 0..4 {
        for col in 0..6 {
            let idx = row * 6 + col;
            let u = 80.0 + (col as f64) * 128.0;
            let v = 70.0 + (row as f64) * 150.0;
            let depth = if flip_inlier_depth && idx < 2 {
                -80.0
            } else {
                depth_base + ((idx % 5) as f64) * 35.0
            };
            let p_cam = Vector3::new(
                ((u - k[(0, 2)]) / k[(0, 0)]) * depth,
                ((v - k[(1, 2)]) / k[(1, 1)]) * depth,
                depth,
            );
            let p_world = Vector3::from(eye) + rot_inv * p_cam;
            let angle = (idx as f64) * 0.7;
            points.push([p_world.x, p_world.y, p_world.z]);
            pixels.push([
                u + inlier_err_px * angle.cos(),
                v + inlier_err_px * angle.sin(),
            ]);
            inliers.push(true);
        }
    }

    // 24 putative outliers (50% of matches) with 250 px error and half behind the camera.
    for out_idx in 0..24 {
        let u = 60.0 + ((out_idx % 6) as f64) * 130.0;
        let v = 50.0 + ((out_idx / 6) as f64) * 150.0;
        let depth: f64 = if out_idx % 2 == 0 { -120.0 } else { 220.0 };
        let p_cam = Vector3::new(
            ((u - k[(0, 2)]) / k[(0, 0)]) * depth.abs(),
            ((v - k[(1, 2)]) / k[(1, 1)]) * depth.abs(),
            depth,
        );
        let p_world = Vector3::from(eye) + rot_inv * p_cam;
        points.push([p_world.x, p_world.y, p_world.z]);
        pixels.push([(u + 250.0).min(790.0), (v + 250.0).min(590.0)]);
        inliers.push(false);
    }

    PoseCandidate {
        eye,
        rotation: rot,
        k,
        size,
        points,
        pixels,
        inliers,
        score: navigate_visual::holdout_pnp::PoseScore {
            focal_px: 600.0,
            geometry_accepted: true,
            holdout_count: 12,
            holdout_inlier_fraction: 0.50,
            holdout_median_px: inlier_err_px,
            image_support_fraction: 0.16,
            inliers: 24,
            pairs: 48,
            // Raw putative positive-depth fraction is contaminated by behind-camera outliers (0.75 < 0.98).
            positive_depth_fraction: 0.75,
            train_inliers: 18,
            train_rms_px: inlier_err_px,
        },
    }
}

#[test]
fn verify_candidate_plausibility_ignores_putative_outliers_and_checks_inliers() {
    let cfg = PhysicalPlausibilityConfig::default();
    let terrain = FlatDemTerrain { dem_m: 40.0 };

    let valid = build_synthetic_candidate_at_depth(500.0, 1.5, false);
    let holdout_rmse = compute_holdout_rmse_px(&valid);
    assert!(
        (holdout_rmse - 1.5).abs() < 0.25,
        "expected ~1.5 px inlier holdout RMSE, got {holdout_rmse}"
    );
    assert_eq!(
        verify_candidate_plausibility(&valid, &terrain, &cfg),
        Ok(())
    );
    assert!(is_sane_seed_candidate(&valid, &terrain, 5.0));

    let scaled_800px = build_synthetic_candidate_at_depth(500.0, 5.5, false);
    assert_eq!(
        verify_candidate_plausibility(&scaled_800px, &terrain, &cfg),
        Ok(())
    );

    let noisy_inliers = build_synthetic_candidate_at_depth(500.0, 7.5, false);
    assert!(matches!(
        verify_candidate_plausibility(&noisy_inliers, &terrain, &cfg),
        Err(PlausibilityError::ExcessiveReprojectionError { .. })
    ));

    let low_agl_relief = build_synthetic_candidate_at_depth(180.0, 8.5, false);
    assert_eq!(
        verify_candidate_plausibility(&low_agl_relief, &terrain, &cfg),
        Ok(())
    );
    let low_agl_excessive = build_synthetic_candidate_at_depth(180.0, 13.0, false);
    assert!(matches!(
        verify_candidate_plausibility(&low_agl_excessive, &terrain, &cfg),
        Err(PlausibilityError::ExcessiveReprojectionError { .. })
    ));

    let behind_camera_inliers = build_synthetic_candidate_at_depth(500.0, 1.5, true);
    assert!(matches!(
        verify_candidate_plausibility(&behind_camera_inliers, &terrain, &cfg),
        Err(PlausibilityError::LowCheiralityFraction { .. })
    ));
}
