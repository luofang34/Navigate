use super::build_anchor_seeds;
use crate::{
    corridor::canonical_tile_poses, stages::MatchStageOptions, surface::Surface, testset::PhotoCase,
};
use image::{Rgb, RgbImage};
use nalgebra::Matrix3;
use std::{error::Error, path::PathBuf};

#[test]
fn build_anchor_seeds_preserves_all_twelve_headings_after_predicted_pose() {
    let img = RgbImage::from_pixel(256, 256, Rgb([100, 110, 95]));
    let surface = Surface::synthetic(41.0, -75.0, 100.0, img);
    let case = PhotoCase {
        id: "seed-check".into(),
        image: "img.png".into(),
        image_sha256: "0".repeat(64),
        size: [800, 600],
        valid_mask: String::new(),
        mask_sha256: String::new(),
        mask_status: String::new(),
        semantic_labels: None,
        semantic_sha256: None,
        prior: None,
        sequence: None,
        time_s: None,
    };
    let stage_opts = MatchStageOptions {
        dataset_dir: PathBuf::new(),
        profile_path: PathBuf::new(),
        catalog_path: PathBuf::new(),
        context_catalog: None,
        output_dir: PathBuf::new(),
        attempts: 12,
        steps: 1,
        heading_step_deg: 30.0,
        height_factors: vec![1.0],
        only_cases: None,
    };
    let pred = ([10.0, 20.0, 900.0], Matrix3::identity());
    let seeds = build_anchor_seeds(
        &surface,
        &case,
        &stage_opts,
        Some((35.0, 0.0)),
        Some(pred),
        12,
    );
    assert_eq!(
        seeds.len(),
        13,
        "predicted pose plus 12 compass headings must yield 13 seeds"
    );
    assert_eq!(seeds[0].0, pred.0);
    let expected_330 =
        surface.camera_rotation(surface.origin_lat, surface.origin_lon, 330.0, 35.0, 0.0);
    assert!((seeds[12].1 - expected_330).norm() < 1e-9);
}

#[test]
fn canonical_tile_poses_includes_nadir_view_with_obliques() -> Result<(), Box<dyn Error>> {
    let img = RgbImage::from_pixel(256, 256, Rgb([100, 110, 95]));
    let surface = Surface::synthetic(41.0, -75.0, 100.0, img);
    let poses = canonical_tile_poses(&surface, (0.0, 0.0, 100.0), 600.0, 480.0, true);
    assert_eq!(
        poses.len(),
        25,
        "expected 24 oblique poses plus 1 nadir pose"
    );
    let &(band, _, nadir_rot) = poses.last().ok_or("missing nadir pose")?;
    assert_eq!(band, 24);
    assert!(
        -nadir_rot[(2, 2)] > 0.999,
        "final canonical pose must be 89 deg nadir"
    );
    Ok(())
}

#[test]
fn test_is_plausible_anchor_candidate_validation() {
    let img = RgbImage::from_pixel(256, 256, Rgb([100, 110, 95]));
    let surface = Surface::synthetic(41.0, -75.0, 100.0, img);
    let eye = [0.0, 0.0, 200.0];
    let rot = Matrix3::identity();
    let k = Matrix3::new(600.0, 0.0, 400.0, 0.0, 600.0, 300.0, 0.0, 0.0, 1.0);
    let size = [800, 600];

    // Eight non-coplanar 3D points in front of the camera (Z > eye[2] = 200.0 m).
    let points = vec![
        [10.0, 7.0, 240.0],
        [-10.0, 7.0, 245.0],
        [10.0, -7.0, 250.0],
        [-10.0, -7.0, 255.0],
        [12.0, 0.0, 260.0],
        [-12.0, 0.0, 265.0],
        [0.0, 8.0, 270.0],
        [0.0, -8.0, 275.0],
    ];

    let pixels: Vec<[f64; 2]> = points
        .iter()
        .map(|p| {
            let z = p[2] - eye[2];
            [
                k[(0, 0)] * p[0] / z + k[(0, 2)],
                k[(1, 1)] * p[1] / z + k[(1, 2)],
            ]
        })
        .collect();

    let inliers = vec![true; points.len()];

    let mut cand = crate::pose::PoseCandidate {
        eye,
        rotation: rot,
        k,
        size,
        points,
        pixels,
        inliers,
        score: crate::pose::PoseScore {
            focal_px: 600.0,
            geometry_accepted: true,
            holdout_count: 2,
            holdout_inlier_fraction: 1.0,
            holdout_median_px: 0.0,
            image_support_fraction: 0.5,
            inliers: 8,
            pairs: 8,
            positive_depth_fraction: 1.0,
            train_inliers: 6,
            train_rms_px: 0.0,
        },
    };
    assert!(super::is_plausible_anchor_candidate(&cand, &surface));
    assert!(
        !super::sweep::is_early_exit_focal_zero(&cand, &surface),
        "8 inliers must not trigger early focal lock"
    );

    cand.score.inliers = 48;
    cand.score.holdout_inlier_fraction = 0.40;
    assert!(
        super::sweep::is_early_exit_focal_zero(&cand, &surface),
        "48 plausible inliers with 0.40 holdout fraction must trigger early focal lock"
    );

    cand.score.holdout_inlier_fraction = 0.39;
    assert!(
        !super::sweep::is_early_exit_focal_zero(&cand, &surface),
        "holdout fraction below 0.40 must not trigger early focal lock"
    );
    cand.score.holdout_inlier_fraction = 1.0;

    // Underground candidate must be rejected
    cand.eye = [0.0, 0.0, -500.0];
    assert!(!super::is_plausible_anchor_candidate(&cand, &surface));
    assert!(!super::sweep::is_early_exit_focal_zero(&cand, &surface));
}
