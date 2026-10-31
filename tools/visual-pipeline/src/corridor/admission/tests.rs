use super::{extend_dedup_enu, top_up_tracks};
use crate::{
    pose::{PoseCandidate, PoseScore},
    surface::Surface,
};
use image::{Rgb, RgbImage};
use nalgebra::Matrix3;
use navigate_match::{
    ActiveTrackSet, DEFAULT_KASHIN_SEED, HadamardProjector, PreparedKashinQuery, SparseFeatures,
};
use std::error::Error;

fn dummy_score() -> PoseScore {
    PoseScore {
        focal_px: 600.0,
        geometry_accepted: true,
        holdout_count: 16,
        holdout_inlier_fraction: 0.875,
        holdout_median_px: 1.2,
        image_support_fraction: 0.35,
        inliers: 48,
        pairs: 64,
        positive_depth_fraction: 1.0,
        train_inliers: 40,
        train_rms_px: 1.1,
    }
}

#[test]
fn top_up_tracks_evicts_out_of_frustum_cap_and_deduplicates_enu_neighbors()
-> Result<(), Box<dyn Error>> {
    let img = RgbImage::from_pixel(256, 256, Rgb([110, 130, 95]));
    let surface = Surface::synthetic(41.0, -75.0, 100.0, img);
    let eye = surface.local(41.0, -75.0, 600.0);
    let rot = surface.camera_rotation(41.0, -75.0, 0.0, 88.0, 0.0);
    let k = Matrix3::new(600.0, 0.0, 399.5, 0.0, 600.0, 299.5, 0.0, 0.0, 1.0);
    let cand = PoseCandidate {
        eye,
        rotation: rot,
        k,
        size: [800, 600],
        points: Vec::new(),
        pixels: Vec::new(),
        inliers: vec![true; 48],
        score: dummy_score(),
    };

    // Fill active tracks to the 2048 cap with out-of-frustum landmarks 20 km East.
    let mut tracks = ActiveTrackSet {
        world_points_enu: (0..2048)
            .map(|i| [20_000.0 + f64::from(i) * 5.0, 0.0, 100.0])
            .collect(),
        sign_bits: vec![7; 2048],
        kashin_i8: vec![[3; 64]; 2048],
    };
    assert_eq!(tracks.len(), 2048);

    // Build two query keypoints separated by < 1.5 m on the ground (0.2 px apart at 500 m AGL)
    // and a third keypoint 60 px away (> 40 m on the ground).
    let mut qry = SparseFeatures::empty();
    for (idx, px) in [[400.0, 300.0], [400.2, 300.0], [460.0, 340.0]]
        .into_iter()
        .enumerate()
    {
        qry.pixels.push(px);
        qry.model_pixels.push([px[0] as f32, px[1] as f32]);
        qry.scores.push(0.9);
        let mut desc = [0.0_f32; 64];
        desc[idx] = 1.0;
        qry.descriptors.extend_from_slice(&desc);
    }
    let projector = HadamardProjector::new(DEFAULT_KASHIN_SEED);
    let prepared = PreparedKashinQuery::from_features(&projector, &qry)?;

    top_up_tracks(&mut tracks, &surface, &prepared, &qry, &cand, [800, 600]);
    assert_eq!(tracks.len(), 2);

    let base = tracks.world_points_enu[0];
    let near_and_far = ActiveTrackSet {
        world_points_enu: vec![
            [base[0] + 0.8, base[1] + 0.5, base[2]],
            [base[0] + 4.0, base[1], base[2]],
        ],
        sign_bits: vec![88, 99],
        kashin_i8: vec![[4; 64], [5; 64]],
    };
    extend_dedup_enu(&mut tracks, &near_and_far, 1.5, 2048);
    assert_eq!(tracks.len(), 3);
    assert_eq!(tracks.sign_bits[2], 99);
    Ok(())
}
