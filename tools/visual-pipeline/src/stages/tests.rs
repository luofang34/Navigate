use super::*;
use crate::{
    metadata::{PhotoClaims, PositionClaim},
    pose::PoseScore,
};
use image::Rgb;
use nalgebra::Vector3;

fn make_case(id: &str, capture_time: &str, lat: f64, lon: f64) -> PhotoCase {
    PhotoCase {
        id: id.into(),
        image: format!("images/{id}.png"),
        image_sha256: "0".repeat(64),
        size: [320, 240],
        valid_mask: String::new(),
        mask_sha256: String::new(),
        mask_status: "full_image_unmasked".into(),
        semantic_labels: None,
        semantic_sha256: None,
        prior: Some(PhotoClaims {
            capture_time: Some(capture_time.into()),
            position: Some(PositionClaim {
                latitude: lat,
                longitude: lon,
                reported_horizontal_error_m: None,
                altitude_m: Some(390.0),
                unsigned_altitude_m: Some(390.0),
                altitude_reference_text: None,
                altitude_datum: "unverified".into(),
            }),
            direction_deg: Some(0.0),
            focal_35mm_mm: Some(28.0),
            ..PhotoClaims::default()
        }),
        sequence: None,
        time_s: None,
    }
}

#[test]
fn match_track_and_refine_stages_execute_with_stage_terrain_renderer() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let dataset_dir = tmp.path().join("dataset");
    let images_dir = dataset_dir.join("images");
    let profiles_dir = dataset_dir.join("profiles");
    fs::create_dir_all(&images_dir).expect("mkdir images");
    fs::create_dir_all(&profiles_dir).expect("mkdir profiles");

    let cat_dir = tmp.path().join("cat");
    let tiles_dir = cat_dir.join("tiles");
    fs::create_dir_all(&tiles_dir).expect("mkdir tiles");
    let (tx, ty, _, _) = crate::surface::mercator_tile_coords(39.75, -75.55, 12);
    let mut img = RgbImage::new(256, 256);
    for y in 0..256_u32 {
        for x in 0..256_u32 {
            let v = (((x.wrapping_mul(37) ^ y.wrapping_mul(73)).wrapping_add((x * y) % 97)) & 0xFF)
                as u8;
            img.put_pixel(
                x,
                y,
                Rgb([v, v.wrapping_add(31), 255_u8.wrapping_sub(v / 2)]),
            );
        }
    }
    img.save(tiles_dir.join("img.png")).expect("save img");
    RgbImage::from_pixel(256, 256, Rgb([128, 40, 0]))
        .save(tiles_dir.join("dem.png"))
        .expect("save dem");
    let cat_path = cat_dir.join("catalog.json");
    write_json(
        &cat_path,
        &json!({
            "format": "photo-reference-catalog",
            "configuration": {
                "bbox": [-75.60, 39.70, -75.50, 39.80],
                "imagery_zoom": 12,
                "dem_zoom": 12,
            },
            "tiles": [
                { "kind": "imagery", "xyz": [12, tx, ty], "path": "tiles/img.png", "sha256": "0", "bytes": 1, "url": "local" },
                { "kind": "dem", "xyz": [12, tx, ty], "path": "tiles/dem.png", "sha256": "0", "bytes": 1, "url": "local" },
            ]
        }),
    )
    .expect("write catalog");

    let surface = Surface::from_catalog(&cat_path).expect("load catalog");
    let c1 = make_case("q-a", "2026-05-12T14:00:00Z", 39.75, -75.55);
    let c2 = make_case("q-b", "2026-05-12T14:00:02Z", 39.7501, -75.5501);

    let k = estimate_intrinsics(&c1);
    let eye1 = surface.local(39.75, -75.55, 390.0);
    let rot1 = surface.camera_rotation(39.75, -75.55, 0.0, 90.0, 0.0);
    let (rgb1, _, _) = surface.render(eye1, &rot1, &k, c1.size);
    rgb1.save(images_dir.join("q-a.png")).expect("save q-a");
    let eye2 = surface.local(39.7501, -75.5501, 390.0);
    let (rgb2, _, _) = surface.render(eye2, &rot1, &k, c2.size);
    rgb2.save(images_dir.join("q-b.png")).expect("save q-b");

    let profile_path = profiles_dir.join("with-prior.json");
    write_json(&profile_path, &json!({ "cases": [c1, c2] })).expect("write profile");

    let mut matcher = PipelineMatcher::pyramidal();
    let match_out = dataset_dir.join("runs").join("pass_match");
    let mut only_a = BTreeSet::new();
    only_a.insert("q-a".to_owned());
    let match_opts = MatchStageOptions {
        dataset_dir: dataset_dir.clone(),
        profile_path: profile_path.clone(),
        catalog_path: cat_path.clone(),
        context_catalog: None,
        output_dir: match_out.clone(),
        attempts: 2,
        steps: 1,
        heading_step_deg: 0.0,
        height_factors: vec![1.0],
        only_cases: Some(only_a),
    };
    let m_res = run_match_stage(&match_opts, &mut matcher).expect("run_match_stage");
    assert_eq!(m_res["evaluated"], 1);

    if m_res["accepted"].as_u64().unwrap_or(0) == 0 {
        let mut pts = Vec::new();
        let mut pix = Vec::new();
        for iy in 0..6 {
            for ix in 0..6 {
                let w = [
                    -80.0 + f64::from(ix) * 32.0,
                    -60.0 + f64::from(iy) * 24.0,
                    40.0,
                ];
                let rel = rot1 * (Vector3::from(w) - Vector3::from(eye1));
                pts.push(w);
                pix.push([
                    k[(0, 0)] * (rel.x / rel.z) + k[(0, 2)],
                    k[(1, 1)] * (rel.y / rel.z) + k[(1, 2)],
                ]);
            }
        }
        let cand = PoseCandidate {
            eye: eye1,
            rotation: rot1,
            k,
            size: [320, 240],
            points: pts.clone(),
            pixels: pix,
            inliers: vec![true; pts.len()],
            score: PoseScore {
                pairs: pts.len(),
                inliers: pts.len(),
                train_inliers: 24,
                train_rms_px: 0.5,
                holdout_count: 12,
                holdout_inlier_fraction: 1.0,
                holdout_median_px: 0.5,
                image_support_fraction: 0.5,
                positive_depth_fraction: 1.0,
                focal_px: k[(0, 0)],
                geometry_accepted: true,
            },
        };
        fs::create_dir_all(match_out.join("q-a")).expect("mkdir q-a");
        save_candidate_npz(&match_out.join("q-a").join("candidate.npz"), &cand).expect("save npz");
        write_json(
            &match_out.join("q-a.json"),
            &json!({
                "id": "q-a",
                "status": "candidate_requires_review",
                "candidate": { "latitude": 39.75, "longitude": -75.55, "altitude_m": 390.0 }
            }),
        )
        .expect("write q-a.json");
    }

    let mut track_opts = match_opts.clone();
    track_opts.output_dir = dataset_dir.join("runs").join("pass_track");
    track_opts.only_cases = None;
    let t_res = run_track_stage(&track_opts, std::slice::from_ref(&match_out), &mut matcher)
        .expect("run_track_stage");
    assert!(t_res["output"].is_string());

    let mut refine_opts = match_opts;
    refine_opts.output_dir = dataset_dir.join("runs").join("pass_refine");
    refine_opts.only_cases = None;
    let r_res = run_refine_stage(&refine_opts, std::slice::from_ref(&match_out), &mut matcher)
        .expect("run_refine_stage");
    assert!(r_res["output"].is_string());
}
