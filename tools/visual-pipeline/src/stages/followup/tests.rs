use super::*;
use crate::{
    metadata::{PhotoClaims, PositionClaim},
    pose::{PoseScore, save_candidate_npz},
};
use image::{Rgb, RgbImage};
use nalgebra::{Matrix3, Vector3};

fn make_case(id: &str, capture_time: &str, lat: f64, lon: f64, dir_deg: Option<f64>) -> PhotoCase {
    PhotoCase {
        id: id.into(),
        image: format!("images/{id}.png"),
        image_sha256: "0".repeat(64),
        size: [800, 600],
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
                altitude_m: Some(350.0),
                unsigned_altitude_m: Some(350.0),
                altitude_reference_text: None,
                altitude_datum: "unverified".into(),
            }),
            direction_deg: dir_deg,
            ..PhotoClaims::default()
        }),
        sequence: None,
        time_s: None,
    }
}

#[test]
fn rank_neighbor_anchors_prefers_same_flight_temporal_and_spatial_neighbors() {
    let surface = Surface::synthetic(
        39.75,
        -75.55,
        40.0,
        RgbImage::from_pixel(64, 64, Rgb([130, 140, 120])),
    );
    let query = make_case(
        "q-8000query",
        "2026-05-12T14:10:10Z",
        39.7500,
        -75.5500,
        None,
    );
    let wrong_flight_lex_first = make_case(
        "q-0001other",
        "2025-11-03T09:00:00Z",
        39.8800,
        -75.7200,
        None,
    );
    let same_flight_far_time = make_case(
        "q-9000samefar",
        "2026-05-12T14:25:00Z",
        39.7560,
        -75.5560,
        None,
    );
    let same_flight_closest = make_case(
        "q-ffffclosest",
        "2026-05-12T14:10:06Z",
        39.7502,
        -75.5501,
        None,
    );

    let cases = [
        query.clone(),
        wrong_flight_lex_first.clone(),
        same_flight_far_time.clone(),
        same_flight_closest.clone(),
    ];
    let cases_by_id: BTreeMap<&str, &PhotoCase> =
        cases.iter().map(|c| (c.id.as_str(), c)).collect();
    let mut group_map = HashMap::new();
    group_map.insert(wrong_flight_lex_first.id.clone(), 0);
    group_map.insert(query.id.clone(), 1);
    group_map.insert(same_flight_far_time.id.clone(), 1);
    group_map.insert(same_flight_closest.id.clone(), 1);

    let k = Matrix3::new(600.0, 0.0, 399.5, 0.0, 600.0, 299.5, 0.0, 0.0, 1.0);
    let rot = surface.camera_rotation(39.75, -75.55, 0.0, 45.0, 0.0);
    let mut anchors: AnchorPoseMap = BTreeMap::new();
    for c in [
        &wrong_flight_lex_first,
        &same_flight_far_time,
        &same_flight_closest,
    ] {
        let pos = c.prior.as_ref().unwrap().position.as_ref().unwrap();
        let eye = surface.local(pos.latitude, pos.longitude, 350.0);
        anchors.insert(c.id.clone(), (eye, rot, k, [800, 600]));
    }

    let ranked = rank_neighbor_anchors(&query, &cases_by_id, &group_map, &anchors, &surface);
    let ids: Vec<&str> = ranked.iter().map(|(id, _, _)| id.as_str()).collect();
    assert_eq!(
        ids,
        vec!["q-ffffclosest", "q-9000samefar", "q-0001other"],
        "same-flight temporal neighbor must rank ahead of lexicographically smaller cross-flight anchors"
    );
}

#[test]
fn build_pass_match_options_filters_unresolved_from_and_defaults_heading_step_zero() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let dataset_dir = tmp.path().join("dataset");
    let profiles_dir = dataset_dir.join("profiles");
    let pass1_dir = dataset_dir.join("runs").join("pass1");
    fs::create_dir_all(&profiles_dir).expect("mkdir profiles");
    fs::create_dir_all(&pass1_dir).expect("mkdir pass1");

    let c1 = make_case(
        "q-solved",
        "2026-05-12T14:00:00Z",
        39.75,
        -75.55,
        Some(72.0),
    );
    let c2 = make_case("q-open", "2026-05-12T14:00:05Z", 39.75, -75.55, Some(72.0));
    let profile_path = profiles_dir.join("with-prior.json");
    write_json(&profile_path, &json!({ "cases": [c1, c2] })).expect("write profile");
    write_json(
        &pass1_dir.join("q-solved.json"),
        &json!({ "id": "q-solved", "status": "candidate_requires_review" }),
    )
    .expect("write solved");
    write_json(
        &pass1_dir.join("q-open.json"),
        &json!({ "id": "q-open", "status": "unresolved" }),
    )
    .expect("write unresolved");

    let pass = PipelinePass {
        id: "pass2".into(),
        kind: "match".into(),
        reference_catalog: None,
        backend: None,
        attempts: None,
        heading_step: None,
        height_factors: None,
        parents: vec![],
        unresolved_from: vec!["pass1".into()],
    };
    let opts = build_pass_match_options(
        &dataset_dir,
        &profile_path,
        &dataset_dir.join("catalog.json"),
        &dataset_dir.join("runs").join("pass2"),
        &pass,
    )
    .expect("options");
    assert_eq!(opts.heading_step_deg, 0.0);
    let allow = opts.only_cases.expect("allow list");
    assert_eq!(allow.len(), 1);
    assert!(allow.contains("q-open"));
    assert!(!allow.contains("q-solved"));
}

#[test]
fn terrain_bundle_stage_refines_shared_npz_terrain_correspondences() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let cat_dir = tmp.path().join("cat");
    let tiles_dir = cat_dir.join("tiles");
    fs::create_dir_all(&tiles_dir).expect("mkdir tiles");
    let (tx, ty, _, _) = crate::surface::mercator_tile_coords(39.75, -75.55, 12);
    let img_path = tiles_dir.join("img.png");
    RgbImage::from_pixel(256, 256, Rgb([140, 150, 130]))
        .save(&img_path)
        .expect("save img");
    // Encode ~40 m elevation (`32768 + 40 = 32808 = 128 * 256 + 40`).
    let dem_path = tiles_dir.join("dem.png");
    RgbImage::from_pixel(256, 256, Rgb([128, 40, 0]))
        .save(&dem_path)
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
    let parent_dir = tmp.path().join("parent");
    fs::create_dir_all(&parent_dir).expect("mkdir parent");
    write_json(
        &parent_dir.join("run.json"),
        &json!({ "catalog_sha256": surface.catalog_sha256 }),
    )
    .expect("write run.json");

    let k = Matrix3::new(600.0, 0.0, 399.5, 0.0, 600.0, 299.5, 0.0, 0.0, 1.0);
    let shared_pts = [
        [-60.0, -40.0, 40.0],
        [60.0, -40.0, 40.0],
        [-60.0, 50.0, 40.0],
        [60.0, 50.0, 40.0],
        [0.0, 0.0, 40.0],
        [25.0, -15.0, 40.0],
    ];
    let cam_eyes = [
        [-80.0, -20.0, 420.0],
        [0.0, 0.0, 420.0],
        [80.0, 20.0, 420.0],
    ];
    for (idx, eye) in cam_eyes.into_iter().enumerate() {
        let id = format!("q-cam{idx}");
        let rot = surface.camera_rotation(39.75, -75.55, 0.0, 90.0, 0.0);
        let mut pixels = Vec::new();
        for w in &shared_pts {
            let rel = rot * (Vector3::from(*w) - Vector3::from(eye));
            pixels.push([
                k[(0, 0)] * (rel.x / rel.z) + k[(0, 2)],
                k[(1, 1)] * (rel.y / rel.z) + k[(1, 2)],
            ]);
        }
        let cand = PoseCandidate {
            eye,
            rotation: rot,
            k,
            size: [800, 600],
            points: shared_pts.to_vec(),
            pixels,
            inliers: vec![true; shared_pts.len()],
            score: PoseScore {
                pairs: shared_pts.len(),
                inliers: shared_pts.len(),
                train_inliers: 4,
                train_rms_px: 0.5,
                holdout_count: 2,
                holdout_inlier_fraction: 1.0,
                holdout_median_px: 0.5,
                image_support_fraction: 0.4,
                positive_depth_fraction: 1.0,
                focal_px: 600.0,
                geometry_accepted: true,
            },
        };
        let cdir = parent_dir.join(&id);
        fs::create_dir_all(&cdir).expect("mkdir case");
        save_candidate_npz(&cdir.join("candidate.npz"), &cand).expect("save npz");
        let [lat, lon, alt] = surface.geodetic(eye);
        write_json(
            &parent_dir.join(format!("{id}.json")),
            &json!({
                "id": id,
                "status": "candidate_requires_review",
                "candidate": { "latitude": lat, "longitude": lon, "altitude_m": alt }
            }),
        )
        .expect("write case report");
    }

    let out_dir = tmp.path().join("bundle_out");
    let res = run_terrain_bundle_stage(&cat_path, &[parent_dir], &out_dir).expect("bundle stage");
    assert_eq!(res["cameras"], 3);
    assert_eq!(res["refined_cameras"], 1);
    assert!(res["points"].as_u64().unwrap_or(0) >= 6);
    assert!(res["refinement"].is_object());
}

fn write_single_tile_catalog(dir: &Path, center_lat: f64, center_lon: f64) -> PathBuf {
    let tiles_dir = dir.join("tiles");
    fs::create_dir_all(&tiles_dir).expect("mkdir tiles");
    let (tx, ty, _, _) = crate::surface::mercator_tile_coords(center_lat, center_lon, 12);
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
    let cat_path = dir.join("catalog.json");
    write_json(
        &cat_path,
        &json!({
            "format": "photo-reference-catalog",
            "configuration": {
                "bbox": [center_lon - 0.05, center_lat - 0.05, center_lon + 0.05, center_lat + 0.05],
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
    cat_path
}

#[test]
fn cross_catalog_bundle_and_preview_reanchor_enu_points_and_globe_view() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let sub_cat_path = write_single_tile_catalog(&tmp.path().join("sub_cat"), 39.85, -75.40);
    let region_cat_path = write_single_tile_catalog(&tmp.path().join("region_cat"), 39.75, -75.55);

    let sub_surface = Surface::from_catalog(&sub_cat_path).expect("load sub catalog");
    let parent_dir = tmp.path().join("sub_run");
    fs::create_dir_all(&parent_dir).expect("mkdir sub_run");
    write_json(
        &parent_dir.join("run.json"),
        &json!({ "catalog_sha256": sub_surface.catalog_sha256 }),
    )
    .expect("write run.json");

    let k = Matrix3::new(600.0, 0.0, 399.5, 0.0, 600.0, 299.5, 0.0, 0.0, 1.0);
    let shared_pts = [
        [-60.0, -40.0, 40.0],
        [60.0, -40.0, 40.0],
        [-60.0, 50.0, 40.0],
        [60.0, 50.0, 40.0],
        [0.0, 0.0, 40.0],
        [25.0, -15.0, 40.0],
    ];
    let cam_eyes = [
        [-70.0, -15.0, 420.0],
        [0.0, 0.0, 420.0],
        [70.0, 15.0, 420.0],
    ];
    for (idx, eye) in cam_eyes.into_iter().enumerate() {
        let id = format!("q-sub{idx}");
        let rot = sub_surface.camera_rotation(39.85, -75.40, 0.0, 90.0, 0.0);
        let pixels: Vec<[f64; 2]> = shared_pts
            .iter()
            .map(|w| {
                let rel = rot * (Vector3::from(*w) - Vector3::from(eye));
                [
                    k[(0, 0)] * (rel.x / rel.z) + k[(0, 2)],
                    k[(1, 1)] * (rel.y / rel.z) + k[(1, 2)],
                ]
            })
            .collect();
        let cand = PoseCandidate {
            eye,
            rotation: rot,
            k,
            size: [800, 600],
            points: shared_pts.to_vec(),
            pixels,
            inliers: vec![true; shared_pts.len()],
            score: PoseScore {
                pairs: shared_pts.len(),
                inliers: shared_pts.len(),
                train_inliers: 4,
                train_rms_px: 0.5,
                holdout_count: 2,
                holdout_inlier_fraction: 1.0,
                holdout_median_px: 0.5,
                image_support_fraction: 0.4,
                positive_depth_fraction: 1.0,
                focal_px: 600.0,
                geometry_accepted: true,
            },
        };
        let cdir = parent_dir.join(&id);
        fs::create_dir_all(&cdir).expect("mkdir case");
        save_candidate_npz(&cdir.join("candidate.npz"), &cand).expect("save npz");
        let [lat, lon, alt] = sub_surface.geodetic(eye);
        write_json(
            &parent_dir.join(format!("{id}.json")),
            &json!({
                "id": id,
                "status": "candidate_requires_review",
                "candidate": { "latitude": lat, "longitude": lon, "altitude_m": alt }
            }),
        )
        .expect("write case report");
    }

    let bundle_out = tmp.path().join("region_bundle");
    let res = run_terrain_bundle_stage(
        &region_cat_path,
        std::slice::from_ref(&parent_dir),
        &bundle_out,
    )
    .expect("cross-catalog bundle stage");
    assert_eq!(res["cameras"], 3);
    assert_eq!(res["refined_cameras"], 1);
    assert!(res["points"].as_u64().unwrap_or(0) >= 6);
    let init_cost = res["refinement"]["initial_cost"]
        .as_f64()
        .expect("initial_cost");
    assert!(
        init_cost < 500.0,
        "cross-catalog ENU point shift must avoid multi-kilometre reprojection blowup, got {init_cost}"
    );

    let preview_out = tmp.path().join("region_preview");
    crate::preview::export_preview(
        tmp.path(),
        &region_cat_path,
        std::slice::from_ref(&parent_dir),
        &preview_out,
    )
    .expect("cross-catalog export_preview");
    let debrief: Value = read_json(&preview_out.join("debrief.json")).expect("read debrief.json");
    let items = debrief["items"].as_array().expect("items");
    assert_eq!(items.len(), 3);
    for item in items {
        let cand_lat = item["candidate"]["latitude"].as_f64().expect("cand lat");
        let cand_lon = item["candidate"]["longitude"].as_f64().expect("cand lon");
        let gv_lat = item["globe_view"]["latitude"].as_f64().expect("gv lat");
        let gv_lon = item["globe_view"]["longitude"].as_f64().expect("gv lon");
        assert!(
            (gv_lat - cand_lat).abs() < 1e-4 && (gv_lon - cand_lon).abs() < 1e-4,
            "preview globe_view ({gv_lat}, {gv_lon}) must match cross-catalog candidate ({cand_lat}, {cand_lon})"
        );
    }
}
