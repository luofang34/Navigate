//! Unit and integration tests for `navigate-visual-pipeline`.

use image::{Rgb, RgbImage};
use nalgebra::{Matrix3, Vector3};
use navigate_visual_pipeline::{
    PriorTrack, Surface, capture_anchor, compose_ground_mosaic, compose_survey_debrief,
    export_preview, export_publication, fit_pose_ransac, freeze_anchors, group_flights,
    horizon_attitude, interpolate_flight_priors, metadata::PhotoClaims, metadata::PositionClaim,
    movie_duration_s, pose::load_candidate_pose, pose::save_candidate_npz, prepare_photo_testset,
    seed_live_prior, spatial_holdout, suitability, util::parse_iso8601_ms,
    util::unix_ms_to_iso8601, util::write_json, video_ops::dense_rotations,
    video_ops::interpolate_eyes, video_ops::kabsch_3d, video_ops::prepare_navigate_debrief,
    video_ops::relative_rotation_ransac, video_ops::shared_focal, video_ops::slerp_mat3,
};
use serde_json::json;
use std::collections::BTreeMap;

fn make_mvhd_mp4(creation_1904_s: u32, timescale: u32, duration: u32) -> Vec<u8> {
    let mut mvhd_body = vec![0u8; 100];
    mvhd_body[0] = 0; // version 0
    mvhd_body[4..8].copy_from_slice(&creation_1904_s.to_be_bytes());
    mvhd_body[12..16].copy_from_slice(&timescale.to_be_bytes());
    mvhd_body[16..20].copy_from_slice(&duration.to_be_bytes());

    let mvhd_size = (8 + mvhd_body.len()) as u32;
    let moov_size = 8 + mvhd_size;
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&moov_size.to_be_bytes());
    bytes.extend_from_slice(b"moov");
    bytes.extend_from_slice(&mvhd_size.to_be_bytes());
    bytes.extend_from_slice(b"mvhd");
    bytes.extend_from_slice(&mvhd_body);
    bytes
}

#[test]
fn quicktime_clock_and_live_prior_seed_work() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let mp4_path = tmp.path().join("flight.mp4");
    // 2024-01-01T00:00:00Z = 1_704_067_200 unix + 2_082_844_800 = 3_786_912_000
    std::fs::write(&mp4_path, make_mvhd_mp4(3_786_912_000, 1000, 60_000)).expect("write mp4");

    let anchor = capture_anchor(&mp4_path)
        .expect("parse clock")
        .expect("nonzero clock");
    assert_eq!(anchor.unix_ms, 1_704_067_200_000);
    let dur = movie_duration_s(&mp4_path)
        .expect("parse duration")
        .expect("duration");
    assert!((dur - 60.0).abs() < 1e-6);

    let iso = unix_ms_to_iso8601(anchor.unix_ms);
    assert_eq!(parse_iso8601_ms(&iso), Some(anchor.unix_ms));

    let replay_dir = tmp.path().join("live");
    std::fs::create_dir_all(replay_dir.join("frames")).expect("mkdir");
    RgbImage::from_pixel(32, 32, Rgb([120, 130, 110]))
        .save(replay_dir.join("frames/0001.jpg"))
        .expect("save frame");
    write_json(
        &replay_dir.join("replay.json"),
        &json!({
            "version": 1,
            "agl_m": 450.0,
            "heading_deg": 90.0,
            "time_anchor": { "at": 0, "unix_ms": 1_704_067_195_000_i64, "source_scale": "utc" },
            "frames": [{
                "time_s": 6.0,
                "image": "frames/0001.jpg",
                "position": [-75.15, 40.98],
                "corners": [[-75.14, 40.99], [-75.14, 40.97], [-75.16, 40.97], [-75.16, 40.99]]
            }]
        }),
    )
    .expect("write replay");

    let map_manifest = tmp.path().join("map.json");
    write_json(&map_manifest, &json!({ "anchor_lat_lon": [40.98, -75.15] })).expect("write map");

    let (acq, summary) = seed_live_prior(
        &replay_dir,
        &mp4_path,
        &map_manifest,
        0.0,
        [1280, 720],
        70.0,
        150.0,
        20.0,
    )
    .expect("seed live prior");
    assert_eq!(summary["registered_frames"], 1);
    let pt = PriorTrack::from_acquisition(&acq)
        .expect("valid track")
        .expect("present");
    assert_eq!(pt.frames.len(), 1);
    assert!(pt.local_pose(1.0, [40.98, -75.15], 5.0).is_some());
}

#[test]
fn photo_testset_and_antimeridian_slerp_priors_work() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let src = tmp.path().join("raw");
    let dst = tmp.path().join("dataset");
    std::fs::create_dir_all(&src).expect("mkdir");

    for (i, c) in [20u8, 80, 160].into_iter().enumerate() {
        let mut img = RgbImage::new(320, 240);
        for (x, y, p) in img.enumerate_pixels_mut() {
            *p = Rgb([c.wrapping_add(x as u8), c.wrapping_add(y as u8), c]);
        }
        img.save(src.join(format!("img_{i}.png"))).expect("save");
    }

    // Output inside source must be rejected.
    assert!(prepare_photo_testset(&src, &src.join("nested"), 200).is_err());

    let res = prepare_photo_testset(&src, &dst, 200).expect("prepare testset");
    assert_eq!(res["photos"], 3);
    assert!(dst.join("contact-sheet-01.jpg").is_file());

    let mk_claim = |t: &str, pos: Option<(f64, f64)>| PhotoClaims {
        capture_time: Some(t.into()),
        camera_model: Some("TestCam".into()),
        position: pos.map(|(lat, lon)| PositionClaim {
            latitude: lat,
            longitude: lon,
            reported_horizontal_error_m: Some(5.0),
            altitude_m: Some(1200.0),
            unsigned_altitude_m: Some(1200.0),
            altitude_reference_text: None,
            altitude_datum: "unverified".into(),
        }),
        ..PhotoClaims::default()
    };

    let groups = group_flights(&[
        (
            "q-1".into(),
            "".into(),
            mk_claim("2025-05-01T10:00:00+00:00", Some((10.0, 179.8))),
        ),
        (
            "q-2".into(),
            "".into(),
            mk_claim("2025-05-01T10:05:00+00:00", None),
        ),
        (
            "q-3".into(),
            "".into(),
            mk_claim("2025-05-01T10:10:00+00:00", Some((10.0, -179.8))),
        ),
    ]);
    assert_eq!(groups.len(), 1);

    // Inject antimeridian flight into dataset to verify SLERP interpolation.
    let mut profile: serde_json::Value =
        navigate_visual_pipeline::util::read_json(&dst.join("profiles/with-prior.json"))
            .expect("read profile");
    let cases = profile["cases"].as_array_mut().expect("cases");
    let ids: Vec<String> = cases
        .iter()
        .map(|c| c["id"].as_str().expect("id").to_owned())
        .collect();
    cases[0]["prior"] =
        serde_json::to_value(mk_claim("2025-05-01T10:00:00+00:00", Some((10.0, 179.8))))
            .expect("json");
    cases[1]["prior"] =
        serde_json::to_value(mk_claim("2025-05-01T10:05:00+00:00", None)).expect("json");
    cases[2]["prior"] =
        serde_json::to_value(mk_claim("2025-05-01T10:10:00+00:00", Some((10.0, -179.8))))
            .expect("json");
    write_json(&dst.join("profiles/with-prior.json"), &profile).expect("write profile");
    write_json(
        &dst.join("manifest.json"),
        &json!({
            "format": "photo-terrain-testset",
            "version": 1,
            "capture_groups": [{ "basis": "test", "photos": ids }],
        }),
    )
    .expect("write manifest");

    let filled = interpolate_flight_priors(&dst).expect("interpolate");
    assert_eq!(filled, 1);
    let updated: serde_json::Value =
        navigate_visual_pipeline::util::read_json(&dst.join("profiles/with-prior.json"))
            .expect("read updated");
    let lon_mid = updated["cases"][1]["prior"]["position"]["longitude"]
        .as_f64()
        .expect("lon");
    assert!(
        lon_mid.abs() > 179.5,
        "antimeridian SLERP crossed wrong way: {lon_mid}"
    );
}

#[test]
fn surface_raycast_horizon_and_pnp_holdout_work() {
    let mut tile = RgbImage::new(256, 256);
    for (x, y, p) in tile.enumerate_pixels_mut() {
        *p = Rgb([x as u8, y as u8, 128]);
    }
    let surface = Surface::synthetic(41.0, -75.2, 200.0, tile);
    let enu = surface.local(41.005, -75.195, 1100.0);
    let geo = surface.geodetic(enu);
    assert!((geo[0] - 41.005).abs() < 1e-4);
    assert!((geo[1] + 75.195).abs() < 1e-4);
    assert!((geo[2] - 1100.0).abs() < 0.05);

    let k = Matrix3::new(120.0, 0.0, 63.5, 0.0, 120.0, 47.5, 0.0, 0.0, 1.0);
    let rot = surface.camera_rotation(41.0, -75.2, 45.0, 60.0, 0.0);
    let eye = surface.local(41.0, -75.2, 1200.0);
    let (_, depth, pts) = surface.render(eye, &rot, &k, [128, 96]);

    let mut world_pts = Vec::new();
    let mut img_px = Vec::new();
    for y in (8..88).step_by(8) {
        for x in (8..120).step_by(8) {
            let idx = (y * 128 + x) as usize;
            if depth[idx] > 1.0 {
                world_pts.push(pts[idx]);
                img_px.push([f64::from(x), f64::from(y)]);
            }
        }
    }
    assert!(world_pts.len() >= 60);
    let held = spatial_holdout(&img_px, [128, 96]);
    assert!(held.iter().any(|&h| h) && held.iter().any(|&h| !h));

    let cand = fit_pose_ransac(&world_pts, &img_px, &k, [128, 96], None).expect("fit pose");
    assert!(cand.score.geometry_accepted);
    assert!(cand.score.holdout_median_px < 1.0);
    let dist = (Vector3::from(cand.eye) - Vector3::from(eye)).norm();
    assert!(dist < 2.0, "PnP eye error {dist} m");

    let tmp = tempfile::tempdir().expect("tempdir");
    let npz_path = tmp.path().join("candidate.npz");
    save_candidate_npz(&npz_path, &cand).expect("save npz");
    let (loaded_eye, _, _, sz) = load_candidate_pose(&npz_path).expect("load npz");
    assert_eq!(sz, [128, 96]);
    assert!((loaded_eye[2] - cand.eye[2]).abs() < 1e-6);

    // Horizon attitude on synthetic 4-class label map (sky=0 top half, ground=1 bottom half).
    let mut sem = vec![1u8; 128 * 96];
    for y in 0..48 {
        for x in 0..128 {
            sem[y * 128 + x] = 0;
        }
    }
    let (down, roll) = horizon_attitude(&sem, 128, 96, &k).expect("horizon");
    assert!(down.abs() < 2.0 && roll.abs() < 2.0);
}

#[test]
fn mosaic_rotations_preview_and_navigate_debrief_work() {
    let good_quad = [
        [-75.20, 41.01],
        [-75.19, 41.01],
        [-75.19, 41.00],
        [-75.20, 41.00],
    ];
    assert!(suitability(&good_quad, Some(0.8)).is_ok());
    assert!(suitability(&good_quad, Some(0.1)).is_err());

    let tmp = tempfile::tempdir().expect("tempdir");
    let mosaic_png = tmp.path().join("mosaic.png");
    let frame_img = RgbImage::from_pixel(64, 64, Rgb([90, 140, 80]));
    let m_res = compose_ground_mosaic(&[(frame_img, good_quad)], 128, &mosaic_png).expect("mosaic");
    assert_eq!(m_res["frames_composited"], 1);
    assert!(mosaic_png.is_file());

    // 3D rotation Kabsch + RANSAC + SLERP + dense closure.
    let k = Matrix3::new(200.0, 0.0, 100.0, 0.0, 200.0, 100.0, 0.0, 0.0, 1.0);
    let r_true = *nalgebra::Rotation3::from_euler_angles(0.05, -0.08, 0.12).matrix();
    let rays_a = [
        [0.0, 0.0, 1.0],
        [0.2, 0.1, 0.97],
        [-0.2, 0.15, 0.96],
        [0.1, -0.25, 0.96],
        [-0.15, -0.1, 0.98],
    ];
    let rays_b: Vec<[f64; 3]> = rays_a
        .iter()
        .map(|a| {
            let v = r_true * Vector3::from(*a);
            [v.x, v.y, v.z]
        })
        .collect();
    let r_est = kabsch_3d(&rays_a, &rays_b).expect("kabsch");
    assert!((r_est - r_true).norm() < 1e-6);

    let px_a: Vec<[f64; 2]> = rays_a
        .iter()
        .map(|v| [200.0 * (v[0] / v[2]) + 100.0, 200.0 * (v[1] / v[2]) + 100.0])
        .collect();
    let px_b: Vec<[f64; 2]> = rays_b
        .iter()
        .map(|v| [200.0 * (v[0] / v[2]) + 100.0, 200.0 * (v[1] / v[2]) + 100.0])
        .collect();
    let (r_ran, inl) = relative_rotation_ransac(&px_a, &px_b, &k, 2.0).expect("ransac");
    assert!(inl.iter().all(|&b| b));
    assert!((r_ran - r_true).norm() < 1e-5);

    let mid = slerp_mat3(&Matrix3::identity(), &r_true, 0.5);
    assert!((mid * mid - r_true).norm() < 1e-6);
    let mut rot_anchors = BTreeMap::new();
    rot_anchors.insert(0, Matrix3::identity());
    rot_anchors.insert(2, r_true);
    let dense = dense_rotations(3, &[mid, mid], &rot_anchors);
    assert_eq!(dense.len(), 3);

    let mut eye_anchors = BTreeMap::new();
    eye_anchors.insert(0, [0.0, 0.0, 100.0]);
    eye_anchors.insert(2, [20.0, 10.0, 120.0]);
    let eyes = interpolate_eyes(3, &eye_anchors);
    assert!((eyes[1][0] - 10.0).abs() < 1e-9);
    assert_eq!(shared_focal(&[1000.0, 1020.0, 1500.0]), Some(1020.0));

    // Preview + freeze + publication + survey + Navigate processor debrief.
    let cat_dir = tmp.path().join("bundle/cat");
    std::fs::create_dir_all(&cat_dir).expect("mkdir");
    write_json(
        &cat_dir.join("catalog.json"),
        &json!({
            "format": "photo-reference-catalog",
            "version": 1,
            "complete": true,
            "configuration": { "bbox": [-75.3, 40.9, -75.1, 41.1], "imagery_zoom": 14, "dem_zoom": 12 },
            "tiles": []
        }),
    )
    .expect("write catalog");
    let run_dir = tmp.path().join("bundle/run1");
    write_json(
        &run_dir.join("run.json"),
        &json!({ "catalog_sha256": "a".repeat(64) }),
    )
    .expect("run");
    write_json(
        &run_dir.join("q-1.json"),
        &json!({ "id": "q-1", "status": "unresolved", "attempts": [], "candidate": null }),
    )
    .expect("q-1");

    let prev_dir = tmp.path().join("preview1");
    let prev = export_preview(
        &tmp.path().join("bundle"),
        &cat_dir.join("catalog.json"),
        std::slice::from_ref(&run_dir),
        &prev_dir,
    )
    .expect("preview");
    assert_eq!(prev["total"], 1);
    let survey = compose_survey_debrief(&[prev_dir], &tmp.path().join("survey")).expect("survey");
    assert_eq!(survey["streams"], 1);
    let frz = freeze_anchors(&[run_dir], &tmp.path().join("frozen")).expect("freeze");
    assert_eq!(frz["frozen_cases"], 0);
    let pub_res = export_publication(
        &tmp.path().join("bundle"),
        "https://example.invalid/files",
        &tmp.path().join("publication.json"),
    )
    .expect("publish");
    assert!(pub_res["files"].as_u64().expect("files") >= 2);

    // Direct Navigate recording Processor debrief test.
    let ev_dir = tmp.path().join("evidence");
    std::fs::create_dir_all(&ev_dir).expect("mkdir");
    image::GrayImage::from_pixel(64, 64, image::Luma([120u8]))
        .save(ev_dir.join("f0.png"))
        .expect("save f0");
    write_json(
        &ev_dir.join("configuration.json"),
        &json!({
            "camera": { "size": [64, 64], "focal": [55.0, 55.0], "principal": [31.5, 31.5] },
            "anchor": [41.0, -75.2],
            "map_release": "test-rel",
            "map_sha256": "b".repeat(64),
            "prior": { "position": [0.0, 0.0, 900.0], "rotation": [0.0, 0.0, 0.0, 1.0] },
            "position_radius_m": 500.0,
            "map_sigma_m": [5.0, 10.0],
            "object_height_m": 20.0,
            "capture_error_ns": 1_000_000,
            "clock_description": "unit test clock",
            "matcher": "synthetic-matcher"
        }),
    )
    .expect("write cfg");
    write_json(
        &ev_dir.join("observations.json"),
        &json!([{
            "sequence": 0,
            "capture_ns": 1_000_000_000_u64,
            "media_ns": 0_u64,
            "image": "f0.png",
            "map_reference": null,
            "map_matches": [],
            "tracking_reference": null,
            "tracking_matches": [],
            "terrain_m": 150.0,
            "revisits": []
        }]),
    )
    .expect("write obs");
    let deb = prepare_navigate_debrief(&ev_dir, &tmp.path().join("nav-debrief")).expect("debrief");
    assert_eq!(deb["frames"], 1);
}
