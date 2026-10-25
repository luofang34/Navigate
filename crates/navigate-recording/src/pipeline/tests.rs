//! The same decoded observations drive live ingestion and recording reconstruction.

use super::*;
use crate::{Camera, Pose, Reference};
use nalgebra::{Vector2, Vector3};
use navigate_visual::{CameraPose, PixelMatch, ReferenceView};

fn config() -> Configuration {
    Configuration {
        camera: Camera {
            size: [320, 240],
            focal: [260.0, 260.0],
            principal: [159.5, 119.5],
        },
        anchor: [40.0, -74.0],
        map_release: "fixture".into(),
        map_sha256: "a".repeat(64),
        prior: Pose {
            position: [0.0, 0.0, 100.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
        },
        position_radius_m: 500.0,
        map_sigma_m: [3.0, 5.0],
        object_height_m: 10.0,
        capture_error_ns: 10_000,
        clock_description: "synthetic capture clock".into(),
        matcher: "fixture correspondences".into(),
    }
}

fn frame(sequence: u64, pose: CameraPose) -> Frame {
    let camera = config().camera.model();
    let image = image::GrayImage::from_fn(camera.width, camera.height, |x, y| {
        let world = camera.unproject(
            &pose,
            Vector2::new(f64::from(x), f64::from(y)),
            pose.position.z,
        );
        let value = (world.x.mul_add(3.0, world.y * 5.0))
            .sin()
            .mul_add(100.0, 128.0);
        image::Luma([value as u8])
    });
    Frame {
        camera,
        stamp: FrameStamp {
            sequence,
            capture_time_ns: sequence * 1_000_000_000,
        },
        image,
    }
}

fn reference() -> ReferenceView {
    let cfg = config();
    let pose = cfg.prior.camera().expect("pose");
    ReferenceView {
        map: MapRevision {
            release_id: cfg.map_release,
            manifest_sha256: cfg.map_sha256,
        },
        frame: LocalFrame::anchor_mercator(cfg.anchor[0], cfg.anchor[1]).expect("frame"),
        pose,
        image: frame(0, pose).image,
        depth_m: vec![100.0; 320 * 240],
    }
}

fn pairs(pose: CameraPose) -> Vec<PixelMatch> {
    let camera = config().camera.model();
    let reference = config().prior.camera().expect("pose");
    (-3..=3)
        .flat_map(|y| {
            (-4..=4).map(move |x| Vector3::new(f64::from(x) * 10.0, f64::from(y) * 10.0, 0.0))
        })
        .map(|world| PixelMatch {
            reference: camera
                .project(&reference, world)
                .expect("visible reference"),
            query: camera.project(&pose, world).expect("visible query"),
        })
        .collect()
}

fn input(sequence: u64, position: [f64; 3], anchored: bool) -> VisualObservation {
    let pose = Pose {
        position,
        rotation: [0.0, 0.0, 0.0, 1.0],
    }
    .camera()
    .expect("pose");
    VisualObservation {
        frame: frame(sequence, pose),
        media_ns: sequence * 1_000_000_000,
        terrain_m: 0.0,
        map_reference: anchored.then(reference),
        map_matches: if anchored { pairs(pose) } else { Vec::new() },
        tracking_reference: (!anchored).then(reference),
        tracking_matches: if anchored { Vec::new() } else { pairs(pose) },
        revisits: Vec::new(),
    }
}

#[test]
fn map_anchor_and_tracking_change_camera_agl_without_fabricating_body_telemetry() {
    let mut processor = Processor::new(config()).expect("processor");
    let first = processor
        .process(input(0, [0.0, 0.0, 100.0], true))
        .expect("anchor");
    assert!(first.pose.is_some(), "{:?}", first.diagnostics);
    assert!(first.map_observation_accepted);
    let next = processor
        .process(input(1, [10.0, 0.0, 120.0], false))
        .expect("tracking");
    assert!(!next.map_observation_accepted);
    assert!((next.camera_agl.expect("AGL")[0] - 120.0).abs() < 0.1);
    assert!((next.pose.expect("pose").position[0] - 10.0).abs() < 0.1);
    let mut gap = input(2, [10.0, 0.0, 120.0], false);
    gap.tracking_matches.clear();
    let gap = processor.process(gap).expect("gap");
    assert!(gap.pose.is_none());
    assert!(gap.camera_agl.is_none());
    let recovered = processor
        .process(input(3, [20.0, 0.0, 110.0], true))
        .expect("recovery");
    assert!(recovered.pose.is_some(), "{:?}", recovered.diagnostics);
}

#[test]
fn stored_pixels_and_correspondences_replay_the_live_processing_path() {
    let directory = tempfile::tempdir().expect("directory");
    let input = input(0, [0.0, 0.0, 100.0], true);
    input
        .frame
        .image
        .save(directory.path().join("frame.png"))
        .expect("image");
    reference()
        .image
        .save(directory.path().join("reference.png"))
        .expect("reference");
    let bytes: Vec<_> = reference()
        .depth_m
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    std::fs::write(directory.path().join("depth.bin"), bytes).expect("depth");
    let record = Observation {
        sequence: 0,
        capture_ns: 0,
        media_ns: 0,
        image: "frame.png".into(),
        terrain_m: 0.0,
        map_reference: Some(Reference {
            image: "reference.png".into(),
            depth: "depth.bin".into(),
            pose: config().prior,
        }),
        map_matches: input
            .map_matches
            .iter()
            .map(|m| [m.reference.into(), m.query.into()])
            .collect(),
        tracking_reference: None,
        tracking_matches: Vec::new(),
        revisits: Vec::new(),
    };
    let mut live = Processor::new(config()).expect("live");
    let mut replay = Processor::new(config()).expect("replay");
    let a = live.process(input).expect("live result");
    let b = replay
        .process_blocking(directory.path(), record)
        .expect("replay result");
    assert_eq!(
        serde_json::to_value(a).expect("JSON"),
        serde_json::to_value(b).expect("JSON")
    );
}

#[test]
fn decoded_tracking_submits_independent_ground_evidence_and_reports_use_bounds() {
    let mut processor = Processor::new(config()).expect("processor");
    processor
        .process(input(0, [0.0, 0.0, 100.0], true))
        .expect("anchor");
    let result = processor
        .process(input(1, [10.0, 0.0, 100.0], false))
        .expect("tracking");
    assert!(
        result
            .diagnostics
            .iter()
            .any(|message| message.starts_with("Ground plane: Accepted")),
        "{:?}",
        result.diagnostics
    );
    let support = result.support.expect("support");
    assert!(support.recent_map_support);
    assert!(support.tilt_bound_deg.is_some());
    let candidates = processor.relocation_candidates().expect("candidates");
    assert!(!candidates.is_empty());
    assert!(
        candidates
            .iter()
            .all(|pose| pose.rotation.iter().all(|x| x.is_finite()))
    );
}

/// Depth of flat ground at height zero seen straight down from `pose`.
fn view(pose: CameraPose) -> ReferenceView {
    ReferenceView {
        pose,
        depth_m: vec![pose.position.z as f32; 320 * 240],
        ..reference()
    }
}

/// Ground points seen by both cameras, from `a` (reference) to `b` (query).
fn between(a: CameraPose, b: CameraPose) -> Vec<PixelMatch> {
    let camera = config().camera.model();
    let middle = (a.position + b.position) * 0.5;
    (-3..=3)
        .flat_map(|y| (-4..=4).map(move |x| (f64::from(x) * 10.0, f64::from(y) * 10.0)))
        .filter_map(|(x, y)| {
            let world = Vector3::new(middle.x + x, middle.y + y, 0.0);
            Some(PixelMatch {
                reference: camera.project(&a, world)?,
                query: camera.project(&b, world)?,
            })
        })
        .collect()
}

/// Out 240 m east and back, one frame per second.
fn out_and_back(sequence: u64) -> CameraPose {
    let i = sequence as f64;
    let (x, y) = if sequence <= 24 {
        (10.0 * i, 0.0)
    } else {
        (10.0 * (48.0 - i) + 3.0, 2.0)
    };
    Pose {
        position: [x, y, 100.0],
        rotation: [0.0, 0.0, 0.0, 1.0],
    }
    .camera()
    .expect("pose")
}

fn tracked(sequence: u64) -> VisualObservation {
    let (from, to) = (out_and_back(sequence - 1), out_and_back(sequence));
    VisualObservation {
        tracking_reference: Some(view(from)),
        tracking_matches: between(from, to),
        ..input(sequence, to.position.into(), false)
    }
}

fn flown() -> Processor {
    let mut processor = Processor::new(config()).expect("processor");
    processor
        .process(input(0, [0.0, 0.0, 100.0], true))
        .expect("anchor");
    for sequence in 1..=48 {
        processor.process(tracked(sequence)).expect("tracking");
    }
    processor
}

#[test]
fn a_return_pass_closes_on_an_earlier_keyframe_through_the_processing_path() {
    let mut processor = flown();
    let requests = processor.revisit_candidates().expect("candidates");
    let request = requests
        .iter()
        .min_by_key(|r| r.earlier)
        .expect("the return pass sees the outbound pass again");
    assert_eq!(request.later, 48);
    let earlier = out_and_back(request.earlier);
    let evidence = |later| RevisitInput {
        later,
        earlier: frame(request.earlier, earlier),
        earlier_surface: view(request.earlier_pose.camera().expect("pose")),
        matches: between(earlier, out_and_back(48)),
    };
    let mut stale = tracked(49);
    stale.revisits = vec![evidence(47)];
    let refused = flown()
        .process(stale)
        .expect("stale evidence is not an error");
    assert!(
        refused
            .diagnostics
            .iter()
            .any(|d| d.ends_with("ignored: not the preceding frame")),
        "{:?}",
        refused.diagnostics
    );
    let mut next = tracked(49);
    next.revisits = vec![evidence(48)];
    let result = processor.process(next).expect("closure");
    let accepted = format!("Revisit of 48 with {}: Accepted", request.earlier);
    assert!(
        result.diagnostics.iter().any(|d| d.starts_with(&accepted)),
        "{:?}",
        result.diagnostics
    );
}

#[test]
fn frame_records_are_kept_only_while_the_session_retains_their_frames() {
    let mut session = SessionConfig::standard();
    session.limits.max_frames = 4;
    let mut processor = Processor::with_session(config(), session).expect("processor");
    processor
        .process(input(0, [0.0, 0.0, 100.0], true))
        .expect("anchor");
    for sequence in 1..=12 {
        processor.process(tracked(sequence)).expect("tracking");
    }
    let kept: Vec<u64> = processor
        .results()
        .expect("every kept frame resolves")
        .iter()
        .map(|frame| frame.sequence)
        .collect();
    assert_eq!(kept, [9, 10, 11, 12]);
    assert!(
        processor.update_terrain(8, 0.0).is_err(),
        "dropped with its frame"
    );
}
