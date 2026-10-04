//! Attitude observability over a long flight, through the real tracking,
//! plane, map-match, and session paths. Tracking depth comes from the bare-earth
//! model rendered at the estimated pose, as a host renders it.

use crate::scene::*;
use crate::support::*;
use nalgebra::{UnitQuaternion, Vector3};
use navigate_visual::{
    Frame, LocalizerConfig, PlaneMotionConfig, PosePrior, PoseVerifier, TrackingReference,
    plane_motion,
};
use navigate_visual_session::{
    AnchorDecision, AnchorObservation, FrameKey, GroundPlaneObservation, MapPose, Pose,
    RelativeMotion, SessionConfig, TerrainNormal, VisualSession, from_camera, to_camera,
};

const FRAMES: usize = 260;
const ANCHORS: [usize; 2] = [0, 25];
/// The map is unavailable after the anchors; from here the host retries
/// relocation every five frames.
const RELOCATION: usize = 240;
/// A map match fixes attitude only to a few degrees. The first anchors carry
/// this error about the camera axis that points along the track.
const ANCHOR_TILT_ERROR_RAD: f64 = -0.05;

struct Sample {
    view_error_deg: f64,
    position_error_m: f64,
    position_bound_m: f64,
    tilt_bound_deg: Option<f64>,
    projectable: bool,
    navigable: bool,
}

struct Report {
    samples: Vec<Option<Sample>>,
    relocated: Option<usize>,
}

fn located(session: &VisualSession, k: &FrameKey, truth: &Pose) -> Option<Sample> {
    let MapPose::Located {
        pose,
        usability,
        position_bound_m,
        tilt_bound_rad,
        ..
    } = session.pose_at(k).ok()?.map
    else {
        return None;
    };
    Some(Sample {
        view_error_deg: view_error_deg(&pose, truth),
        position_error_m: (pose.translation.vector - truth.translation.vector).norm(),
        position_bound_m,
        tilt_bound_deg: tilt_bound_rad.map(f64::to_degrees),
        projectable: usability.ground_projection,
        navigable: usability.navigation,
    })
}

fn estimate(session: &VisualSession, k: &FrameKey) -> Option<Pose> {
    match session.pose_at(k).ok()?.map {
        MapPose::Located { pose, .. } => Some(pose),
        MapPose::Unlocated(_) => None,
    }
}

/// Map match against a reference rendered at `render`, as a host does.
fn map_anchor(
    frame: &Frame,
    k: FrameKey,
    render: &Pose,
    truth: &Pose,
) -> Option<AnchorObservation> {
    let reference = dem_view(render);
    let matches = map_matches(render, truth);
    let verifier = PoseVerifier::new(LocalizerConfig::default()).ok()?;
    let prior = PosePrior {
        pose: to_camera(render),
        position_radius_m: 80.0,
        attitude_radius_rad: 0.5,
    };
    let estimate = verifier
        .verify(frame, &reference, &prior, &matches, "synthetic-map")
        .ok()?;
    Some(AnchorObservation::from_estimate(
        k,
        &estimate,
        reliability(),
    ))
}

/// One host tracking step: depth at the estimated previous pose, matches of the true scene.
fn track(
    session: &mut VisualSession,
    previous: (&FrameKey, &Frame, &Pose),
    current: (&FrameKey, &Frame, &Pose),
    ground: bool,
    seed: &mut u64,
) {
    let Some(at) = estimate(session, previous.0) else {
        return;
    };
    let reference = dem_view(&at);
    let matches = scene_matches(previous.2, current.2, seed);
    let verifier = PoseVerifier::new(LocalizerConfig::default()).unwrap();
    let surface = TrackingReference {
        observation: previous.1,
        surface: &reference,
    };
    let prior = PosePrior {
        pose: to_camera(&at),
        position_radius_m: 1_500.0,
        attitude_radius_rad: std::f64::consts::PI,
    };
    if let Ok(proposal) = verifier.track(current.1, surface, &prior, &matches, "synthetic") {
        let motion =
            RelativeMotion::from_tracking(*previous.0, *current.0, &reference.pose, &proposal);
        session.apply_motion(motion).unwrap();
    }
    if !ground {
        return;
    }
    let (Ok(plane), Some(terrain)) = (
        plane_motion(&camera(), &matches, &PlaneMotionConfig::default()),
        TerrainNormal::from_reference(&camera(), &reference),
    ) else {
        return;
    };
    let p = at.translation.vector;
    let relief = ROOF_M / (p.z - dem(p.x, p.y)).max(10.0);
    if let Some(observation) =
        GroundPlaneObservation::from_plane_motion(*previous.0, &plane, terrain, relief)
    {
        session.submit_ground_plane(observation).unwrap();
    }
}

fn relocate(session: &mut VisualSession, k: &FrameKey, frame: &Frame, truth: &Pose) -> bool {
    for candidate in session.relocation_candidates(k, None).unwrap() {
        let render = from_camera(&candidate.pose);
        if let Some(observation) = map_anchor(frame, *k, &render, truth)
            && matches!(
                session.submit_anchor(observation),
                Ok(AnchorDecision::Accepted { .. } | AnchorDecision::Relocated { .. })
            )
        {
            return true;
        }
    }
    false
}

fn anchor(session: &mut VisualSession, frame: &Frame, k: FrameKey, truth: &Pose) {
    let render = Pose::from_parts(
        (truth.translation.vector + Vector3::new(6.0, -4.0, 3.0)).into(),
        truth.rotation,
    );
    let mut observation = map_anchor(frame, k, &render, truth).unwrap();
    observation.pose.rotation *= UnitQuaternion::from_euler_angles(0.0, ANCHOR_TILT_ERROR_RAD, 0.0);
    session.submit_anchor(observation).unwrap();
}

fn fly(ground: bool) -> Report {
    let mut session = VisualSession::new(SessionConfig::standard(), calibration()).unwrap();
    let mut seed = 7_u64;
    let mut previous: Option<(FrameKey, Frame, Pose)> = None;
    let mut report = Report {
        samples: Vec::new(),
        relocated: None,
    };
    for i in 0..FRAMES {
        let (k, frame, truth_i) = (key(0, i as u64), image_frame(i as u64), truth(i));
        session.observe_frame(record(k, &frame)).unwrap();
        if let Some((pk, pf, pt)) = &previous {
            track(
                &mut session,
                (pk, pf, pt),
                (&k, &frame, &truth_i),
                ground,
                &mut seed,
            );
        }
        if ANCHORS.contains(&i) {
            anchor(&mut session, &frame, k, &truth_i);
        }
        if i >= RELOCATION
            && (i - RELOCATION).is_multiple_of(5)
            && relocate(&mut session, &k, &frame, &truth_i)
        {
            report.relocated.get_or_insert(i);
        }
        report.samples.push(located(&session, &k, &truth_i));
        previous = Some((k, frame, truth_i));
    }
    report
}

/// Largest view error in a range, with its frame.
fn worst(report: &Report, range: std::ops::Range<usize>) -> (f64, usize) {
    report.samples[range.clone()]
        .iter()
        .zip(range)
        .filter_map(|(s, i)| s.as_ref().map(|s| (s.view_error_deg, i)))
        .fold((0.0, 0), |a, b| if b.0 > a.0 { b } else { a })
}

/// Uses are offered only when the true error is inside their limits.
fn assert_uses_are_honest(report: &Report, name: &str) {
    for (i, s) in report.samples.iter().enumerate() {
        let Some(s) = s else { continue };
        if s.projectable {
            assert!(
                s.view_error_deg <= 6.0 && s.position_error_m <= 30.0,
                "{name}: frame {i} projectable with {:.1} deg, {:.1} m",
                s.view_error_deg,
                s.position_error_m
            );
        }
        if s.navigable {
            assert!(
                s.position_error_m <= 50.0,
                "{name}: frame {i} navigable with {:.1} m error",
                s.position_error_m
            );
        }
        if let Some(bound) = s.tilt_bound_deg {
            assert!(
                s.view_error_deg <= 3.0 * bound + 0.5,
                "{name}: frame {i}: {:.1} deg error, {bound:.1} deg tilt bound",
                s.view_error_deg
            );
        }
        assert!(
            s.position_error_m <= 3.0 * s.position_bound_m,
            "{name}: frame {i}: {:.1} m error, {:.1} m bound",
            s.position_error_m,
            s.position_bound_m
        );
    }
}

#[test]
fn map_depth_tracking_keeps_an_anchor_tilt_error_and_reports_it_honestly() {
    let report = fly(false);
    let (tilt, _) = worst(&report, 30..RELOCATION);
    assert!(
        tilt > 2.5,
        "nothing observes tilt between anchors: worst {tilt:.1} deg"
    );
    assert_uses_are_honest(&report, "tracking only");
    assert!(
        report.relocated.is_some(),
        "relocation proposals use the anchor tilt, not the tracked tilt"
    );
}

#[test]
fn ground_plane_observations_correct_tilt_over_a_long_flight() {
    let report = fly(true);
    let (late, at) = worst(&report, 60..FRAMES);
    let errors: Vec<f64> = report.samples[60..]
        .iter()
        .flatten()
        .map(|s| s.view_error_deg)
        .collect();
    let mean = errors.iter().sum::<f64>() / errors.len() as f64;
    // The declared ground-normal error is about two degrees for 15 m roofs at 100 m.
    assert!(
        late < 3.0 && mean < 1.0,
        "ground planes bound tilt: worst {late:.1} deg at frame {at}, mean {mean:.2} deg"
    );
    assert_uses_are_honest(&report, "with ground planes");
    let relocated = report
        .relocated
        .expect("relocation proposals reach the map again");
    let s = report.samples[relocated]
        .as_ref()
        .expect("the relocated frame is located");
    // The scene hides a 4% focal error that the calibration does not declare.
    // The map match carries it, so the frame follows the map within three
    // times the declared bound, the same rule as the other honesty checks.
    assert!(
        s.position_error_m < 8.0 && s.position_error_m <= 3.0 * s.position_bound_m,
        "relocated frame: {:.1} m error, {:.1} m bound",
        s.position_error_m,
        s.position_bound_m
    );
}
