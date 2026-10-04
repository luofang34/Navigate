//! A map match fixes the ground under the image much better than it fixes
//! the camera tilt or position alone. Through the real tracking, plane,
//! map-match, and session paths, ground planes must correct the tilt without
//! moving the matched ground. The map-match tilt error comes from the
//! geometry: a local slope error of the map elevation model and roofs that
//! the elevation model does not contain.

use crate::scene::*;
use crate::support::*;
use nalgebra::{Vector2, Vector3};
use navigate_visual::{
    Frame, LocalizerConfig, PlaneMotionConfig, PosePrior, PoseVerifier, TrackingReference,
    plane_motion,
};
use navigate_visual_session::{
    AnchorDecision, AnchorObservation, Calibration, CameraMount, FrameKey, GroundPlaneObservation,
    LensModel, MapPose, MapReliability, Pose, RegionChange, RelativeMotion, SessionConfig,
    SurfaceModel, TerrainNormal, VerticalReference, VisualSession, to_camera,
};

const FRAMES: usize = 130;
/// Anchors fall between keyframes, so they also exercise the transfer to a keyframe.
const ANCHOR_EVERY: usize = 25;
/// Local slope error of the map elevation model. The map match fits the
/// camera to the wrong slope: it turns and moves the camera together, and the
/// ground under the image stays in place.
const MAP_SLOPE_ERROR: f64 = 0.07;

/// The declared calibration and map budget of a host that knows neither the
/// lens nor the mount, with imagery of unknown age and 20 m objects.
fn host_calibration() -> Calibration {
    Calibration {
        intrinsics: camera(),
        lens: LensModel::Unknown,
        mount: CameraMount::Unknown,
        map_vertical: VerticalReference::Unknown,
    }
}

fn host_reliability() -> MapReliability {
    MapReliability {
        horizontal_m: 3.0,
        vertical_m: 5.0,
        imagery_age_years: None,
        age_growth_m_per_year: 1.0,
        unknown_age_years: 5.0,
        surface: SurfaceModel::BareEarth {
            max_object_height_m: 20.0,
        },
        change: RegionChange::Unknown,
    }
}

fn estimate(session: &VisualSession, k: &FrameKey) -> Option<Pose> {
    match session.pose_at(k).ok()?.map {
        MapPose::Located { pose, .. } => Some(pose),
        MapPose::Unlocated(_) => None,
    }
}

fn map_anchor(frame: &Frame, k: FrameKey, truth: &Pose, seed: &mut u64) -> AnchorObservation {
    let render = Pose::from_parts(
        (truth.translation.vector + Vector3::new(6.0, -4.0, 3.0)).into(),
        truth.rotation,
    );
    // The error changes sign between anchors, as local model errors do.
    let sign = if (k.index / ANCHOR_EVERY as u64).is_multiple_of(2) {
        1.0
    } else {
        -1.0
    };
    let slope = Vector2::new(sign * MAP_SLOPE_ERROR, 0.5 * MAP_SLOPE_ERROR);
    let matches = object_map_matches(&render, truth, slope, seed);
    let prior = PosePrior {
        pose: to_camera(&render),
        position_radius_m: 80.0,
        attitude_radius_rad: 0.5,
    };
    let estimate = PoseVerifier::new(LocalizerConfig::default())
        .unwrap()
        .verify(frame, &dem_view(&render), &prior, &matches, "synthetic-map")
        .unwrap();
    AnchorObservation::from_estimate(k, &estimate, host_reliability())
}

fn track(
    session: &mut VisualSession,
    previous: (&FrameKey, &Frame, &Pose),
    current: (&FrameKey, &Frame, &Pose),
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

/// Fly with anchors and ground planes; return the final revised horizontal
/// error of the ground under the image centre, for each frame.
fn fly() -> Vec<Option<f64>> {
    let mut session = VisualSession::new(SessionConfig::standard(), host_calibration()).unwrap();
    let (mut seed, mut map_seed) = (11_u64, 5_u64);
    let mut flown = Vec::new();
    let mut previous: Option<(FrameKey, Frame, Pose)> = None;
    for i in 0..FRAMES {
        let (k, frame, truth_i) = (key(0, i as u64), image_frame(i as u64), truth(i));
        session.observe_frame(record(k, &frame)).unwrap();
        if let Some((pk, pf, pt)) = &previous {
            track(
                &mut session,
                (pk, pf, pt),
                (&k, &frame, &truth_i),
                &mut seed,
            );
        }
        if i.is_multiple_of(ANCHOR_EVERY) {
            let decision = session.submit_anchor(map_anchor(&frame, k, &truth_i, &mut map_seed));
            assert!(
                matches!(decision, Ok(AnchorDecision::Accepted { .. })),
                "frame {i}: {decision:?}"
            );
        }
        flown.push((k, truth_i));
        previous = Some((k, frame, truth_i));
    }
    flown
        .iter()
        .map(|(k, truth)| {
            let estimate = estimate(&session, k)?;
            let (a, b) = (centre_ground(&estimate)?, centre_ground(truth)?);
            Some((a - b).xy().norm())
        })
        .collect()
}

#[test]
fn ground_planes_correct_anchor_tilt_without_moving_the_matched_ground() {
    let errors = fly();
    let located: Vec<(usize, f64)> = errors
        .iter()
        .enumerate()
        .filter_map(|(i, e)| e.map(|e| (i, e)))
        .collect();
    assert!(located.len() >= FRAMES - 1, "located {}", located.len());
    let worst = located
        .iter()
        .fold((0, 0.0), |a, b| if b.1 > a.1 { *b } else { a });
    let at_anchors: Vec<f64> = (0..FRAMES)
        .step_by(ANCHOR_EVERY)
        .filter_map(|i| errors[i])
        .collect();
    let mean = located.iter().map(|(_, e)| e).sum::<f64>() / located.len() as f64;
    // A 4 degree slope error tilts the map match and moves the camera by
    // several metres. The ground under the image must stay near the map.
    assert!(
        worst.1 < 2.0 && mean < 1.0,
        "ground error: worst {:.2} m at frame {}, mean {mean:.2} m, at anchors {at_anchors:.2?}",
        worst.1,
        worst.0
    );
}
