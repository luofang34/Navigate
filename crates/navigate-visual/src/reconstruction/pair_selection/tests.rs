//! Scheduling checks with changing image coverage.
#![allow(clippy::expect_used, clippy::panic)]
use super::*;
fn fixture() -> (CameraModel, ImageTracks) {
    let camera = CameraModel {
        width: 640,
        height: 480,
        fx: 450.0,
        fy: 450.0,
        cx: 319.5,
        cy: 239.5,
    };
    let tracks = (0..360)
        .map(|id| {
            let range = if id < 300 { 0..7 } else { 6..12 };
            ImageTrack {
                feature_id: id,
                observations: range
                    .map(|camera_index| ScenePointObservation {
                        camera_index,
                        pixel: Vector2::new(
                            30.0 + (id % 20) as f64 * 20.0 + camera_index as f64 * 12.0,
                            30.0 + (id / 20 % 10) as f64 * 38.0,
                        ),
                    })
                    .collect(),
            }
        })
        .collect();
    (
        camera,
        ImageTracks {
            observation_sha256: (0..12).map(|i| format!("{i:064x}")).collect(),
            tracks,
        },
    )
}
#[test]
fn initial_budget_covers_late_surface_change_despite_stronger_early_links() {
    let (camera, graph) = fixture();
    let pairs = candidate_seed_pairs(&camera, &graph).expect("valid links");
    let parts: BTreeSet<_> = pairs
        .iter()
        .take(3)
        .map(|p| (p[0] + p[1]) * 3 / 24)
        .collect();
    assert_eq!(parts, BTreeSet::from([0, 1, 2]));
    assert!(pairs.iter().take(3).any(|p| p[0] >= 6 && p[1] >= 8));
    assert_eq!(pairs.iter().collect::<BTreeSet<_>>().len(), pairs.len());
}
#[test]
fn stationary_links_do_not_propose_a_translation_seed() {
    let (camera, mut graph) = fixture();
    for track in &mut graph.tracks {
        let pixel = track.observations[0].pixel;
        for o in &mut track.observations {
            o.pixel = pixel;
        }
    }
    assert!(
        candidate_seed_pairs(&camera, &graph)
            .expect("valid stationary links")
            .is_empty()
    );
}
#[test]
fn invalid_calibration_is_rejected_before_scheduling() {
    let (mut camera, graph) = fixture();
    camera.fx = 0.0;
    assert!(candidate_seed_pairs(&camera, &graph).is_err());
}
