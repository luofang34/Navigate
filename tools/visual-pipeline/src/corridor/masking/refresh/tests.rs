use super::{KeyLayers, RefreshLayers, Refresher};
use image::RgbImage;
use nalgebra::{Matrix3, Rotation3};
use std::sync::mpsc;

const DIMS: [u32; 2] = [4, 3];
/// Upper bound on polls for a released job, so that a regression fails instead of hanging.
const POLL_LIMIT: usize = 10_000_000;

fn layers(sky: u8) -> RefreshLayers {
    RefreshLayers {
        dims: DIMS,
        airframe_keep: vec![255; 12],
        sky_keep: vec![sky; 12],
        eval_ms: f64::from(sky),
    }
}

fn rotation(angle: f64) -> Matrix3<f64> {
    *Rotation3::from_euler_angles(0.0, 0.0, angle).matrix()
}

/// A background refresher whose jobs wait for one permit each from the test.
fn gated() -> (Refresher, mpsc::Sender<u8>, mpsc::Receiver<()>) {
    let (permit_tx, permit_rx) = mpsc::channel::<u8>();
    let (started_tx, started_rx) = mpsc::channel::<()>();
    let refresher = Refresher::spawn_background(move |_rgb| {
        started_tx.send(()).unwrap();
        Ok(layers(permit_rx.recv().unwrap()))
    })
    .unwrap();
    (refresher, permit_tx, started_rx)
}

#[test]
fn frames_keep_warping_old_layers_while_a_refresh_runs() {
    let (mut refresher, permit, started) = gated();
    let mut key = KeyLayers::default();
    let rgb = RgbImage::new(DIMS[0], DIMS[1]);
    let interval = 2;

    // The first frame has no layers to warp, so it waits for its own refresh.
    permit.send(10).unwrap();
    let first = refresher
        .step(&mut key, &rgb, Some(&rotation(0.0)), interval)
        .unwrap();
    started.recv().unwrap();
    assert!(first.refreshed && !first.submitted);
    assert_eq!(key.sky_keep, vec![10; 12]);

    // The second frame reaches the interval and goes to the worker.
    let submit = refresher
        .step(&mut key, &rgb, Some(&rotation(0.1)), interval)
        .unwrap();
    assert!(submit.submitted && !submit.refreshed);
    refresher.note_solved_rotation(&mut key, rotation(0.2));
    started.recv().unwrap();

    // The worker is blocked; later frames still use the first layers.
    for frame in 0..20 {
        let step = refresher
            .step(&mut key, &rgb, Some(&rotation(0.3)), interval)
            .unwrap();
        assert!(!step.refreshed && !step.submitted, "frame {frame} stalled");
        refresher.note_solved_rotation(&mut key, rotation(0.3));
        assert_eq!(key.sky_keep, vec![10; 12]);
    }

    permit.send(20).unwrap();
    let arrived = (0..POLL_LIMIT)
        .find_map(|_| {
            let step = refresher
                .step(&mut key, &rgb, Some(&rotation(0.4)), interval)
                .unwrap();
            std::thread::yield_now();
            step.refreshed.then_some(step)
        })
        .expect("released refresh must arrive");
    assert_eq!(arrived.segformer_ms, Some(20.0));
    assert_eq!(key.sky_keep, vec![20; 12]);
    // The layers belong to the submitted frame, so they keep its solved rotation.
    refresher.note_solved_rotation(&mut key, rotation(0.4));
    assert_eq!(key.rot, Some(rotation(0.2)));
}

#[test]
fn unsolved_submitted_frames_keep_their_estimated_rotation() {
    let (mut refresher, permit, started) = gated();
    let mut key = KeyLayers::default();
    let rgb = RgbImage::new(DIMS[0], DIMS[1]);
    permit.send(1).unwrap();
    refresher.step(&mut key, &rgb, None, 1).unwrap();
    started.recv().unwrap();
    let submit = refresher
        .step(&mut key, &rgb, Some(&rotation(0.5)), 1)
        .unwrap();
    assert!(submit.submitted);
    started.recv().unwrap();
    permit.send(2).unwrap();
    let arrived = (0..POLL_LIMIT).any(|_| {
        std::thread::yield_now();
        refresher.step(&mut key, &rgb, None, 1).unwrap().refreshed
    });
    assert!(arrived, "released refresh must arrive");
    assert_eq!(key.rot, Some(rotation(0.5)));
}
