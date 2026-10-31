use super::{KeyLayers, RefreshLayers, Refresher, SegEngine, four_class_keep_layers};
use burn::tensor::Device;
use image::RgbImage;
use nalgebra::{Matrix3, Rotation3};
use navigate_segment::{
    AIRFRAME, CLASSES, GROUND, MaskPolicy, PROPELLER, PreprocessorConfig, SKY, Segformer,
    SegformerConfig,
};
use std::error::Error;
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

    // Frame 1 submits to the background worker without blocking the frame loop.
    let first = refresher
        .step(&mut key, &rgb, Some(&rotation(0.0)), interval)
        .unwrap();
    started.recv().unwrap();
    assert!(first.submitted && !first.refreshed);
    assert!(!key.usable(DIMS));
    refresher.note_solved_rotation(&mut key, rotation(0.05));

    // While Frame 1's Segformer job runs, early frames return immediately without usable key layers.
    for frame in 0..4 {
        let step = refresher
            .step(&mut key, &rgb, Some(&rotation(0.08)), interval)
            .unwrap();
        assert!(
            !step.refreshed && !step.submitted,
            "cold frame {frame} stalled"
        );
        assert!(!key.usable(DIMS));
    }

    // Release Frame 1's Segformer layers; they install with Frame 1's solved rotation.
    permit.send(10).unwrap();
    let first_arrived = (0..POLL_LIMIT)
        .find_map(|_| {
            let step = refresher
                .step(&mut key, &rgb, Some(&rotation(0.1)), interval)
                .unwrap();
            std::thread::yield_now();
            step.refreshed.then_some(step)
        })
        .expect("first refresh must arrive");
    assert_eq!(first_arrived.segformer_ms, Some(10.0));
    assert!(key.usable(DIMS));
    assert_eq!(key.sky_keep, vec![10; 12]);
    assert_eq!(key.rot, Some(rotation(0.05)));

    // Because >= interval frames elapsed since Frame 1 was submitted, that same step also submits a new refresh.
    assert!(first_arrived.submitted);
    refresher.note_solved_rotation(&mut key, rotation(0.2));
    started.recv().unwrap();

    // The worker is blocked; later frames still warp the first layers without stalling.
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
    let submit = refresher
        .step(&mut key, &rgb, Some(&rotation(0.5)), 1)
        .unwrap();
    assert!(submit.submitted && !submit.refreshed);
    started.recv().unwrap();
    permit.send(2).unwrap();
    let arrived = (0..POLL_LIMIT).any(|_| {
        std::thread::yield_now();
        refresher.step(&mut key, &rgb, None, 1).unwrap().refreshed
    });
    assert!(arrived, "released refresh must arrive");
    assert_eq!(key.rot, Some(rotation(0.5)));
}

#[test]
fn four_class_ground_pixels_remain_valid_while_airframe_and_propeller_occlude()
-> Result<(), Box<dyn Error>> {
    let labels = vec![
        SKY, SKY, GROUND, GROUND, // row 0
        GROUND, GROUND, AIRFRAME, PROPELLER, // row 1
    ];
    let (air_keep, sky_keep) = four_class_keep_layers(&labels, SKY, 4, 2, 0);
    assert_eq!(sky_keep, vec![0, 0, 255, 255, 255, 255, 255, 255]);
    assert_eq!(air_keep, vec![255, 255, 255, 255, 255, 255, 0, 0]);

    // Verify SegEngine::eval on a 4-class Segformer biased to predict GROUND (label 1).
    let device = Device::flex();
    let cfg = SegformerConfig::mit_b0(&CLASSES);
    let mut model = Segformer::new(&cfg, &device);
    let bias = burn::tensor::Tensor::<1>::from_data(
        burn::tensor::TensorData::new(vec![-5.0_f32, 10.0, -5.0, -5.0], [4]),
        &device,
    );
    model.decode_head.classifier.bias = Some(burn::module::Param::from_tensor(bias));
    let preproc = PreprocessorConfig {
        size: [("height".to_owned(), 32), ("width".to_owned(), 32)]
            .into_iter()
            .collect(),
        ..Default::default()
    };
    let engine = SegEngine {
        model,
        preproc,
        airplane_indices: Vec::new(),
        sky_label: SKY,
        is_four_class: true,
        device,
        policy: MaskPolicy::AirframeOcclusions,
        margin_px: 1,
    };
    let rgb = RgbImage::new(16, 12);
    let (air, sky) = engine.eval(&rgb)?;
    assert!(air.iter().all(|&v| v == 255), "GROUND must stay unoccluded");
    assert!(sky.iter().all(|&v| v == 255), "GROUND must not become sky");
    Ok(())
}
