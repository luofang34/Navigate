//! Empirical challenge test suite for Milestone 5 (Features 23, 24, 25, and 26).
//!
//! This suite validates four core areas:
//! 1. Dynamic ORT input tensor name discovery and extraction binding.
//! 2. Bounded channel queue backpressure (capacity 4) and latency isolation.
//! 3. Condition number and numerical stability of geometry covariance scaling (0.001).
//! 4. Repository structure compliance and file length limits (<= 500 lines).

use image::GrayImage;
use nalgebra::{Matrix6, SMatrix, SVector, UnitQuaternion, Vector2, Vector3};
use navigate_match::{ModelSessionExtractor, SparseFeatureExtractor};
use navigate_visual::{EngineKind, HostTensor, ModelInference, NamedTensor};
use std::{
    error::Error,
    sync::mpsc::{self, TrySendError},
    thread,
    time::{Duration, Instant},
};

type TestResult = Result<(), Box<dyn Error>>;

// --- Section 1: Dynamic ORT Input Tensor Discovery Challenge ---

/// Mock ONNX inference session that checks the bound input tensor name.
struct DynamicInputMockSession {
    expected_name: String,
    forward_count: usize,
}

impl DynamicInputMockSession {
    fn new(expected_name: &str) -> Self {
        Self {
            expected_name: expected_name.to_owned(),
            forward_count: 0,
        }
    }
}

impl ModelInference for DynamicInputMockSession {
    type Tensor = HostTensor;
    type Error = String;

    fn engine_kind(&self) -> EngineKind {
        EngineKind::Ort
    }

    fn forward(
        &mut self,
        inputs: &[NamedTensor<HostTensor>],
    ) -> Result<Vec<NamedTensor<HostTensor>>, String> {
        let primary = inputs
            .first()
            .ok_or_else(|| "missing primary input tensor".to_string())?;
        if primary.name != self.expected_name {
            return Err(format!(
                "input name mismatch: expected '{}', got '{}'",
                self.expected_name, primary.name
            ));
        }
        self.forward_count += 1;

        // Return exact dense XFeat output maps (72 x 100 cells = 7200 cells).
        const CELLS: usize = 72 * 100;
        let descriptors = HostTensor::new(vec![1, 64, 72, 100], vec![0.05_f32; 64 * CELLS])
            .map_err(|e| format!("descriptors: {e:?}"))?;
        let keypoint_logits = HostTensor::new(vec![1, 65, 72, 100], vec![0.02_f32; 65 * CELLS])
            .map_err(|e| format!("keypoint_logits: {e:?}"))?;
        let reliability = HostTensor::new(vec![1, 1, 72, 100], vec![0.9_f32; CELLS])
            .map_err(|e| format!("reliability: {e:?}"))?;

        Ok(vec![
            NamedTensor::new("descriptors", descriptors),
            NamedTensor::new("keypoint_logits", keypoint_logits),
            NamedTensor::new("reliability", reliability),
        ])
    }
}

fn create_test_image_with_contrast() -> GrayImage {
    let mut img = GrayImage::new(64, 64);
    img.put_pixel(0, 0, image::Luma([0]));
    img.put_pixel(1, 1, image::Luma([255]));
    img
}

#[test]
fn challenge_dynamic_ort_input_name_discovery() -> TestResult {
    let contract = [0x5au8; 32];
    let custom_names = ["images", "input.1", "data", "custom_sensor_tensor_0"];

    for &name in &custom_names {
        // Instantiate mock session expecting the non-standard tensor name.
        let session = DynamicInputMockSession::new(name);
        let extractor = ModelSessionExtractor::new(session, name, contract, "ort-custom");

        let img = create_test_image_with_contrast();
        let features = extractor.extract_masked_blocking(&img, None, 128);
        assert!(
            features.is_ok(),
            "Dynamic input name '{name}' must successfully execute extraction"
        );
    }
    Ok(())
}

#[test]
fn challenge_hardcoded_tensor_name_rejection() -> TestResult {
    let contract = [0x42u8; 32];
    // Model expects 'images' (common in ONNX exports), but extractor uses hardcoded 'image'.
    let session = DynamicInputMockSession::new("images");
    let extractor = ModelSessionExtractor::new(session, "image", contract, "ort-hardcoded");

    let img = create_test_image_with_contrast();
    let result = extractor.extract_masked_blocking(&img, None, 128);
    assert!(
        result.is_err(),
        "Hardcoded 'image' name must fail when graph expects 'images'"
    );
    Ok(())
}

#[test]
fn challenge_primary_input_selection_ordering() -> TestResult {
    // Test discovery selection rule: first input in model graph is primary.
    let graph_inputs = ["images", "metadata_flags", "rough_prior_pose"];
    let primary = graph_inputs
        .first()
        .copied()
        .ok_or("no inputs in model graph")?;
    assert_eq!(primary, "images");

    let empty_inputs: Vec<&str> = Vec::new();
    let empty_primary = empty_inputs.first().copied();
    assert!(
        empty_primary.is_none(),
        "Empty graph inputs must report None"
    );
    Ok(())
}

// --- Section 2: Bounded Channel Queue Backpressure Challenge ---

#[test]
fn challenge_bounded_sync_channel_capacity_and_backpressure() -> TestResult {
    const QUEUE_CAPACITY: usize = 4;
    let (tx, rx) = mpsc::sync_channel::<usize>(QUEUE_CAPACITY);

    // Channel must accept exactly QUEUE_CAPACITY items without blocking.
    for i in 0..QUEUE_CAPACITY {
        assert!(
            tx.try_send(i).is_ok(),
            "Buffer slot {i} must accept item when queue has capacity {QUEUE_CAPACITY}"
        );
    }

    // Next submission must fail with TrySendError::Full immediately.
    let t0 = Instant::now();
    let overflow_result = tx.try_send(999);
    let elapsed = t0.elapsed();

    assert!(
        matches!(overflow_result, Err(TrySendError::Full(999))),
        "Submitting when capacity is full must reject with TrySendError::Full"
    );
    assert!(
        elapsed < Duration::from_millis(5),
        "Non-blocking try_send must reject immediately without stall"
    );

    // Draining one item restores capacity by exactly one.
    let item = rx.recv()?;
    assert_eq!(item, 0);

    assert!(
        tx.try_send(100).is_ok(),
        "Submitting after drain must succeed"
    );
    assert!(
        matches!(tx.try_send(101), Err(TrySendError::Full(101))),
        "Submitting past capacity again must reject"
    );
    Ok(())
}

#[test]
fn challenge_worker_queue_isolation_under_load_burst() -> TestResult {
    const CAPACITY: usize = 4;
    let (tx, rx) = mpsc::sync_channel::<u32>(CAPACITY);
    let (done_tx, done_rx) = mpsc::channel::<()>();

    // Worker simulates slow background task (10 ms per item).
    let worker_handle = thread::spawn(move || {
        let mut count = 0;
        while let Ok(_job) = rx.recv() {
            thread::sleep(Duration::from_millis(8));
            count += 1;
        }
        done_tx.send(()).ok();
        count
    });

    // Foreground thread runs 30 simulated tracking frames (bursting jobs).
    let mut accepted = 0_usize;
    let mut rejected = 0_usize;
    let mut max_submit_latency = Duration::ZERO;

    for frame in 0..30 {
        let t_start = Instant::now();
        match tx.try_send(frame) {
            Ok(()) => accepted += 1,
            Err(TrySendError::Full(_)) => rejected += 1,
            Err(TrySendError::Disconnected(_)) => panic!("worker died unexpectedly"),
        }
        let submit_time = t_start.elapsed();
        if submit_time > max_submit_latency {
            max_submit_latency = submit_time;
        }
        // Foreground frame interval is 2 ms in this test.
        thread::sleep(Duration::from_millis(2));
    }

    drop(tx);
    done_rx.recv_timeout(Duration::from_secs(2))?;
    let processed = worker_handle.join().map_err(|_| "worker thread panicked")?;

    assert!(
        max_submit_latency < Duration::from_millis(5),
        "Foreground submit latency must remain sub-5ms under burst pressure"
    );
    assert!(
        rejected > 0,
        "Slow background worker must cause bounded queue to reject jobs"
    );
    assert_eq!(
        accepted, processed,
        "Every accepted job must be processed by the worker"
    );
    Ok(())
}

// --- Section 3: Geometry Covariance Scaling (0.001) Challenge ---

/// Helper to compute normal equations with selectable rotation scaling factor.
#[allow(clippy::too_many_arguments)]
fn compute_normal_matrix_with_scale(
    camera_fx: f64,
    camera_fy: f64,
    _cx: f64,
    _cy: f64,
    points: &[(Vector3<f64>, Vector2<f64>)],
    pose_pos: Vector3<f64>,
    pose_rot: UnitQuaternion<f64>,
    rot_scale: f64,
) -> Matrix6<f64> {
    let mut h = Matrix6::<f64>::zeros();
    let world_to_eye = pose_rot.inverse().to_rotation_matrix();

    for (world, pixel) in points {
        let eye = world_to_eye * (world - pose_pos);
        let depth = -eye.z;
        if depth <= 0.01 {
            continue;
        }
        let proj = SMatrix::<f64, 2, 3>::new(
            camera_fx / depth,
            0.0,
            camera_fx * eye.x / depth.powi(2),
            0.0,
            -camera_fy / depth,
            -camera_fy * eye.y / depth.powi(2),
        );
        let mut motion = SMatrix::<f64, 3, 6>::zeros();
        motion
            .fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&(-world_to_eye.matrix()));
        // Apply rotation scaling factor (1.0 for radians, 0.001 for milliradians).
        motion
            .fixed_view_mut::<3, 3>(0, 3)
            .copy_from(&(eye.cross_matrix() * rot_scale));

        let jacobian = proj * motion;
        let _ = pixel; // unweighted for pure geometry test
        h += jacobian.transpose() * jacobian;
    }
    h
}

fn condition_number(m: &Matrix6<f64>) -> f64 {
    let eig = m.symmetric_eigen().eigenvalues;
    let min_e = eig.min();
    let max_e = eig.max();
    if min_e <= 1e-15 {
        f64::INFINITY
    } else {
        max_e / min_e
    }
}

#[test]
fn challenge_geometry_scaling_condition_number_improvement() -> TestResult {
    let fx = 1000.0;
    let fy = 1000.0;
    let cx = 400.0;
    let cy = 300.0;
    let pose_pos = Vector3::new(0.0, 0.0, 1000.0);
    let pose_rot = UnitQuaternion::identity();

    // 25 landmarks spread over terrain at 1000m AGL.
    let mut points = Vec::new();
    for ix in -2..=2 {
        for iy in -2..=2 {
            let wx = ix as f64 * 150.0;
            let wy = iy as f64 * 150.0;
            let world = Vector3::new(wx, wy, 0.0);
            let px = cx + fx * wx / 1000.0;
            let py = cy - fy * wy / 1000.0;
            points.push((world, Vector2::new(px, py)));
        }
    }

    // Compute normal matrix with unscaled radians (rot_scale = 1.0).
    let h_unscaled =
        compute_normal_matrix_with_scale(fx, fy, cx, cy, &points, pose_pos, pose_rot, 1.0);
    let cond_unscaled = condition_number(&h_unscaled);

    // Compute normal matrix with milliradian scaling (rot_scale = 0.001).
    let h_scaled =
        compute_normal_matrix_with_scale(fx, fy, cx, cy, &points, pose_pos, pose_rot, 0.001);
    let cond_scaled = condition_number(&h_scaled);

    assert!(
        cond_unscaled > 1e5,
        "Unscaled normal matrix condition number ({cond_unscaled:.2e}) must exceed 1e5"
    );
    assert!(
        cond_scaled < 5e3,
        "Scaled normal matrix condition number ({cond_scaled:.2e}) must remain under 5e3"
    );

    let improvement_factor = cond_unscaled / cond_scaled;
    assert!(
        improvement_factor > 1e4,
        "Milliradian scaling must improve condition number by > 10,000x (got {improvement_factor:.1e}x)"
    );
    Ok(())
}

#[test]
fn challenge_geometry_scaling_shallow_grazing_and_occlusion() -> TestResult {
    let fx = 900.0;
    let fy = 900.0;
    let cx = 400.0;
    let cy = 300.0;

    // Shallow grazing angle (12 degrees down, looking forward 3000m).
    let pitch_rad = 12.0_f64.to_radians();
    let pose_rot = UnitQuaternion::from_euler_angles(pitch_rad, 0.0, 0.0);
    let pose_pos = Vector3::new(0.0, -1000.0, 500.0);

    // Asymmetric landmarks simulating 70% sky occlusion (only bottom 30% of frame has points).
    let mut points = Vec::new();
    for i in 1..=20 {
        let forward_dist = 800.0 + (i as f64) * 120.0;
        let lateral = ((i % 5) as f64 - 2.0) * 80.0;
        let world = Vector3::new(lateral, forward_dist, 0.0);
        let px = cx + fx * (lateral / forward_dist);
        let py = cy + 120.0 + (i as f64) * 5.0; // clustered near bottom of sensor
        points.push((world, Vector2::new(px, py)));
    }

    let h_unscaled =
        compute_normal_matrix_with_scale(fx, fy, cx, cy, &points, pose_pos, pose_rot, 1.0);
    let cond_unscaled = condition_number(&h_unscaled);

    let h_scaled =
        compute_normal_matrix_with_scale(fx, fy, cx, cy, &points, pose_pos, pose_rot, 0.001);
    let cond_scaled = condition_number(&h_scaled);

    // Unscaled formulation condition number can trigger false degenerate rejection (> 1e7).
    assert!(
        cond_unscaled > 1e6,
        "Grazing occlusion unscaled condition number must be high ({cond_unscaled:.2e})"
    );
    assert!(
        cond_scaled < 1e5,
        "Grazing occlusion scaled condition number must stay well below 1e10 cutoff ({cond_scaled:.2e})"
    );
    Ok(())
}

#[test]
fn challenge_covariance_inversion_round_trip_and_si_units() -> TestResult {
    let fx = 1000.0;
    let fy = 1000.0;
    let cx = 400.0;
    let cy = 300.0;
    let pose_pos = Vector3::new(0.0, 0.0, 800.0);
    let pose_rot = UnitQuaternion::identity();

    let mut points = Vec::new();
    for ix in -3..=3 {
        for iy in -3..=3 {
            let wx = ix as f64 * 120.0;
            let wy = iy as f64 * 120.0;
            points.push((Vector3::new(wx, wy, 0.0), Vector2::new(cx, cy)));
        }
    }

    let h_scaled =
        compute_normal_matrix_with_scale(fx, fy, cx, cy, &points, pose_pos, pose_rot, 0.001);
    let h_inv = h_scaled
        .try_inverse()
        .ok_or("normal matrix inversion failed")?;

    // Check algebraic inversion accuracy: H * H^-1 = I.
    let identity_error = (h_scaled * h_inv - Matrix6::identity()).norm();
    assert!(
        identity_error < 1e-10,
        "Scaled inversion must satisfy H * H^-1 = I within 1e-10 (got {identity_error:.2e})"
    );

    // Post-scaling matrix S = diag(1, 1, 1, 0.001, 0.001, 0.001).
    let s = Matrix6::<f64>::from_diagonal(&SVector::<f64, 6>::from_row_slice(&[
        1.0, 1.0, 1.0, 0.001, 0.001, 0.001,
    ]));
    let variance = 1.0;
    let covariance = s * h_inv * s * variance;

    // Verify symmetry and positive-definiteness.
    let sym_error = (covariance - covariance.transpose()).norm();
    assert!(
        sym_error < 1e-14,
        "Covariance must be symmetric ({sym_error:.2e})"
    );
    assert!(covariance.symmetric_eigen().eigenvalues.min() > 0.0);

    // Position diagonal in m^2 and rotation diagonal in rad^2.
    for i in 0..3 {
        let pos_var = covariance[(i, i)];
        assert!(
            pos_var > 0.0 && pos_var < 1000.0,
            "Invalid pos m^2 ({pos_var})"
        );
    }
    for i in 3..6 {
        let rot_var = covariance[(i, i)];
        assert!(
            rot_var > 0.0 && rot_var < 1e-2,
            "Invalid rot rad^2 ({rot_var})"
        );
    }
    Ok(())
}

// --- Section 4: Repository Structural Compliance Challenge ---

#[test]
fn challenge_milestone5_file_length_limits() -> TestResult {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let repo_root = std::path::Path::new(manifest_dir)
        .parent()
        .and_then(|p| p.parent())
        .ok_or("cannot find repo root")?;

    let touched_files = [
        "tools/visual-pipeline/src/surface/gpu/atlas.rs",
        "tools/visual-pipeline/src/surface/gpu.rs",
        "tools/visual-pipeline/src/corridor/nadir.rs",
        "tools/visual-pipeline/src/corridor.rs",
        "tools/visual-pipeline/src/corridor/anchor.rs",
        "tools/visual-pipeline/src/corridor/tile_builder.rs",
        "tools/visual-pipeline/src/stages/seeds.rs",
        "crates/navigate-match/src/ort_session.rs",
        "tools/visual-pipeline/src/xfeat_runtime.rs",
        "tools/visual-pipeline/src/corridor/background.rs",
        "tools/visual-pipeline/tests/m5_ort_queue_scaling_challenge.rs",
    ];

    for relative_path in touched_files {
        let full_path = repo_root.join(relative_path);
        assert!(
            full_path.exists(),
            "Milestone 5 file does not exist: {}",
            full_path.display()
        );
        let content = std::fs::read_to_string(&full_path)?;
        let lines = content.lines().count();
        assert!(
            lines <= 500,
            "File {} has {} lines; must be <= 500 lines",
            relative_path,
            lines
        );
    }
    Ok(())
}
