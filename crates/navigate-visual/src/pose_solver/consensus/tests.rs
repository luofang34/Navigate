use super::*;
use crate::pose_solver::residual;

fn fixture() -> (CameraModel, CameraPose, Vec<Correspondence>) {
    let camera = CameraModel {
        width: 640,
        height: 480,
        fx: 550.0,
        fy: 540.0,
        cx: 319.5,
        cy: 239.5,
    };
    let pose = CameraPose {
        position: Vector3::new(110.0, -82.0, 700.0),
        orientation: UnitQuaternion::from_euler_angles(0.7, 0.02, -0.13),
    };
    let points = (0..160)
        .map(|i| {
            let pixel = Vector2::new(
                30.0 + f64::from(i % 16) * 35.0,
                20.0 + f64::from(i / 16) * 40.0,
            );
            Correspondence {
                world: camera.unproject(&pose, pixel, 500.0 + f64::from(i % 7) * 30.0),
                pixel: if i % 3 == 0 {
                    pixel + Vector2::new(80.0, -40.0)
                } else {
                    pixel
                },
            }
        })
        .collect();
    (camera, pose, points)
}

#[test]
fn bounded_scoring_preserves_full_support_and_tie_rejection() {
    let (camera, pose, mut points) = fixture();
    points.push(Correspondence {
        world: pose.position,
        pixel: Vector2::zeros(),
    });
    points.push(Correspondence {
        world: Vector3::repeat(f64::NAN),
        pixel: Vector2::zeros(),
    });
    let full: Vec<_> = points
        .iter()
        .enumerate()
        .filter_map(|(i, p)| (residual(&camera, &pose, p) <= 2.5).then_some(i))
        .collect();
    assert_eq!(inliers(&camera, &points, &pose, 2.5), full);
    for floor in 0..=points.len() {
        assert_eq!(
            better_support(&camera, &points, &pose, 2.5, floor),
            (full.len() > floor).then_some(full.len())
        );
    }
}

#[test]
fn projection_preserves_reprojection_threshold_boundaries() {
    let (camera, pose, points) = fixture();
    let projection = Projection::new(&camera, &pose);
    for point in &points {
        let error = residual(&camera, &pose, point);
        for threshold in [error.next_down(), error, error.next_up(), 0.0, 2.5] {
            assert_eq!(projection.supports(point, threshold), error <= threshold);
        }
    }
}

#[test]
fn consensus_preserves_seeded_pose_with_outliers_for_both_motion_models() {
    let (camera, truth, points) = fixture();
    let initial = CameraPose {
        position: truth.position + Vector3::new(20.0, -10.0, 15.0),
        orientation: truth.orientation * UnitQuaternion::from_euler_angles(0.01, -0.02, 0.02),
    };
    for motion in [Motion::Free, Motion::FixedTilt] {
        let expected = full_consensus(&camera, &points, initial, 2.5, motion);
        let actual = initialize(&camera, &points, initial, 2.5, motion);
        assert_eq!(actual.position, expected.position);
        assert_eq!(actual.orientation, expected.orientation);
    }
}

fn full_consensus(
    camera: &CameraModel,
    points: &[Correspondence],
    initial: CameraPose,
    threshold: f64,
    motion: Motion,
) -> CameraPose {
    let find = |pose: &CameraPose| -> Vec<usize> {
        points
            .iter()
            .enumerate()
            .filter_map(|(i, p)| (residual(camera, pose, p) <= threshold).then_some(i))
            .collect()
    };
    let mut best = initial;
    let mut support = find(&best);
    let mut seed = 0x7812_531a_u64;
    let mut budget = 4096;
    let mut trial = 0_usize;
    while trial < budget && support.len() < points.len() {
        trial = trial.wrapping_add(1);
        let mut indices = [usize::MAX; 4];
        for slot in 0..indices.len() {
            loop {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                let index = (seed >> 32) as usize % points.len();
                if !indices[..slot].contains(&index) {
                    indices[slot] = index;
                    break;
                }
            }
        }
        let sample: Vec<_> = indices.iter().map(|&i| points[i].clone()).collect();
        let Ok(pose) = optimize_motion(camera, &sample, initial, motion) else {
            continue;
        };
        let found = find(&pose);
        if found.len() <= support.len() {
            continue;
        }
        best = pose;
        support = found;
        let fraction = support.len() as f64 / points.len() as f64;
        budget = budget
            .min((0.001_f64.ln() / (1.0 - fraction.powi(4)).ln()).ceil() as usize)
            .max(64);
    }
    if support.len() >= 6 {
        let selected: Vec<_> = support.iter().map(|&i| points[i].clone()).collect();
        if let Ok(refined) = optimize_motion(camera, &selected, best, motion)
            && find(&refined).len() >= support.len()
        {
            best = refined;
        }
    }
    best
}
