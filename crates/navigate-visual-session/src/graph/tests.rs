use super::*;
use nalgebra::{Translation3, UnitQuaternion, Vector3};

fn pose(x: f64, y: f64, z: f64, roll: f64, pitch: f64, yaw: f64) -> Pose {
    Pose::from_parts(
        Translation3::new(x, y, z),
        UnitQuaternion::from_euler_angles(roll, pitch, yaw),
    )
}

fn perturbed(problem: &Problem, column: usize, h: f64) -> Problem {
    let mut step = vec![0.0; 6 * problem.poses.len() + 3 * problem.biases.len()];
    step[column] = h;
    let mut copy = problem.clone();
    copy.apply(&step);
    copy
}

#[test]
fn analytic_jacobians_match_central_differences() {
    // Zero residuals, where the Jacobians are exact.
    let a = pose(1.0, 2.0, 100.0, 0.1, -0.2, 0.3);
    let b = pose(4.5, 0.7, 101.0, 0.12, -0.15, 0.75);
    let bias = Vector3::new(0.4, -0.3, 0.2);
    let measured = a.inv_mul(&b);
    let anchored = Pose::from_parts(Translation3::from(b.translation.vector + bias), b.rotation);
    let problem = Problem {
        poses: vec![a, b],
        biases: vec![bias],
        factors: vec![
            Factor::Relative {
                a: 0,
                b: 1,
                a_to_b: measured,
                sigma_m: 0.5,
                sigma_rad: 0.02,
                robust: false,
            },
            Factor::Anchor {
                node: 1,
                bias: Some(0),
                pose: anchored,
                position_sqrt_info: Matrix3::new(2.0, 0.0, 0.0, 0.3, 1.5, 0.0, 0.1, 0.2, 1.0),
                sigma_rad: 0.05,
            },
        ],
        disabled: vec![false, false],
    };
    let mut problem = problem;
    problem.factors.push(Factor::Direction {
        node: 0,
        world: Vector3::z(),
        camera: a.rotation.inverse() * Vector3::z(),
        sigma_rad: 0.03,
    });
    problem.disabled.push(false);
    for factor in &problem.factors {
        let lin = problem.linearize(factor).unwrap();
        for (column, jacobian, width) in &lin.blocks {
            for k in 0..*width {
                let h = 1e-6;
                let plus = perturbed(&problem, column + k, h)
                    .linearize(factor)
                    .unwrap()
                    .residual;
                let minus = perturbed(&problem, column + k, -h)
                    .linearize(factor)
                    .unwrap()
                    .residual;
                let numeric = (plus - minus) / (2.0 * h);
                let analytic = jacobian.column(k);
                for row in 0..lin.rows {
                    let error = (numeric[row] - analytic[row]).abs();
                    assert!(
                        error < 1e-5 * (1.0 + numeric[row].abs()),
                        "column {} row {row}: {} vs {}",
                        column + k,
                        numeric[row],
                        analytic[row]
                    );
                }
            }
        }
    }
}

#[test]
fn a_closure_removes_heading_drift_around_a_loop() {
    // Twenty poses around a square. Odometry has a heading bias; the closure is exact.
    let truth: Vec<Pose> = (0..20)
        .map(|i| {
            let side = i / 5;
            let t = f64::from(i % 5) * 20.0;
            let (x, y) = match side {
                0 => (t, 0.0),
                1 => (100.0, t),
                2 => (100.0 - t, 100.0),
                _ => (0.0, 100.0 - t),
            };
            pose(
                x,
                y,
                100.0,
                0.0,
                0.0,
                f64::from(side) * std::f64::consts::FRAC_PI_2,
            )
        })
        .collect();
    let bias = UnitQuaternion::from_euler_angles(0.0, 0.0, 0.01);
    let mut drifted = vec![truth[0]];
    let mut factors = vec![Factor::Prior {
        node: 0,
        pose: truth[0],
        sigma_m: 0.01,
        sigma_rad: 0.001,
    }];
    for i in 1..20 {
        let mut step = truth[i - 1].inv_mul(&truth[i]);
        step.rotation = bias * step.rotation;
        drifted.push(drifted[i - 1] * step);
        factors.push(Factor::Relative {
            a: i - 1,
            b: i,
            a_to_b: step,
            sigma_m: 1.0,
            sigma_rad: 0.01,
            robust: false,
        });
    }
    let closure = truth[0].inv_mul(&truth[19]);
    factors.push(Factor::Relative {
        a: 0,
        b: 19,
        a_to_b: closure,
        sigma_m: 0.2,
        sigma_rad: 0.002,
        robust: true,
    });
    let disabled = vec![false; factors.len()];
    let mut problem = Problem {
        poses: drifted.clone(),
        biases: vec![],
        factors,
        disabled,
    };
    let end_error =
        |poses: &[Pose]| (poses[19].translation.vector - truth[19].translation.vector).norm();
    let before = end_error(&drifted);
    let report = problem.solve(30);
    let after = end_error(&problem.poses);
    assert!(report.final_cost < report.initial_cost);
    assert!(
        before > 10.0 && after < 1.0,
        "loop end error {before} m -> {after} m"
    );
}

#[test]
fn a_shared_bias_takes_a_common_anchor_offset() {
    // Four anchors with the same 5 m offset and tight independent errors.
    // The bias prior allows 10 m, so the bias takes most of the offset only
    // when an independent prior holds the trajectory.
    let truth: Vec<Pose> = (0..4)
        .map(|i| pose(f64::from(i) * 30.0, 0.0, 100.0, 0.0, 0.0, 0.0))
        .collect();
    let mut factors = vec![Factor::BiasPrior {
        bias: 0,
        sqrt_info: Matrix3::identity() / 10.0,
    }];
    for i in 0..4 {
        if i > 0 {
            factors.push(Factor::Relative {
                a: i - 1,
                b: i,
                a_to_b: truth[i - 1].inv_mul(&truth[i]),
                sigma_m: 0.05,
                sigma_rad: 0.001,
                robust: false,
            });
        }
        let shifted = Pose::from_parts(
            Translation3::from(truth[i].translation.vector + Vector3::new(5.0, 0.0, 0.0)),
            truth[i].rotation,
        );
        factors.push(Factor::Anchor {
            node: i,
            bias: Some(0),
            pose: shifted,
            position_sqrt_info: Matrix3::identity() / 0.5,
            sigma_rad: 0.01,
        });
    }
    factors.push(Factor::Prior {
        node: 0,
        pose: truth[0],
        sigma_m: 0.5,
        sigma_rad: 0.01,
    });
    let disabled = vec![false; factors.len()];
    let mut problem = Problem {
        poses: truth.clone(),
        biases: vec![Vector3::zeros()],
        factors,
        disabled,
    };
    problem.solve(30);
    assert!(problem.biases[0].x > 4.0, "bias {:?}", problem.biases[0]);
    assert!((problem.poses[3].translation.vector - truth[3].translation.vector).norm() < 1.0);
}

#[test]
fn a_ground_direction_bounds_tilt_and_leaves_heading_free() {
    let truth = pose(0.0, 0.0, 100.0, 0.05, -0.03, 1.2);
    let start = pose(0.0, 0.0, 100.0, 0.6, 0.4, 1.2);
    let factors = vec![
        Factor::Direction {
            node: 0,
            world: Vector3::z(),
            camera: truth.rotation.inverse() * Vector3::z(),
            sigma_rad: 0.02,
        },
        Factor::Prior {
            node: 0,
            pose: start,
            sigma_m: 1.0,
            sigma_rad: 10.0,
        },
    ];
    let mut problem = Problem {
        poses: vec![start],
        biases: vec![],
        disabled: vec![false; 2],
        factors,
    };
    problem.solve(30);
    let up = problem.poses[0].rotation.inverse() * Vector3::z();
    assert!(
        up.angle(&(truth.rotation.inverse() * Vector3::z())) < 1e-3,
        "tilt follows the direction"
    );
}
