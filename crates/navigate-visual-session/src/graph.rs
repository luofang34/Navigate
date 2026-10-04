//! Sparse pose graph over keyframe poses and map-error bias offsets.
//!
//! Pose increments use world-frame perturbations: position `p + rho` and
//! rotation `Exp(phi) R`. Bias variables are 3-D map offsets that all
//! anchors in one map cell share: an anchor measures the camera position plus
//! the bias. The common map error is then not counted again for each anchor.
//!
//! The Jacobians omit the right Jacobian of SO(3). They are exact at zero
//! rotation residual, and the damped iteration converges for the small
//! residuals that the gates allow.

mod linear;

use crate::pose::{Pose, skew};
use linear::{BlockSystem, Dims};
use nalgebra::{Matrix3, Matrix6, SMatrix, SVector, Translation3, UnitQuaternion, Vector3};

/// One graph factor. Indices refer to `Problem::poses` and `Problem::biases`.
#[derive(Clone, Debug)]
pub(crate) enum Factor {
    /// Measured pose of camera `b` in the frame of camera `a`.
    Relative {
        a: usize,
        b: usize,
        a_to_b: Pose,
        sigma_m: f64,
        sigma_rad: f64,
        robust: bool,
    },
    /// Map-frame pose of one camera, offset by a shared map-cell bias.
    ///
    /// The residual is the position error and the camera-frame rotation
    /// error. One whitening matrix holds both, because a map match fixes the
    /// ground under the image better than it fixes position or tilt alone.
    Anchor {
        node: usize,
        bias: Option<usize>,
        pose: Pose,
        /// Whitening matrix `L^-1`, where `L L^T` is the pose covariance:
        /// position in metres, then rotation `R = R_measured Exp(theta)` in radians.
        sqrt_info: Box<Matrix6<f64>>,
    },
    /// Weak gauge prior for a pose graph component without anchors.
    Prior {
        node: usize,
        pose: Pose,
        sigma_m: f64,
        sigma_rad: f64,
    },
    /// A world direction observed in the camera frame, such as the ground
    /// normal. It constrains tilt and leaves the heading free.
    Direction {
        node: usize,
        /// Unit direction in the world frame.
        world: Vector3<f64>,
        /// Unit direction in the camera eye frame.
        camera: Vector3<f64>,
        sigma_rad: f64,
    },
    /// Zero-mean prior of a map-cell bias.
    BiasPrior {
        bias: usize,
        /// Square root of the inverse bias covariance.
        sqrt_info: Matrix3<f64>,
    },
}

/// Graph variables and factors. The solver changes only the variables.
#[derive(Clone, Debug, Default)]
pub(crate) struct Problem {
    pub poses: Vec<Pose>,
    pub biases: Vec<Vector3<f64>>,
    pub factors: Vec<Factor>,
    /// Factors excluded from the cost, for example retracted closures.
    pub disabled: Vec<bool>,
}

/// Result of one optimization.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct SolveReport {
    pub initial_cost: f64,
    pub final_cost: f64,
    pub iterations: usize,
}

const CAUCHY_SCALE: f64 = 3.0;

/// Residual, Jacobian blocks, and robust weight of one factor.
struct Linearized {
    residual: SVector<f64, 6>,
    rows: usize,
    blocks: Vec<(usize, SMatrix<f64, 6, 6>, usize)>,
    robust: bool,
}

impl Problem {
    fn dims(&self) -> Dims {
        Dims::new(self.poses.len(), self.biases.len())
    }

    fn active(&self, index: usize) -> bool {
        !self.disabled.get(index).copied().unwrap_or(false)
    }

    /// Whitened squared residual of one factor at the current variables.
    pub fn squared_residual(&self, index: usize) -> Option<f64> {
        let factor = self.factors.get(index)?;
        let lin = self.linearize(factor)?;
        Some(lin.residual.rows(0, lin.rows).norm_squared())
    }

    fn cost(&self) -> f64 {
        (0..self.factors.len())
            .filter(|&i| self.active(i))
            .filter_map(|i| {
                let factor = self.factors.get(i)?;
                let lin = self.linearize(factor)?;
                let s = lin.residual.rows(0, lin.rows).norm_squared();
                Some(if lin.robust {
                    CAUCHY_SCALE.powi(2) * (s / CAUCHY_SCALE.powi(2)).ln_1p()
                } else {
                    s
                })
            })
            .sum::<f64>()
            * 0.5
    }

    fn linearize(&self, factor: &Factor) -> Option<Linearized> {
        let dims = self.dims();
        match factor {
            Factor::Relative {
                a,
                b,
                a_to_b,
                sigma_m,
                sigma_rad,
                robust,
            } => {
                let poses = (self.poses.get(*a)?, self.poses.get(*b)?);
                let columns = (dims.pose(*a), dims.pose(*b));
                Some(relative(
                    poses,
                    a_to_b,
                    (*sigma_m, *sigma_rad),
                    *robust,
                    columns,
                ))
            }
            Factor::Anchor {
                node,
                bias,
                pose,
                sqrt_info,
            } => self.anchor(*node, *bias, pose, sqrt_info),
            Factor::Prior {
                node,
                pose,
                sigma_m,
                sigma_rad,
            } => {
                let current = self.poses.get(*node)?;
                let info = isotropic(*sigma_m, *sigma_rad);
                Some(absolute(
                    current,
                    pose,
                    None,
                    &info,
                    dims.pose(*node),
                    false,
                ))
            }
            Factor::Direction {
                node,
                world,
                camera,
                sigma_rad,
            } => {
                let current = self.poses.get(*node)?;
                Some(direction(
                    current,
                    world,
                    camera,
                    *sigma_rad,
                    dims.pose(*node),
                ))
            }
            Factor::BiasPrior { bias, sqrt_info } => Some(bias_prior(
                self.biases.get(*bias)?,
                sqrt_info,
                dims.bias(*bias),
            )),
        }
    }

    fn anchor(
        &self,
        node: usize,
        bias: Option<usize>,
        pose: &Pose,
        sqrt_info: &Matrix6<f64>,
    ) -> Option<Linearized> {
        let dims = self.dims();
        let current = self.poses.get(node)?;
        let offset = match bias {
            Some(j) => Some((*self.biases.get(j)?, dims.bias(j))),
            None => None,
        };
        Some(absolute(
            current,
            pose,
            offset,
            sqrt_info,
            dims.pose(node),
            true,
        ))
    }

    /// Optimize all variables with damped Gauss-Newton steps.
    pub fn solve(&mut self, max_iterations: usize) -> SolveReport {
        let initial_cost = self.cost();
        let mut cost = initial_cost;
        let mut damping = 1e-4;
        let mut iterations = 0;
        while iterations < max_iterations && damping < 1e8 {
            iterations += 1;
            let Some(step) = self.step(damping) else {
                break;
            };
            let previous = (self.poses.clone(), self.biases.clone());
            self.apply(&step);
            let next = self.cost();
            if next.is_finite() && next <= cost {
                let improvement = cost - next;
                cost = next;
                damping = (damping * 0.3).max(1e-9);
                if improvement <= 1e-9 * cost.max(1.0) {
                    break;
                }
            } else {
                (self.poses, self.biases) = previous;
                damping *= 10.0;
            }
        }
        SolveReport {
            initial_cost,
            final_cost: cost,
            iterations,
        }
    }

    fn step(&self, damping: f64) -> Option<Vec<f64>> {
        let dims = self.dims();
        let mut system = BlockSystem::new(dims);
        for (index, factor) in self.factors.iter().enumerate() {
            if !self.active(index) {
                continue;
            }
            let Some(lin) = self.linearize(factor) else {
                continue;
            };
            let s = lin.residual.rows(0, lin.rows).norm_squared();
            let weight = if lin.robust {
                1.0 / (1.0 + s / CAUCHY_SCALE.powi(2))
            } else {
                1.0
            };
            system.add(&lin.blocks, &lin.residual, lin.rows, weight);
        }
        system.solve(damping)
    }

    fn apply(&mut self, step: &[f64]) {
        let dims = self.dims();
        for (i, pose) in self.poses.iter_mut().enumerate() {
            let at = dims.pose(i);
            let (Some(rho), Some(phi)) = (vector(step, at), vector(step, at + 3)) else {
                continue;
            };
            let rotation =
                crate::pose::unit(&(UnitQuaternion::from_scaled_axis(phi) * pose.rotation));
            *pose = Pose::from_parts(Translation3::from(pose.translation.vector + rho), rotation);
        }
        for (j, bias) in self.biases.iter_mut().enumerate() {
            if let Some(delta) = vector(step, dims.bias(j)) {
                *bias += delta;
            }
        }
    }
}

fn vector(values: &[f64], at: usize) -> Option<Vector3<f64>> {
    Some(Vector3::new(
        *values.get(at)?,
        *values.get(at + 1)?,
        *values.get(at + 2)?,
    ))
}

/// Relative-pose factor. `sigmas` are the position and rotation errors.
fn relative(
    (pa, pb): (&Pose, &Pose),
    a_to_b: &Pose,
    (sigma_m, sigma_rad): (f64, f64),
    robust: bool,
    (column_a, column_b): (usize, usize),
) -> Linearized {
    let ra = pa.rotation.to_rotation_matrix().into_inner();
    let rb = pb.rotation.to_rotation_matrix().into_inner();
    let d = pb.translation.vector - pa.translation.vector;
    let et = ra.transpose() * d - a_to_b.translation.vector;
    let er = (a_to_b.rotation.inverse() * pa.rotation.inverse() * pb.rotation).scaled_axis();
    let mut residual = SVector::<f64, 6>::zeros();
    residual.fixed_rows_mut::<3>(0).copy_from(&(et / sigma_m));
    residual.fixed_rows_mut::<3>(3).copy_from(&(er / sigma_rad));
    let mut ja = SMatrix::<f64, 6, 6>::zeros();
    let mut jb = SMatrix::<f64, 6, 6>::zeros();
    ja.fixed_view_mut::<3, 3>(0, 0)
        .copy_from(&(-ra.transpose() / sigma_m));
    ja.fixed_view_mut::<3, 3>(0, 3)
        .copy_from(&(ra.transpose() * skew(&d) / sigma_m));
    ja.fixed_view_mut::<3, 3>(3, 3)
        .copy_from(&(-rb.transpose() / sigma_rad));
    jb.fixed_view_mut::<3, 3>(0, 0)
        .copy_from(&(ra.transpose() / sigma_m));
    jb.fixed_view_mut::<3, 3>(3, 3)
        .copy_from(&(rb.transpose() / sigma_rad));
    Linearized {
        residual,
        rows: 6,
        blocks: vec![(column_a, ja, 6), (column_b, jb, 6)],
        robust,
    }
}

/// Whitening matrix of independent isotropic position and rotation errors.
fn isotropic(sigma_m: f64, sigma_rad: f64) -> Matrix6<f64> {
    let mut info = Matrix6::zeros();
    for i in 0..3 {
        info[(i, i)] = 1.0 / sigma_m;
        info[(i + 3, i + 3)] = 1.0 / sigma_rad;
    }
    info
}

/// Absolute-pose factor with an optional bias (value, column).
///
/// The unwhitened residual is `[p + bias - p_m; Log(R_m^-1 R)]`. With
/// `R' = Exp(phi) R`, its rotation Jacobian is `R^T`.
fn absolute(
    current: &Pose,
    measured: &Pose,
    bias: Option<(Vector3<f64>, usize)>,
    sqrt_info: &Matrix6<f64>,
    column: usize,
    robust: bool,
) -> Linearized {
    let r = current.rotation.to_rotation_matrix().into_inner();
    let offset = bias.map_or_else(Vector3::zeros, |(value, _)| value);
    let mut error = SVector::<f64, 6>::zeros();
    error
        .fixed_rows_mut::<3>(0)
        .copy_from(&(current.translation.vector + offset - measured.translation.vector));
    error
        .fixed_rows_mut::<3>(3)
        .copy_from(&(measured.rotation.inverse() * current.rotation).scaled_axis());
    let mut motion = SMatrix::<f64, 6, 6>::identity();
    motion
        .fixed_view_mut::<3, 3>(3, 3)
        .copy_from(&r.transpose());
    let mut blocks = vec![(column, sqrt_info * motion, 6)];
    if let Some((_, bias_column)) = bias {
        let mut jb = SMatrix::<f64, 6, 6>::zeros();
        jb.fixed_view_mut::<6, 3>(0, 0)
            .copy_from(&sqrt_info.fixed_view::<6, 3>(0, 0));
        blocks.push((bias_column, jb, 3));
    }
    Linearized {
        residual: sqrt_info * error,
        rows: 6,
        blocks,
        robust,
    }
}

/// Residual `R^T w - c`. With `R' = Exp(phi) R`, its rotation Jacobian is `R^T [w]x`.
fn direction(
    current: &Pose,
    world: &Vector3<f64>,
    camera: &Vector3<f64>,
    sigma_rad: f64,
    column: usize,
) -> Linearized {
    let rt = current
        .rotation
        .to_rotation_matrix()
        .into_inner()
        .transpose();
    let mut residual = SVector::<f64, 6>::zeros();
    residual
        .fixed_rows_mut::<3>(0)
        .copy_from(&((rt * world - camera) / sigma_rad));
    let mut jacobian = SMatrix::<f64, 6, 6>::zeros();
    jacobian
        .fixed_view_mut::<3, 3>(0, 3)
        .copy_from(&(rt * skew(world) / sigma_rad));
    Linearized {
        residual,
        rows: 3,
        blocks: vec![(column, jacobian, 6)],
        robust: true,
    }
}

fn bias_prior(value: &Vector3<f64>, sqrt_info: &Matrix3<f64>, column: usize) -> Linearized {
    let mut residual = SVector::<f64, 6>::zeros();
    residual
        .fixed_rows_mut::<3>(0)
        .copy_from(&(sqrt_info * value));
    let mut jacobian = SMatrix::<f64, 6, 6>::zeros();
    jacobian.fixed_view_mut::<3, 3>(0, 0).copy_from(sqrt_info);
    Linearized {
        residual,
        rows: 3,
        blocks: vec![(column, jacobian, 3)],
        robust: false,
    }
}

#[cfg(test)]
mod tests;
