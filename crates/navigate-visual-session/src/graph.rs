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
use nalgebra::{Matrix3, SMatrix, SVector, Translation3, UnitQuaternion, Vector3};

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
    Anchor {
        node: usize,
        bias: Option<usize>,
        pose: Pose,
        /// Whitening matrix `L^-1`, where `L L^T` is the position covariance.
        position_sqrt_info: Matrix3<f64>,
        sigma_rad: f64,
    },
    /// Weak gauge prior for a pose graph component without anchors.
    Prior {
        node: usize,
        pose: Pose,
        sigma_m: f64,
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
                position_sqrt_info,
                sigma_rad,
            } => {
                let current = self.poses.get(*node)?;
                let offset = match bias {
                    Some(j) => Some((*self.biases.get(*j)?, dims.bias(*j))),
                    None => None,
                };
                let weights = (position_sqrt_info, *sigma_rad);
                Some(absolute(
                    current,
                    pose,
                    offset,
                    weights,
                    dims.pose(*node),
                    true,
                ))
            }
            Factor::Prior {
                node,
                pose,
                sigma_m,
                sigma_rad,
            } => {
                let current = self.poses.get(*node)?;
                let info = Matrix3::identity() / *sigma_m;
                Some(absolute(
                    current,
                    pose,
                    None,
                    (&info, *sigma_rad),
                    dims.pose(*node),
                    false,
                ))
            }
            Factor::BiasPrior { bias, sqrt_info } => Some(bias_prior(
                self.biases.get(*bias)?,
                sqrt_info,
                dims.bias(*bias),
            )),
        }
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
            let rotation = UnitQuaternion::from_scaled_axis(phi) * pose.rotation;
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

/// Absolute-pose factor with an optional bias (value, column).
fn absolute(
    current: &Pose,
    measured: &Pose,
    bias: Option<(Vector3<f64>, usize)>,
    (position_sqrt_info, sigma_rad): (&Matrix3<f64>, f64),
    column: usize,
    robust: bool,
) -> Linearized {
    let r = current.rotation.to_rotation_matrix().into_inner();
    let offset = bias.map_or_else(Vector3::zeros, |(value, _)| value);
    let ep = current.translation.vector + offset - measured.translation.vector;
    let er = (measured.rotation.inverse() * current.rotation).scaled_axis();
    let mut residual = SVector::<f64, 6>::zeros();
    residual
        .fixed_rows_mut::<3>(0)
        .copy_from(&(position_sqrt_info * ep));
    residual.fixed_rows_mut::<3>(3).copy_from(&(er / sigma_rad));
    let mut j = SMatrix::<f64, 6, 6>::zeros();
    j.fixed_view_mut::<3, 3>(0, 0).copy_from(position_sqrt_info);
    j.fixed_view_mut::<3, 3>(3, 3)
        .copy_from(&(r.transpose() / sigma_rad));
    let mut blocks = vec![(column, j, 6)];
    if let Some((_, bias_column)) = bias {
        let mut jb = SMatrix::<f64, 6, 6>::zeros();
        jb.fixed_view_mut::<3, 3>(0, 0)
            .copy_from(position_sqrt_info);
        blocks.push((bias_column, jb, 3));
    }
    Linearized {
        residual,
        rows: 6,
        blocks,
        robust,
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
