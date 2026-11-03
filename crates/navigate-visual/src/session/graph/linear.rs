//! Sparse symmetric block system and a preconditioned conjugate-gradient solve.
//!
//! Memory grows with the number of factors, not with the square of the
//! variable count, so the keyframe limit alone bounds the solver.

use nalgebra::{DMatrix, DVector, SMatrix, SVector};
use std::collections::BTreeMap;

/// Column layout: six columns for each pose, then three for each bias.
#[derive(Clone, Copy, Debug)]
pub(super) struct Dims {
    poses: usize,
    biases: usize,
}

impl Dims {
    pub fn new(poses: usize, biases: usize) -> Self {
        Self { poses, biases }
    }
    pub fn pose(&self, index: usize) -> usize {
        6 * index
    }
    pub fn bias(&self, index: usize) -> usize {
        6 * self.poses + 3 * index
    }
    fn total(&self) -> usize {
        6 * self.poses + 3 * self.biases
    }
    fn width(&self, column: usize) -> usize {
        if column < 6 * self.poses { 6 } else { 3 }
    }
    fn columns(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.poses)
            .map(|i| self.pose(i))
            .chain((0..self.biases).map(|j| self.bias(j)))
    }
}

/// Normal equations `H x = -g` stored as dense blocks keyed by start columns.
pub(super) struct BlockSystem {
    dims: Dims,
    blocks: BTreeMap<(usize, usize), DMatrix<f64>>,
    gradient: DVector<f64>,
}

const MAX_CG_ITERATIONS: usize = 4_000;

impl BlockSystem {
    pub fn new(dims: Dims) -> Self {
        Self {
            dims,
            blocks: BTreeMap::new(),
            gradient: DVector::zeros(dims.total()),
        }
    }

    /// Add one whitened factor. Each block is (start column, Jacobian, width).
    pub fn add(
        &mut self,
        jacobians: &[(usize, SMatrix<f64, 6, 6>, usize)],
        residual: &SVector<f64, 6>,
        rows: usize,
        weight: f64,
    ) {
        let r = residual.rows(0, rows);
        for (ci, ji, wi) in jacobians {
            let ji = ji.view((0, 0), (rows, *wi));
            let g = ji.transpose() * r * weight;
            let mut target = self.gradient.rows_mut(*ci, *wi);
            target += g;
            for (cj, jj, wj) in jacobians {
                let jj = jj.view((0, 0), (rows, *wj));
                let block = ji.transpose() * jj * weight;
                let entry = self
                    .blocks
                    .entry((*ci, *cj))
                    .or_insert_with(|| DMatrix::zeros(*wi, *wj));
                *entry += block;
            }
        }
    }

    /// Solve the damped system. `None` reports a singular or non-finite system.
    pub fn solve(mut self, damping: f64) -> Option<Vec<f64>> {
        let columns: Vec<usize> = self.dims.columns().collect();
        let mut preconditioner = Vec::with_capacity(columns.len());
        for &c in &columns {
            let width = self.dims.width(c);
            let block = self
                .blocks
                .entry((c, c))
                .or_insert_with(|| DMatrix::zeros(width, width));
            for k in 0..width {
                let value = block[(k, k)];
                block[(k, k)] = value + damping * value.abs().max(1e-6) + 1e-12;
            }
            preconditioner.push((c, block.clone().cholesky()?.inverse()));
        }
        let b = -&self.gradient;
        let x = self.conjugate_gradient(&b, &preconditioner)?;
        x.iter()
            .all(|v| v.is_finite())
            .then(|| x.iter().copied().collect())
    }

    fn multiply(&self, x: &DVector<f64>) -> DVector<f64> {
        let mut y = DVector::zeros(x.len());
        for ((i, j), block) in &self.blocks {
            let product = block * x.rows(*j, block.ncols());
            let mut target = y.rows_mut(*i, block.nrows());
            target += product;
        }
        y
    }

    fn precondition(
        &self,
        r: &DVector<f64>,
        preconditioner: &[(usize, DMatrix<f64>)],
    ) -> DVector<f64> {
        let mut z = DVector::zeros(r.len());
        for (c, inverse) in preconditioner {
            let product = inverse * r.rows(*c, inverse.nrows());
            z.rows_mut(*c, inverse.nrows()).copy_from(&product);
        }
        z
    }

    fn conjugate_gradient(
        &self,
        b: &DVector<f64>,
        preconditioner: &[(usize, DMatrix<f64>)],
    ) -> Option<DVector<f64>> {
        let tolerance = 1e-10 * b.norm().max(1e-30);
        let mut x = DVector::zeros(b.len());
        let mut r = b.clone();
        let mut z = self.precondition(&r, preconditioner);
        let mut p = z.clone();
        let mut rz = r.dot(&z);
        for _ in 0..MAX_CG_ITERATIONS {
            if r.norm() <= tolerance {
                break;
            }
            let ap = self.multiply(&p);
            let curvature = p.dot(&ap);
            if !(curvature.is_finite() && curvature > 0.0) {
                return None;
            }
            let alpha = rz / curvature;
            x.axpy(alpha, &p, 1.0);
            r.axpy(-alpha, &ap, 1.0);
            z = self.precondition(&r, preconditioner);
            let next = r.dot(&z);
            p = &z + &p * (next / rz);
            rz = next;
        }
        Some(x)
    }
}
