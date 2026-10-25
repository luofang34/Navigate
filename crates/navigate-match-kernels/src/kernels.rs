//! CubeCL kernels: a register-tiled matrix product and a convolution that forms its
//! input columns on the fly (implicit GEMM).
//!
//! Both kernels need no matrix accelerator, so they compile to WGSL for browsers as
//! well as to MSL, SPIR-V and CUDA. Each cube computes a 128 × 128 output tile with
//! 256 units, 8 × 8 outputs per unit, and steps through `K` in slices of 16 that the
//! cube stages in shared memory.
// `#[cube(launch)]` generates a module, a struct and a launch function for each
// kernel without documentation; the kernel functions carry the documentation.
#![allow(missing_docs)]

use cubecl::{cube, prelude::*};

/// Output rows and columns of the tile that one cube computes.
pub const TILE: usize = 128;
const SLICE: usize = 16;
const PER_UNIT: usize = 8;
/// Units of one cube.
pub const UNITS: usize = (TILE / PER_UNIT) * (TILE / PER_UNIT);

/// `out[z] = a[z] · b[z] + bias`, each batch `z` contiguous; a batch stride of 0 broadcasts.
///
/// `dims` holds `[m, n, k, a_batch_stride, b_batch_stride]`.
#[cube(launch)]
pub fn matmul<F: Float>(
    a: &Tensor<F>,
    b: &Tensor<F>,
    bias: &Tensor<F>,
    out: &mut Tensor<F>,
    dims: &Tensor<u32>,
    #[comptime] trans_a: bool,
    #[comptime] trans_b: bool,
    #[comptime] relu: bool,
    #[define(F)] _dtype: ElemType,
) {
    let m = dims[0] as usize;
    let n = dims[1] as usize;
    let k = dims[2] as usize;
    let a_off = CUBE_POS_Z as usize * dims[3] as usize;
    let b_off = CUBE_POS_Z as usize * dims[4] as usize;
    let o_off = CUBE_POS_Z as usize * m * n;
    let unit = UNIT_POS as usize;
    let row0 = CUBE_POS_Y as usize * TILE;
    let col0 = CUBE_POS_X as usize * TILE;
    let mut sa = Shared::<[F]>::new_slice(SLICE * TILE);
    let mut sb = Shared::<[F]>::new_slice(SLICE * TILE);
    let mut acc = Array::<F>::new(PER_UNIT * PER_UNIT);
    let mut k0 = 0usize;
    while k0 < k {
        #[unroll]
        for i in 0..(TILE * SLICE / UNITS) {
            let idx = unit + i * UNITS;
            let mut v = F::new(0.0_f32);
            if comptime![trans_a] {
                let r = idx % TILE;
                let c = idx / TILE;
                if row0 + r < m && k0 + c < k {
                    v = a[a_off + (k0 + c) * m + row0 + r];
                }
                sa[c * TILE + r] = v;
            } else {
                let r = idx / SLICE;
                let c = idx % SLICE;
                if row0 + r < m && k0 + c < k {
                    v = a[a_off + (row0 + r) * k + k0 + c];
                }
                sa[c * TILE + r] = v;
            }
        }
        #[unroll]
        for i in 0..(TILE * SLICE / UNITS) {
            let idx = unit + i * UNITS;
            let mut v = F::new(0.0_f32);
            if comptime![trans_b] {
                let r = idx % SLICE;
                let c = idx / SLICE;
                if k0 + r < k && col0 + c < n {
                    v = b[b_off + (col0 + c) * k + k0 + r];
                }
                sb[r * TILE + c] = v;
            } else {
                let r = idx / TILE;
                let c = idx % TILE;
                if k0 + r < k && col0 + c < n {
                    v = b[b_off + (k0 + r) * n + col0 + c];
                }
                sb[r * TILE + c] = v;
            }
        }
        sync_cube();
        accumulate::<F>(&sa, &sb, &mut acc, unit);
        sync_cube();
        k0 += SLICE;
    }
    let tx = unit % (TILE / PER_UNIT);
    let ty = unit / (TILE / PER_UNIT);
    #[unroll]
    for i in 0..PER_UNIT {
        let row = row0 + ty * PER_UNIT + i;
        #[unroll]
        for j in 0..PER_UNIT {
            let col = col0 + tx * PER_UNIT + j;
            if row < m && col < n {
                let mut v = acc[i * PER_UNIT + j] + bias[col];
                if comptime![relu] {
                    v = F::max(v, F::new(0.0_f32));
                }
                out[o_off + row * n + col] = v;
            }
        }
    }
}

/// `out[img, co, y, x] = bias[co] + Σ weight[co, ci, ky, kx] · input[img, ci, y·s + ky − p, x·s + kx − p]`.
///
/// `dims` holds `[channels_in, height_in, width_in, channels_out, height_out,
/// width_out, images]`. Positions outside the input read zero.
#[cube(launch)]
pub fn convolution<F: Float>(
    input: &Tensor<F>,
    weight: &Tensor<F>,
    bias: &Tensor<F>,
    out: &mut Tensor<F>,
    dims: &Tensor<u32>,
    #[comptime] kernel: usize,
    #[comptime] stride: usize,
    #[comptime] relu: bool,
    #[define(F)] _dtype: ElemType,
) {
    let cin = dims[0] as usize;
    let hin = dims[1] as usize;
    let win = dims[2] as usize;
    let cout = dims[3] as usize;
    let wout = dims[5] as usize;
    let plane = dims[4] as usize * wout;
    let n = dims[6] as usize * plane;
    let taps = kernel * kernel;
    let pad = kernel / 2;
    let k = cin * taps;
    let unit = UNIT_POS as usize;
    let row0 = CUBE_POS_Y as usize * TILE;
    let col0 = CUBE_POS_X as usize * TILE;
    // UNITS is a multiple of TILE, so each unit loads the same output column every slice.
    let column = col0 + unit % TILE;
    let column_valid = column < n;
    let pixel = column % plane;
    let base = (column / plane) * cin * hin * win;
    let y0 = (pixel / wout) * stride;
    let x0 = (pixel % wout) * stride;
    let mut sa = Shared::<[F]>::new_slice(SLICE * TILE);
    let mut sb = Shared::<[F]>::new_slice(SLICE * TILE);
    let mut acc = Array::<F>::new(PER_UNIT * PER_UNIT);
    let mut k0 = 0usize;
    while k0 < k {
        #[unroll]
        for i in 0..(TILE * SLICE / UNITS) {
            let idx = unit + i * UNITS;
            let r = idx / SLICE;
            let c = idx % SLICE;
            let mut v = F::new(0.0_f32);
            if row0 + r < cout && k0 + c < k {
                v = weight[(row0 + r) * k + k0 + c];
            }
            sa[c * TILE + r] = v;
        }
        #[unroll]
        for i in 0..(TILE * SLICE / UNITS) {
            let idx = unit + i * UNITS;
            let r = idx / TILE;
            let kk = k0 + r;
            let mut v = F::new(0.0_f32);
            if kk < k && column_valid {
                let tap = kk % taps;
                let y = y0 + tap / kernel;
                let x = x0 + tap % kernel;
                if y >= pad && x >= pad && y < hin + pad && x < win + pad {
                    v = input[base + ((kk / taps) * hin + y - pad) * win + x - pad];
                }
            }
            sb[r * TILE + idx % TILE] = v;
        }
        sync_cube();
        accumulate::<F>(&sa, &sb, &mut acc, unit);
        sync_cube();
        k0 += SLICE;
    }
    let tx = unit % (TILE / PER_UNIT);
    let ty = unit / (TILE / PER_UNIT);
    #[unroll]
    for i in 0..PER_UNIT {
        let co = row0 + ty * PER_UNIT + i;
        #[unroll]
        for j in 0..PER_UNIT {
            let col = col0 + tx * PER_UNIT + j;
            if co < cout && col < n {
                let mut v = acc[i * PER_UNIT + j] + bias[co];
                if comptime![relu] {
                    v = F::max(v, F::new(0.0_f32));
                }
                out[((col / plane) * cout + co) * plane + col % plane] = v;
            }
        }
    }
}

/// Multiply the staged slices into the unit's 8 × 8 accumulator block.
#[cube]
fn accumulate<F: Float>(sa: &Shared<[F]>, sb: &Shared<[F]>, acc: &mut Array<F>, unit: usize) {
    let tx = unit % (TILE / PER_UNIT);
    let ty = unit / (TILE / PER_UNIT);
    let mut fa = Array::<F>::new(PER_UNIT);
    let mut fb = Array::<F>::new(PER_UNIT);
    #[unroll]
    for kk in 0..SLICE {
        #[unroll]
        for i in 0..PER_UNIT {
            fa[i] = sa[kk * TILE + ty * PER_UNIT + i];
            fb[i] = sb[kk * TILE + tx * PER_UNIT + i];
        }
        #[unroll]
        for i in 0..PER_UNIT {
            #[unroll]
            for j in 0..PER_UNIT {
                acc[i * PER_UNIT + j] += fa[i] * fb[j];
            }
        }
    }
}
