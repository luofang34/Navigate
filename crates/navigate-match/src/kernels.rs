//! Matrix products and convolutions with fast kernels on GPU runtimes.
//!
//! The burn defaults for these operations are slow in a browser: without a matrix
//! accelerator, its WGSL convolution reached about 40 GFLOP/s and its matrix product
//! about 165 GFLOP/s in Chrome on an Apple GPU. The CubeCL kernels in `navigate-match-kernels` run
//! the same operations several times faster. The CPU runtime keeps the burn
//! operations, so the parity test on the CPU also checks the operation contract.
#[cfg(feature = "gpu")]
mod cube;

use burn::backend::{Backend, Dispatch, TensorMetadata, backend_extension, tensor::FloatTensor};
use burn::tensor::{Shape, Tensor, ops::ConvOptions};

/// Operations with a fast GPU implementation and a plain burn implementation.
#[backend_extension(
    Cube: cfg(feature = "gpu"),
    Flex: cfg(feature = "flex")
)]
pub trait MatchOps: Backend {
    /// `a · b + bias` over the last two dimensions; `bias` has the output width.
    ///
    /// `a` is `[.., M, K]`, or `[.., K, M]` with `trans_a`. `b` is `[.., K, N]`, or
    /// `[.., N, K]` with `trans_b`. Both have rank 3; a batch of 1 broadcasts.
    fn product(
        a: FloatTensor<Self>,
        b: FloatTensor<Self>,
        bias: FloatTensor<Self>,
        trans_a: bool,
        trans_b: bool,
        relu: bool,
    ) -> FloatTensor<Self> {
        let a = if trans_a {
            Self::float_swap_dims(a, 1, 2)
        } else {
            a
        };
        let b = if trans_b {
            Self::float_swap_dims(b, 1, 2)
        } else {
            b
        };
        let out = Self::float_matmul(a, b);
        let shape = out.shape();
        let width = shape[2];
        let bias = Self::float_expand(Self::float_reshape(bias, Shape::new([1, 1, width])), shape);
        let out = Self::float_add(out, bias);
        if relu { Self::relu(out) } else { out }
    }

    /// Square-kernel convolution with zero padding of half the kernel side.
    fn convolution(
        input: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: FloatTensor<Self>,
        stride: usize,
        relu: bool,
    ) -> FloatTensor<Self> {
        let kernel = weight.shape()[2];
        let options = ConvOptions::new([stride; 2], [kernel / 2; 2], [1; 2], 1);
        let out = Self::conv2d(input, weight, Some(bias), options);
        if relu { Self::relu(out) } else { out }
    }
}

#[cfg(feature = "flex")]
impl MatchOps for burn::backend::Flex {}

/// Operand layout and fused activation of one [`product`].
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Layout {
    pub trans_a: bool,
    pub trans_b: bool,
    pub relu: bool,
}

/// `a · b + bias` on rank-3 tensors; see [`MatchOps::product`].
pub(crate) fn product(a: Tensor<3>, b: Tensor<3>, bias: Tensor<1>, layout: Layout) -> Tensor<3> {
    Tensor::from_dispatch(<Dispatch as MatchOps>::product(
        a.into_dispatch(),
        b.into_dispatch(),
        bias.into_dispatch(),
        layout.trans_a,
        layout.trans_b,
        layout.relu,
    ))
}

/// `a · b + bias` on matrices.
pub(crate) fn product2(a: Tensor<2>, b: Tensor<2>, bias: Tensor<1>, layout: Layout) -> Tensor<2> {
    product(a.unsqueeze::<3>(), b.unsqueeze::<3>(), bias, layout).squeeze_dim(0)
}

/// Square-kernel convolution; see [`MatchOps::convolution`].
pub(crate) fn convolution(
    input: Tensor<4>,
    weight: Tensor<4>,
    bias: Tensor<1>,
    stride: usize,
    relu: bool,
) -> Tensor<4> {
    Tensor::from_dispatch(<Dispatch as MatchOps>::convolution(
        input.into_dispatch(),
        weight.into_dispatch(),
        bias.into_dispatch(),
        stride,
        relu,
    ))
}

#[cfg(test)]
mod tests;
