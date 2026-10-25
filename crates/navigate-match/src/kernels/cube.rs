//! Launches of the CubeCL kernels on burn's CubeCL tensors.
use super::MatchOps;
use burn::backend::{cubecl::dtype_to_storage_type, tensor::FloatTensor};
use burn::tensor::DType;
use burn_cubecl::{CubeBackend, kernel::into_contiguous, tensor::CubeTensor};
use cubecl::{CubeCount, CubeDim};
use navigate_match_kernels as source;

/// An empty contiguous tensor on the device of `like`.
fn empty(like: &CubeTensor, shape: Vec<usize>, dtype: DType) -> CubeTensor {
    let count: usize = shape.iter().product();
    let buffer = like.client.empty(count * dtype.size());
    CubeTensor::new_contiguous(
        like.client.clone(),
        like.device.clone(),
        shape.into(),
        buffer,
        dtype,
    )
}

/// Kernel size arguments as a small `u32` tensor.
fn dims(like: &CubeTensor, values: &[usize]) -> CubeTensor {
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|&v| (v as u32).to_le_bytes())
        .collect();
    let buffer = like.client.create_from_slice(&bytes);
    CubeTensor::new_contiguous(
        like.client.clone(),
        like.device.clone(),
        vec![values.len()].into(),
        buffer,
        DType::U32,
    )
}

fn grid(columns: usize, rows: usize, batch: usize) -> CubeCount {
    let cubes = |v: usize| v.div_ceil(source::TILE) as u32;
    CubeCount::Static(cubes(columns), cubes(rows), batch as u32)
}

impl MatchOps for CubeBackend {
    fn product(
        a: FloatTensor<Self>,
        b: FloatTensor<Self>,
        bias: FloatTensor<Self>,
        trans_a: bool,
        trans_b: bool,
        relu: bool,
    ) -> FloatTensor<Self> {
        let (a, b, bias) = (
            into_contiguous(a),
            into_contiguous(b),
            into_contiguous(bias),
        );
        let ad = a.meta.shape().to_vec();
        let bd = b.meta.shape().to_vec();
        let (m, k) = if trans_a {
            (ad[2], ad[1])
        } else {
            (ad[1], ad[2])
        };
        let n = if trans_b { bd[1] } else { bd[2] };
        let batch = ad[0].max(bd[0]);
        let stride = |d: &[usize]| if d[0] == 1 { 0 } else { d[1] * d[2] };
        let sizes = dims(&a, &[m, n, k, stride(&ad), stride(&bd)]);
        let dtype = a.dtype;
        let out = empty(&a, vec![batch, m, n], dtype);
        source::matmul::launch(
            &out.client,
            grid(n, m, batch),
            CubeDim::new_1d(source::UNITS as u32),
            a.into_tensor_arg(),
            b.into_tensor_arg(),
            bias.into_tensor_arg(),
            out.clone().into_tensor_arg(),
            sizes.into_tensor_arg(),
            trans_a,
            trans_b,
            relu,
            dtype_to_storage_type(dtype),
        );
        out
    }

    fn convolution(
        input: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: FloatTensor<Self>,
        stride: usize,
        relu: bool,
    ) -> FloatTensor<Self> {
        let (input, weight, bias) = (
            into_contiguous(input),
            into_contiguous(weight),
            into_contiguous(bias),
        );
        let id = input.meta.shape().to_vec();
        let wd = weight.meta.shape().to_vec();
        let (images, cin, h, w) = (id[0], id[1], id[2], id[3]);
        let (cout, kernel) = (wd[0], wd[2]);
        let pad = kernel / 2;
        let ho = (h + 2 * pad - kernel) / stride + 1;
        let wo = (w + 2 * pad - kernel) / stride + 1;
        let sizes = dims(&input, &[cin, h, w, cout, ho, wo, images]);
        let dtype = input.dtype;
        let out = empty(&input, vec![images, cout, ho, wo], dtype);
        source::convolution::launch(
            &out.client,
            grid(images * ho * wo, cout, 1),
            CubeDim::new_1d(source::UNITS as u32),
            input.into_tensor_arg(),
            weight.into_tensor_arg(),
            bias.into_tensor_arg(),
            out.clone().into_tensor_arg(),
            sizes.into_tensor_arg(),
            kernel,
            stride,
            relu,
            dtype_to_storage_type(dtype),
        );
        out
    }
}
