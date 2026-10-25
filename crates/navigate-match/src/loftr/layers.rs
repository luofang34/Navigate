//! Inference layers that hold device tensors made from named weights.
use crate::{
    MatchError, Weights,
    kernels::{self, Layout},
};
use burn::tensor::{Device, Tensor, TensorData};

/// Upload host values as a tensor of the given shape.
pub(crate) fn upload<const D: usize>(
    values: Vec<f32>,
    shape: [usize; D],
    device: &Device,
) -> Tensor<D> {
    Tensor::from_data(TensorData::new(values, shape), device)
}

/// Load `{name}.bias` of `width` values, or zeros when the layer has no bias.
fn bias(
    weights: &mut Weights,
    name: &str,
    width: usize,
    present: bool,
    device: &Device,
) -> Result<Tensor<1>, MatchError> {
    let values = if present {
        weights.take(&format!("{name}.bias"), &[width])?
    } else {
        vec![0.0; width]
    };
    Ok(upload(values, [width], device))
}

/// A square-kernel 2D convolution with half-kernel zero padding.
#[derive(Clone, Debug)]
pub(crate) struct Conv {
    weight: Tensor<4>,
    bias: Tensor<1>,
    stride: usize,
}

impl Conv {
    /// Load `{name}.weight` of shape `[out, in, k, k]` and, with `has_bias`, `{name}.bias`.
    pub(crate) fn load(
        weights: &mut Weights,
        name: &str,
        [out, input, kernel]: [usize; 3],
        stride: usize,
        has_bias: bool,
        device: &Device,
    ) -> Result<Self, MatchError> {
        let shape = [out, input, kernel, kernel];
        let weight = upload(
            weights.take(&format!("{name}.weight"), &shape)?,
            shape,
            device,
        );
        Ok(Self {
            weight,
            bias: bias(weights, name, out, has_bias, device)?,
            stride,
        })
    }

    pub(crate) fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        kernels::convolution(
            x,
            self.weight.clone(),
            self.bias.clone(),
            self.stride,
            false,
        )
    }

    /// The convolution followed by `max(0, x)`, in one kernel on GPU runtimes.
    pub(crate) fn forward_relu(&self, x: Tensor<4>) -> Tensor<4> {
        kernels::convolution(x, self.weight.clone(), self.bias.clone(), self.stride, true)
    }
}

/// A linear map `y = x Wᵀ + b` over the last dimension.
#[derive(Clone, Debug)]
pub(crate) struct Linear {
    /// Transposed weight, `[in, out]`, contiguous.
    weight: Tensor<2>,
    bias: Tensor<1>,
    output: usize,
}

impl Linear {
    /// Load `{name}.weight` of shape `[out, in]` and, with `has_bias`, `{name}.bias`.
    pub(crate) fn load(
        weights: &mut Weights,
        name: &str,
        [out, input]: [usize; 2],
        has_bias: bool,
        device: &Device,
    ) -> Result<Self, MatchError> {
        let values = weights.take(&format!("{name}.weight"), &[out, input])?;
        let mut transposed = vec![0.0; values.len()];
        for (row, chunk) in values.chunks(input.max(1)).enumerate() {
            for (column, value) in chunk.iter().enumerate() {
                transposed[column * out + row] = *value;
            }
        }
        Ok(Self {
            weight: upload(transposed, [input, out], device),
            bias: bias(weights, name, out, has_bias, device)?,
            output: out,
        })
    }

    pub(crate) fn forward<const D: usize>(&self, x: Tensor<D>) -> Tensor<D> {
        self.apply(x, false)
    }

    /// The map followed by `max(0, x)`, in one kernel on GPU runtimes.
    pub(crate) fn forward_relu<const D: usize>(&self, x: Tensor<D>) -> Tensor<D> {
        self.apply(x, true)
    }

    fn apply<const D: usize>(&self, x: Tensor<D>, relu: bool) -> Tensor<D> {
        let mut shape = x.dims();
        let input = shape[D - 1];
        let rows = shape.iter().take(D - 1).product::<usize>();
        let layout = Layout {
            relu,
            ..Layout::default()
        };
        let y = kernels::product2(
            x.reshape([rows, input]),
            self.weight.clone(),
            self.bias.clone(),
            layout,
        );
        shape[D - 1] = self.output;
        y.reshape(shape)
    }
}

/// Layer normalization over the last dimension, with PyTorch's default epsilon.
#[derive(Clone, Debug)]
pub(crate) struct LayerNorm {
    weight: Tensor<1>,
    bias: Tensor<1>,
}

const LAYER_NORM_EPSILON: f64 = 1e-5;

impl LayerNorm {
    pub(crate) fn load(
        weights: &mut Weights,
        name: &str,
        width: usize,
        device: &Device,
    ) -> Result<Self, MatchError> {
        Ok(Self {
            weight: upload(
                weights.take(&format!("{name}.weight"), &[width])?,
                [width],
                device,
            ),
            bias: upload(
                weights.take(&format!("{name}.bias"), &[width])?,
                [width],
                device,
            ),
        })
    }

    pub(crate) fn forward<const D: usize>(&self, x: Tensor<D>) -> Tensor<D> {
        let mean = x.clone().mean_dim(D - 1);
        let centred = x - mean;
        let variance = centred.clone().powi_scalar(2).mean_dim(D - 1);
        let normalized = centred / (variance + LAYER_NORM_EPSILON).sqrt();
        normalized * self.weight.clone().unsqueeze::<D>() + self.bias.clone().unsqueeze::<D>()
    }
}
