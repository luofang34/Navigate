//! Inference layers that hold device tensors made from named weights.
use crate::{MatchError, Weights};
use burn::tensor::{Device, Tensor, TensorData, module::conv2d, ops::ConvOptions};

/// Upload host values as a tensor of the given shape.
pub(crate) fn upload<const D: usize>(
    values: Vec<f32>,
    shape: [usize; D],
    device: &Device,
) -> Tensor<D> {
    Tensor::from_data(TensorData::new(values, shape), device)
}

/// A 2D convolution with an optional bias.
#[derive(Clone, Debug)]
pub(crate) struct Conv {
    weight: Tensor<4>,
    bias: Option<Tensor<1>>,
    options: ConvOptions<2>,
}

impl Conv {
    /// Load `{name}.weight` of shape `[out, in, k, k]` and, with `bias`, `{name}.bias`.
    pub(crate) fn load(
        weights: &mut Weights,
        name: &str,
        [out, input, kernel]: [usize; 3],
        stride: usize,
        bias: bool,
        device: &Device,
    ) -> Result<Self, MatchError> {
        let shape = [out, input, kernel, kernel];
        let weight = upload(
            weights.take(&format!("{name}.weight"), &shape)?,
            shape,
            device,
        );
        let bias = if bias {
            Some(upload(
                weights.take(&format!("{name}.bias"), &[out])?,
                [out],
                device,
            ))
        } else {
            None
        };
        let padding = kernel / 2;
        Ok(Self {
            weight,
            bias,
            options: ConvOptions::new([stride; 2], [padding; 2], [1; 2], 1),
        })
    }

    pub(crate) fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        conv2d(
            x,
            self.weight.clone(),
            self.bias.clone(),
            self.options.clone(),
        )
    }
}

/// A linear map `y = x Wᵀ + b` over the last dimension.
#[derive(Clone, Debug)]
pub(crate) struct Linear {
    /// Transposed weight, `[in, out]`.
    weight: Tensor<2>,
    bias: Option<Tensor<1>>,
    output: usize,
}

impl Linear {
    /// Load `{name}.weight` of shape `[out, in]` and, with `bias`, `{name}.bias`.
    pub(crate) fn load(
        weights: &mut Weights,
        name: &str,
        [out, input]: [usize; 2],
        bias: bool,
        device: &Device,
    ) -> Result<Self, MatchError> {
        let weight = upload(
            weights.take(&format!("{name}.weight"), &[out, input])?,
            [out, input],
            device,
        )
        .transpose();
        let bias = if bias {
            Some(upload(
                weights.take(&format!("{name}.bias"), &[out])?,
                [out],
                device,
            ))
        } else {
            None
        };
        Ok(Self {
            weight,
            bias,
            output: out,
        })
    }

    pub(crate) fn forward<const D: usize>(&self, x: Tensor<D>) -> Tensor<D> {
        let mut shape = x.dims();
        let input = shape[D - 1];
        let rows = shape.iter().take(D - 1).product::<usize>();
        let mut y = x.reshape([rows, input]).matmul(self.weight.clone());
        if let Some(bias) = &self.bias {
            y = y + bias.clone().unsqueeze::<2>();
        }
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
