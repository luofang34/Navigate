//! ResNet-FPN 8/2 feature extractor with batch normalization folded into the convolutions.
use super::{interpolate::upsample_align_corners, layers::Conv};
use crate::{MatchError, Weights};
use burn::tensor::{
    Device, Tensor,
    activation::{leaky_relu, relu},
};

/// `nn.LeakyReLU()` default slope.
const LEAKY_SLOPE: f64 = 0.01;
/// Channels after the stem and in each of the three stages.
const INITIAL: usize = 128;
const STAGES: [usize; 3] = [128, 196, 256];

/// Coarse (1/8) and fine (1/2) feature maps for a batch of images.
pub(crate) struct Features {
    /// `[N, 256, H/8, W/8]`.
    pub coarse: Tensor<4>,
    /// `[N, 128, H/2, W/2]`.
    pub fine: Tensor<4>,
}

#[derive(Clone)]
struct Block {
    conv1: Conv,
    conv2: Conv,
    downsample: Option<Conv>,
}

impl Block {
    fn load(
        weights: &mut Weights,
        name: &str,
        [input, output, stride]: [usize; 3],
        device: &Device,
    ) -> Result<Self, MatchError> {
        let downsample = (stride != 1)
            .then(|| {
                Conv::load(
                    weights,
                    &format!("{name}.downsample.0"),
                    [output, input, 1],
                    stride,
                    true,
                    device,
                )
            })
            .transpose()?;
        Ok(Self {
            conv1: Conv::load(
                weights,
                &format!("{name}.conv1"),
                [output, input, 3],
                stride,
                true,
                device,
            )?,
            conv2: Conv::load(
                weights,
                &format!("{name}.conv2"),
                [output, output, 3],
                1,
                true,
                device,
            )?,
            downsample,
        })
    }

    fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let y = self.conv2.forward(self.conv1.forward_relu(x.clone()));
        let shortcut = match &self.downsample {
            Some(downsample) => downsample.forward(x),
            None => x,
        };
        relu(shortcut + y)
    }
}

/// Two convolutions with a leaky ReLU between them, as `layer*_outconv2`.
#[derive(Clone)]
struct Merge {
    first: Conv,
    second: Conv,
}

impl Merge {
    fn load(
        weights: &mut Weights,
        name: &str,
        [width, output]: [usize; 2],
        device: &Device,
    ) -> Result<Self, MatchError> {
        Ok(Self {
            first: Conv::load(
                weights,
                &format!("{name}.0"),
                [width, width, 3],
                1,
                true,
                device,
            )?,
            second: Conv::load(
                weights,
                &format!("{name}.3"),
                [output, width, 3],
                1,
                false,
                device,
            )?,
        })
    }

    fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        self.second
            .forward(leaky_relu(self.first.forward(x), LEAKY_SLOPE))
    }
}

#[derive(Clone)]
pub(crate) struct Backbone {
    stem: Conv,
    stages: Vec<[Block; 2]>,
    layer3_out: Conv,
    layer2_out: Conv,
    layer2_merge: Merge,
    layer1_out: Conv,
    layer1_merge: Merge,
}

impl Backbone {
    pub(crate) fn load(weights: &mut Weights, device: &Device) -> Result<Self, MatchError> {
        let mut stages = Vec::with_capacity(STAGES.len());
        let mut input = INITIAL;
        for (index, output) in STAGES.into_iter().enumerate() {
            let stride = if index == 0 { 1 } else { 2 };
            let name = format!("backbone.layer{}", index + 1);
            stages.push([
                Block::load(
                    weights,
                    &format!("{name}.0"),
                    [input, output, stride],
                    device,
                )?,
                Block::load(weights, &format!("{name}.1"), [output, output, 1], device)?,
            ]);
            input = output;
        }
        let [c1, c2, c3] = STAGES;
        Ok(Self {
            stem: Conv::load(weights, "backbone.conv1", [INITIAL, 1, 7], 2, true, device)?,
            stages,
            layer3_out: Conv::load(
                weights,
                "backbone.layer3_outconv",
                [c3, c3, 1],
                1,
                false,
                device,
            )?,
            layer2_out: Conv::load(
                weights,
                "backbone.layer2_outconv",
                [c3, c2, 1],
                1,
                false,
                device,
            )?,
            layer2_merge: Merge::load(weights, "backbone.layer2_outconv2", [c3, c2], device)?,
            layer1_out: Conv::load(
                weights,
                "backbone.layer1_outconv",
                [c2, c1, 1],
                1,
                false,
                device,
            )?,
            layer1_merge: Merge::load(weights, "backbone.layer1_outconv2", [c2, c1], device)?,
        })
    }

    /// Extract features from `[N, 1, H, W]` images with values in 0 through 1.
    pub(crate) fn forward(&self, images: Tensor<4>) -> Features {
        let mut x = self.stem.forward_relu(images);
        let mut levels = Vec::with_capacity(self.stages.len());
        for [first, second] in &self.stages {
            x = second.forward(first.forward(x));
            levels.push(x.clone());
        }
        let [x1, x2, x3] = [levels.remove(0), levels.remove(0), x];
        let coarse = self.layer3_out.forward(x3);
        let x2_out = self.layer2_out.forward(x2);
        let up = upsample_align_corners(coarse.clone(), spatial(&x2_out));
        let x2_out = self.layer2_merge.forward(x2_out + up);
        let x1_out = self.layer1_out.forward(x1);
        let up = upsample_align_corners(x2_out, spatial(&x1_out));
        let fine = self.layer1_merge.forward(x1_out + up);
        Features { coarse, fine }
    }
}

fn spatial(x: &Tensor<4>) -> [usize; 2] {
    let [_, _, h, w] = x.dims();
    [h, w]
}
