//! Resident Core ML graph and tensors behind the sparse feature boundary.
use crate::{InferenceError, Provider, features::Features, preprocessing, xfeat_dense};
use image::GrayImage;
use rustnn::{
    mlcontext::{
        Backend, MLContext, MLContextOptions, MLGraph, MLNamedTensors, MLPowerPreference, MLTensor,
        MLTensorDescriptor, RustNNOptions,
    },
    operator_enums::MLOperandDataType,
};
use sha2::{Digest, Sha256};
use std::path::Path;

const OUTPUTS: [(&str, usize); 3] = [
    ("dense_descriptors", 64),
    ("keypoint_logits", 65),
    ("reliability", 1),
];

pub(crate) struct DenseGraph {
    graph: MLGraph<'static>,
    input: MLTensor,
    channels: usize,
    outputs: Vec<(MLTensor, Vec<f32>)>,
    context: MLContext<'static>,
    pub digest: String,
    pub device: String,
}

fn failure(operation: &'static str, source: rustnn::error::Error) -> InferenceError {
    InferenceError::Rustnn {
        operation,
        source: Box::new(source),
    }
}

impl DenseGraph {
    pub fn statistics(&self) -> Option<crate::TensorIoStatistics> {
        match self.context.rustnn_backend_statistics()? {
            rustnn::mlcontext::BackendStatistics::Coreml(stats) => Some(stats.into()),
            _ => None,
        }
    }
    pub fn load_blocking(path: &Path, provider: Provider) -> Result<Self, InferenceError> {
        let data =
            rustnn::load_graph_from_path(path).map_err(|source| InferenceError::RustnnGraph {
                path: path.to_owned(),
                source: Box::new(source),
            })?;
        let channels = input_channels(&data)?;
        let mut digest = Sha256::new();
        // Hash the loaded topology and every resolved weight, including external files.
        digest.update(
            data.hash_identifier("xfeat-dense-image-v1", rustnn::graph::WeightsToHash::None),
        );
        let mut constants: Vec<_> = data.constant_operand_ids_to_handles.iter().collect();
        constants.sort_by_key(|(id, _)| **id);
        for (id, value) in constants {
            digest.update(id.to_le_bytes());
            digest.update((value.data.len() as u64).to_le_bytes());
            digest.update(&value.data);
        }
        let (power, accelerated) = match provider {
            Provider::Cpu => (MLPowerPreference::Default, false),
            Provider::CoreMlGpu => (MLPowerPreference::HighPerformance, true),
            Provider::CoreMlAne => (MLPowerPreference::LowPower, true),
            _ => {
                return Err(InferenceError::Invalid(
                    "rustnn Core ML requires CPU, GPU, or ANE".into(),
                ));
            }
        };
        let mut tuning = RustNNOptions::default();
        tuning.coreml.reuse_tensor_storage = true;
        let options = MLContextOptions::new(power, accelerated)
            .with_rustnn_backend_hint(Backend::Coreml)
            .with_rustnn_options(tuning);
        let mut context = MLContext::create(&options).map_err(|e| failure("create context", e))?;
        let device = format!("{:?}", context.rustnn_device());
        let graph = context
            .rustnn_build_graph(data)
            .map_err(|e| failure("build dense graph", e))?;
        let descriptor = MLTensorDescriptor::new(
            MLOperandDataType::Float32,
            vec![1, channels as u64, 600, 800],
        );
        let input = context
            .create_tensor(&descriptor.to_writable())
            .map_err(|e| failure("create input", e))?;
        let mut outputs = Vec::new();
        for (_, channels) in OUTPUTS {
            let descriptor = MLTensorDescriptor::new(
                MLOperandDataType::Float32,
                vec![1, channels as u64, 72, 100],
            );
            let tensor = context
                .create_tensor(&descriptor.to_readable())
                .map_err(|e| failure("create output", e))?;
            outputs.push((tensor, vec![0.0; channels * 72 * 100]));
        }
        Ok(Self {
            graph,
            input,
            channels,
            outputs,
            context,
            digest: format!("{:x}", digest.finalize()),
            device,
        })
    }

    pub fn extract(&mut self, image: &GrayImage, limit: usize) -> Result<Features, InferenceError> {
        let input = self.run_blocking(image)?;
        let [(_, descriptors), (_, logits), (_, reliability)] = self.outputs.as_mut_slice() else {
            return Err(InferenceError::Invalid("dense output buffers".into()));
        };
        xfeat_dense::decode_in_place(descriptors, logits, reliability, &input, image, limit)
    }

    pub fn grid_blocking(&mut self, image: &GrayImage) -> Result<Vec<f32>, InferenceError> {
        self.run_blocking(image)?;
        Ok(self.outputs[0].1.clone())
    }

    fn run_blocking(
        &mut self,
        image: &GrayImage,
    ) -> Result<preprocessing::InputImage, InferenceError> {
        let input = preprocessing::prepare(image, 800, 600, self.channels)?;
        self.context
            .write_tensor(&self.input, &input.data)
            .map_err(|e| failure("write image", e))?;
        let mut inputs = MLNamedTensors::new();
        inputs.insert("images", &self.input);
        let mut outputs = MLNamedTensors::new();
        for ((name, _), (tensor, _)) in OUTPUTS.iter().zip(&self.outputs) {
            outputs.insert(name, tensor);
        }
        self.context
            .dispatch(&mut self.graph, &inputs, &outputs)
            .map_err(|e| failure("dispatch dense graph", e))?;
        for (tensor, values) in &mut self.outputs {
            self.context
                .read_tensor(tensor, values)
                .map_err(|e| failure("read dense output", e))?;
            if values.iter().any(|v| !v.is_finite()) {
                return Err(InferenceError::Invalid("non-finite rustnn output".into()));
            }
        }
        Ok(input)
    }
}

fn input_channels(data: &rustnn::graph::GraphInfo) -> Result<usize, InferenceError> {
    let [id] = data.input_operands.as_slice() else {
        return Err(InferenceError::Invalid(
            "dense graph requires one image input".into(),
        ));
    };
    let operand = data
        .operand(*id)
        .ok_or_else(|| InferenceError::Invalid("missing images operand".into()))?;
    let shape = operand.descriptor.static_shape();
    if operand.name.as_deref() != Some("images")
        || operand.descriptor.data_type != rustnn::graph::DataType::Float32
    {
        return Err(InferenceError::Invalid("dense image name or type".into()));
    }
    match shape.as_deref() {
        Some([1, channels @ (1 | 3), 600, 800]) => Ok(*channels as usize),
        _ => Err(InferenceError::Invalid(
            "dense image shape must be 1x{1,3}x600x800".into(),
        )),
    }
}
