//! Unified [`ModelInference`] session wrapper for ONNX Runtime (`ort`).
use crate::{ExecutionConfig, InferenceError, native_runtime};
use navigate_visual::{EngineKind, HostTensor, ModelInference, NamedTensor};
use ort::{session::Session, value::TensorRef};
use std::path::Path;

/// Native ONNX Runtime session implementing [`ModelInference`].
pub struct OrtModelSession {
    session: Session,
    output_names: Vec<String>,
}

impl OrtModelSession {
    /// Load an ONNX graph from `path` with explicit `config`.
    ///
    /// # Errors
    ///
    /// Returns [`InferenceError`] when `path` cannot be opened or session creation fails.
    pub fn load_blocking(path: &Path, config: &ExecutionConfig) -> Result<Self, InferenceError> {
        let session = native_runtime::session_blocking(path, config)?;
        let output_names = session
            .outputs()
            .iter()
            .map(|o| o.name().to_owned())
            .collect();
        Ok(Self {
            session,
            output_names,
        })
    }

    /// Borrow the underlying ONNX Runtime [`Session`].
    pub fn session(&self) -> &Session {
        &self.session
    }

    /// Return the primary input tensor name of the ONNX model graph.
    ///
    /// # Errors
    ///
    /// Returns [`InferenceError`] if the session has no input tensors.
    pub fn primary_input_name(&self) -> Result<&str, InferenceError> {
        self.session
            .inputs()
            .first()
            .map(|i| i.name())
            .ok_or_else(|| InferenceError::Invalid("ORT model has no input tensors".into()))
    }
}

impl ModelInference for OrtModelSession {
    type Tensor = HostTensor;
    type Error = InferenceError;

    fn engine_kind(&self) -> EngineKind {
        EngineKind::Ort
    }

    fn forward(
        &mut self,
        inputs: &[NamedTensor<HostTensor>],
    ) -> Result<Vec<NamedTensor<HostTensor>>, InferenceError> {
        let mut ort_inputs = Vec::with_capacity(inputs.len());
        for named in inputs {
            let view = TensorRef::from_array_view((
                named.tensor.shape().to_vec(),
                named.tensor.as_slice(),
            ))
            .map_err(|e| native_runtime::runtime("borrow input tensor", e))?
            .into_dyn();
            ort_inputs.push((named.name.clone(), view));
        }
        let outputs = self
            .session
            .run(ort_inputs)
            .map_err(|e| native_runtime::runtime("run OrtModelSession", e))?;
        let mut results = Vec::with_capacity(self.output_names.len());
        for name in &self.output_names {
            let (shape, slice) = outputs
                .get(name.as_str())
                .ok_or_else(|| InferenceError::Invalid(format!("missing output {name}")))?
                .try_extract_tensor::<f32>()
                .map_err(|e| native_runtime::runtime("extract output tensor", e))?;
            let dims: Vec<usize> = shape
                .iter()
                .map(|&d| {
                    usize::try_from(d)
                        .map_err(|_| InferenceError::Invalid("negative output dimension".into()))
                })
                .collect::<Result<_, _>>()?;
            let tensor = HostTensor::new(dims, slice.to_vec())
                .map_err(|_| InferenceError::Invalid(format!("non-finite output {name}")))?;
            results.push(NamedTensor::new(name.clone(), tensor));
        }
        Ok(results)
    }
}
