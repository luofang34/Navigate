//! Framework-independent tensor and model inference traits.
//!
//! Upper-layer visual pipelines pass inputs and outputs as [`NamedTensor`] values
//! through [`ModelInference`] or [`AsyncModelInference`]. Concrete engines (`burn`,
//! WebNN, ONNX Runtime, Core ML, and vendor neural processing units) implement these
//! traits without leaking framework types into domain crates.
use crate::VisualError;

/// Active inference runtime track.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EngineKind {
    /// Browser WebNN runtime (`navigator.ml`).
    WebNn,
    /// Browser Burn WebGPU runtime (`burn/webgpu`).
    BurnWebGpu,
    /// Native Burn GPU runtime (`wgpu`, `metal`, `vulkan`, or `cuda`).
    BurnGpu,
    /// Burn CPU runtime (`burn/flex`).
    BurnCpu,
    /// Native ONNX Runtime session (`ort`).
    Ort,
    /// Native Core ML graph session (`rustnn`).
    RustnnCoreMl,
    /// Embedded vendor neural processing unit runtime.
    VendorNpu,
}

impl EngineKind {
    /// Return the stable identifier string of this runtime track.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::WebNn => "webnn",
            Self::BurnWebGpu => "burn-webgpu",
            Self::BurnGpu => "burn-gpu",
            Self::BurnCpu => "burn-cpu",
            Self::Ort => "ort",
            Self::RustnnCoreMl => "rustnn-coreml",
            Self::VendorNpu => "vendor-npu",
        }
    }
}

/// Contiguous float32 tensor interface shared by inference engines.
pub trait GenericTensor {
    /// Return tensor dimensions from outermost axis to innermost axis.
    fn shape(&self) -> &[usize];

    /// Copy tensor elements in row-major order into a float32 vector.
    fn to_vec(&self) -> Vec<f32>;
}

/// Host-memory float32 tensor with validated shape and finite values.
#[derive(Clone, Debug, PartialEq)]
pub struct HostTensor {
    shape: Vec<usize>,
    values: Vec<f32>,
}

impl HostTensor {
    /// Construct a host tensor when `values` matches `shape` and holds finite numbers.
    ///
    /// # Errors
    ///
    /// Returns [`VisualError::Invalid`] when the element count does not match `shape`
    /// or when any element is not finite.
    pub fn new(shape: Vec<usize>, values: Vec<f32>) -> Result<Self, VisualError> {
        let expected = shape
            .iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(d))
            .ok_or(VisualError::Invalid {
                field: "tensor shape overflow",
            })?;
        if values.len() != expected {
            return Err(VisualError::Invalid {
                field: "tensor buffer length",
            });
        }
        if values.iter().any(|v| !v.is_finite()) {
            return Err(VisualError::Invalid {
                field: "non-finite tensor value",
            });
        }
        Ok(Self { shape, values })
    }

    /// Return tensor dimensions from outermost axis to innermost axis.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Borrow the contiguous row-major float32 slice.
    pub fn as_slice(&self) -> &[f32] {
        &self.values
    }
}

impl GenericTensor for HostTensor {
    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn to_vec(&self) -> Vec<f32> {
        self.values.clone()
    }
}

/// Named tensor binding for one model input or model output.
#[derive(Clone, Debug, PartialEq)]
pub struct NamedTensor<T: GenericTensor = HostTensor> {
    /// Graph input or output name.
    pub name: String,
    /// Tensor value.
    pub tensor: T,
}

impl<T: GenericTensor> NamedTensor<T> {
    /// Bind `tensor` to graph port `name`.
    pub fn new(name: impl Into<String>, tensor: T) -> Self {
        Self {
            name: name.into(),
            tensor,
        }
    }
}

/// Synchronous model inference contract for native and embedded engines.
pub trait ModelInference {
    /// Tensor type accepted and returned by this engine.
    type Tensor: GenericTensor;
    /// Error type returned by this engine.
    type Error;

    /// Return the active runtime track.
    fn engine_kind(&self) -> EngineKind;

    /// Run one synchronous forward pass on named input tensors.
    ///
    /// # Errors
    ///
    /// Returns an error when an input name or shape is invalid or when execution fails.
    fn forward(
        &mut self,
        inputs: &[NamedTensor<Self::Tensor>],
    ) -> Result<Vec<NamedTensor<Self::Tensor>>, Self::Error>;
}

/// Asynchronous model inference contract for browser and GPU engines.
pub trait AsyncModelInference {
    /// Tensor type accepted and returned by this engine.
    type Tensor: GenericTensor;
    /// Error type returned by this engine.
    type Error;

    /// Return the active runtime track.
    fn engine_kind(&self) -> EngineKind;

    /// Run one asynchronous forward pass on named input tensors.
    ///
    /// # Errors
    ///
    /// Returns an error when an input name or shape is invalid or when execution fails.
    fn forward_async(
        &mut self,
        inputs: &[NamedTensor<Self::Tensor>],
    ) -> impl Future<Output = Result<Vec<NamedTensor<Self::Tensor>>, Self::Error>>;
}
