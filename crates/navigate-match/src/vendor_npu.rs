//! Embedded vendor neural processing unit adapter seam.
//!
//! Embedded targets with a vendor compiler (such as Rockchip RKNN or HailoRT)
//! implement [`VendorNpuBackend`] in a board crate and pass it to [`VendorNpuSession`].
//! This crate keeps zero `unsafe` blocks while exposing [`ModelInference`].
use crate::{
    MatchError,
    traits::{EngineKind, HostTensor, ModelInference, NamedTensor},
};

/// Hardware neural processing unit driver implemented by a board support crate.
pub trait VendorNpuBackend: Send {
    /// Return the vendor runtime identifier, such as `"rknn"` or `"hailort"`.
    fn vendor_name(&self) -> &'static str;

    /// Execute one compiled graph step on validated host tensors.
    ///
    /// # Errors
    ///
    /// Returns [`MatchError::Device`] when the hardware driver rejects the call.
    fn execute(
        &mut self,
        inputs: &[NamedTensor<HostTensor>],
    ) -> Result<Vec<NamedTensor<HostTensor>>, MatchError>;
}

/// [`ModelInference`] wrapper around a [`VendorNpuBackend`].
pub struct VendorNpuSession<B: VendorNpuBackend> {
    backend: B,
}

impl<B: VendorNpuBackend> VendorNpuSession<B> {
    /// Wrap a board neural processing unit backend.
    pub fn new(backend: B) -> Self {
        Self { backend }
    }

    /// Return the vendor runtime identifier.
    pub fn vendor_name(&self) -> &'static str {
        self.backend.vendor_name()
    }

    /// Borrow the underlying backend.
    pub fn backend(&self) -> &B {
        &self.backend
    }
}

impl<B: VendorNpuBackend> ModelInference for VendorNpuSession<B> {
    type Tensor = HostTensor;
    type Error = MatchError;

    fn engine_kind(&self) -> EngineKind {
        EngineKind::VendorNpu
    }

    fn forward(
        &mut self,
        inputs: &[NamedTensor<HostTensor>],
    ) -> Result<Vec<NamedTensor<HostTensor>>, MatchError> {
        let outputs = self.backend.execute(inputs)?;
        for output in &outputs {
            if output.tensor.as_slice().iter().any(|v| !v.is_finite()) {
                return Err(MatchError::Output("vendor NPU returned non-finite output"));
            }
        }
        Ok(outputs)
    }
}
