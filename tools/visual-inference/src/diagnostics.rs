//! Tensor transfers reported by the selected inference adapter.
use serde::Serialize;

/// Cumulative rustnn Core ML counters, excluding driver-internal work.
#[derive(Clone, Debug, Serialize)]
pub struct TensorIoStatistics {
    /// Permitted Core ML compute units, not measured operator placement.
    pub compute_units: String,
    /// Native tensor allocation count.
    pub native_allocations: u64,
    /// Bytes read into application output buffers.
    pub host_read_bytes: u64,
    /// Bytes written from application input buffers.
    pub host_write_bytes: u64,
    /// Input bytes copied inside the rustnn dispatch adapter.
    pub dispatch_input_copy_bytes: u64,
    /// Output bytes copied inside the rustnn dispatch adapter.
    pub dispatch_output_copy_bytes: u64,
    /// Core ML outputs that reused a requested persistent backing.
    pub output_backings_accepted: u64,
}

#[cfg(all(feature = "rustnn-coreml", target_os = "macos"))]
impl From<rustnn::mlcontext::CoremlTensorStatistics> for TensorIoStatistics {
    fn from(stats: rustnn::mlcontext::CoremlTensorStatistics) -> Self {
        Self {
            compute_units: stats.last_compute_units.into(),
            native_allocations: stats.native_allocations,
            host_read_bytes: stats.host_read_bytes,
            host_write_bytes: stats.host_write_bytes,
            dispatch_input_copy_bytes: stats.input_copy_bytes,
            dispatch_output_copy_bytes: stats.output_copy_bytes,
            output_backings_accepted: stats.output_backings_accepted,
        }
    }
}
