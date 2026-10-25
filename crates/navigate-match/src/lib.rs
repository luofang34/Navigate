//! Learned image matchers in pure Rust on burn.
//!
//! One implementation runs on WebGPU in a browser, on native GPUs through wgpu
//! (Vulkan, Metal) or CUDA, and on the CPU. The host chooses the runtime with the
//! burn device that it gives to the model; Cargo features compile runtimes in.
//!
//! [`Loftr`] is LoFTR-DS with the outdoor dual-softmax weights. [`LoftrMatcher`]
//! puts it behind the Navigate matcher port with the input transform of the ONNX
//! adapter. [`loftr_from_checkpoint`] converts the training checkpoint into the
//! weight file that [`Loftr::load`] reads.
mod convert;
mod error;
mod input;
mod loftr;
mod matcher;
mod weights;

pub use convert::loftr_from_checkpoint;
pub use error::MatchError;
pub use loftr::{Correspondence, Loftr};
pub use matcher::LoftrMatcher;
pub use weights::{WeightTensor, Weights};
