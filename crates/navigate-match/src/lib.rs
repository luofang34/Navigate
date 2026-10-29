//! Learned image matchers and dual-engine inference traits in pure Rust on burn.
//!
//! One implementation runs on WebGPU or CPU in a browser, on native GPUs through wgpu
//! (Vulkan, Metal) or CUDA, and on native CPUs. [`ModelInference`] and
//! [`AsyncModelInference`] define the unified tensor session contract across browser,
//! native, and embedded neural processing unit backends.
mod convert;
mod error;
mod input;
pub mod kashin;
mod kernels;
mod lighterglue;
mod loftr;
mod matcher;
#[cfg(not(target_arch = "wasm32"))]
mod native_engine;
mod staged;
mod traits;
mod vendor_npu;
#[cfg(target_arch = "wasm32")]
mod web_engine;
mod weights;
mod xfeat;

pub use convert::loftr_from_checkpoint;
pub use error::MatchError;
pub use kashin::{
    DEFAULT_KASHIN_SEED, HadamardProjector, KashinDescriptor, PrecomputedMatch,
    PrecomputedReferenceBank, PrecomputedViewEntry, ViewSketch,
};
pub use lighterglue::LighterGlue;
pub use loftr::{Correspondence, Loftr};
pub use matcher::LoftrMatcher;
#[cfg(not(target_arch = "wasm32"))]
pub use native_engine::{NativeMlEngine, XFeatDenseSession};
pub use staged::{BurnXFeatFeatures, XFeatLighterGlueMatcher};
pub use traits::{
    AsyncModelInference, EngineKind, GenericTensor, HostTensor, ModelInference, NamedTensor,
};
pub use vendor_npu::{VendorNpuBackend, VendorNpuSession};
#[cfg(target_arch = "wasm32")]
pub use web_engine::{WebMlEngine, WebNnAvailability, probe_webnn};
pub use weights::{WeightTensor, Weights};
pub use xfeat::{DenseMaps, SparseFeatures, XFEAT_INPUT_HEIGHT, XFEAT_INPUT_WIDTH, XFeat};
