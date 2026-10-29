//! Pure-Rust SegFormer-B0 inference, occlusion mask policies, propeller detection,
//! and 4-class semantic fine-tuning on `burn`.

pub mod airframe;
pub mod config;
pub mod error;
#[cfg(feature = "optim")]
pub mod finetune;
pub mod labels;
pub mod masks;
pub mod model;
pub mod morphology;
pub mod npy;
pub mod video;

pub use airframe::{border_support_mask, support_mask};
pub use config::{
    AIRFRAME, CLASSES, ColorMode, GROUND, IGNORE_INDEX, PROPELLER, PreprocessorConfig, SKY,
    SegformerConfig,
};
pub use error::SegmentError;
#[cfg(feature = "optim")]
pub use finetune::{DatasetSample, FinetuneReport, Score, TrainPolicy, evaluate, finetune};
pub use labels::{
    ObjectClass, compose_labels, contact_sheet, drop_specks, parse_object, split_name,
};
pub use masks::{MaskPolicy, MaskSummary, OBSTRUCTION_LABELS, SURFACE_LABELS, surface_mask};
pub use model::{Segformer, SegmentationPrediction};
pub use npy::{NpyArray, NpyDType, read_npy, read_npz, write_npy, write_npz};
pub use video::{FrameMaskResult, PropellerDetector, class_masks, frame_masks};
