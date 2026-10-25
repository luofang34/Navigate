//! Camera evidence, recording contracts, and Navigate session processing.

mod error;
mod pipeline;
mod wire;

pub use error::Error;
pub use pipeline::{Processor, RevisitInput, VisualObservation};
pub use wire::{
    Camera, Configuration, Observation, OutputSupport, Pose, Reference, ResultFrame, Revisit,
    RevisitRequest,
};
