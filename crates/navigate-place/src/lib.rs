//! Place index and search plan for localization of a photo or a vehicle camera.
//!
//! A place index holds one descriptor for each map crop of a region, at one or more
//! crop sizes. A search plan selects the crop sizes, the view directions and the
//! index cells that a query must use, from the search region and the optional priors
//! (heading, height above ground, camera angle). A ranking compares the query
//! descriptors with the index records of the selected cells. The caller encodes the
//! query and verifies the ranked places; this crate does no I/O and holds no model.
mod canvas;
mod error;
mod format;
mod hilbert;
mod plan;
mod pooling;
mod rank;
mod region;

pub use canvas::{CANVAS_PX, Canvas, GRID, Tokens, cell_weights};
pub use error::PlaceError;
pub use format::{
    IndexManifest, ModelEntry, QueryHead, Record, ScaleEntry, ShardEntry, decode_shard,
    encode_shard,
};
pub use plan::{
    Fallback, Footprint, HeadingPrior, HeightPrior, PitchPrior, ScalePlan, SearchPlan,
    SearchRequest, plan,
};
pub use pooling::{Descriptor, describe};
pub use rank::{Ranked, rank};
pub use region::{CELL_ZOOM, Cell, Region, cells};
