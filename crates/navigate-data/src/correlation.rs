//! Correlated map error.

use serde::{Deserialize, Serialize};

use crate::{DataError, GridCell, Layer, Revision, SourceKey};

/// The key of one correlated-error cell: fixes from the same source,
/// revision, and cell share the registration error of the map, whatever
/// package they came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct CorrelationKey {
    /// The source.
    pub source: SourceKey,
    /// Its revision.
    pub revision: Revision,
    /// The cell at the correlation zoom of the layer.
    pub cell: GridCell,
}

impl Layer {
    /// The correlation key of a point, `[lat, lon]` in degrees.
    ///
    /// # Errors
    ///
    /// Refuses a point outside the grid scheme.
    pub fn correlation_key(&self, point: [f64; 2]) -> Result<CorrelationKey, DataError> {
        Ok(CorrelationKey {
            source: self.key.source,
            revision: self.revision,
            cell: GridCell::containing(point, self.correlation_zoom)?,
        })
    }
}
