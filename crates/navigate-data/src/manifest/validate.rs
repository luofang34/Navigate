//! The ordering and value rules of a manifest.

use alloc::string::String;

use crate::{
    Accuracy, DataError, LayerKind, MAX_ZOOM, Sigma,
    manifest::{Layer, LayerZoom, Manifest, PACKAGE_FORMAT, PACKAGE_FORMAT_VERSION},
};

fn invalid(field: &'static str, detail: impl Into<String>) -> DataError {
    DataError::Invalid {
        field,
        detail: detail.into(),
    }
}

impl Manifest {
    /// Checks the format, the ordering, and the values.
    ///
    /// # Errors
    ///
    /// Refuses another format or version, an empty release identifier, a
    /// grid that is not the fixed grid, layers out of order or repeated, and
    /// any layer that [`Layer::validate`] refuses.
    pub fn validate(&self) -> Result<(), DataError> {
        if self.format != PACKAGE_FORMAT || self.version != PACKAGE_FORMAT_VERSION {
            return Err(invalid(
                "format",
                alloc::format!("{} version {}", self.format, self.version),
            ));
        }
        if self.release_id.is_empty() {
            return Err(invalid("release_id", "empty"));
        }
        if self.grid != crate::GridSpec::default() {
            return Err(invalid("grid", alloc::format!("{:?}", self.grid)));
        }
        if self.layers.windows(2).any(|w| match w {
            [a, b] => a.id >= b.id,
            _ => false,
        }) {
            return Err(invalid("layers", "not in strict identifier order"));
        }
        self.layers.iter().try_for_each(Layer::validate)
    }
}

impl Layer {
    /// Checks one layer.
    ///
    /// # Errors
    ///
    /// Refuses an empty identifier or source identity, zero tile pixels, an
    /// elevation layer without a vertical datum, an imagery layer with one,
    /// a sigma that is not finite and non-negative, and any zoom level that
    /// breaks the ordering rules.
    pub fn validate(&self) -> Result<(), DataError> {
        if self.id.is_empty() || self.source.id.is_empty() || self.source.revision.is_empty() {
            return Err(invalid("layer", alloc::format!("{:?}", self.id)));
        }
        if self.tile_pixels == 0 {
            return Err(invalid("tile_pixels", "zero"));
        }
        let datum_ok = match self.kind {
            LayerKind::Elevation => self.vertical_datum.is_some(),
            _ => self.vertical_datum.is_none(),
        };
        if !datum_ok {
            return Err(invalid(
                "vertical_datum",
                alloc::format!("layer {}", self.id),
            ));
        }
        check_accuracy(&self.accuracy)?;
        if self.zooms.windows(2).any(|w| match w {
            [a, b] => a.zoom >= b.zoom,
            _ => false,
        }) {
            return Err(invalid("zooms", "not in strict increasing order"));
        }
        self.zooms.iter().try_for_each(check_zoom)
    }
}

fn check_accuracy(accuracy: &Accuracy) -> Result<(), DataError> {
    for sigma in [
        accuracy.horizontal,
        accuracy.vertical,
        accuracy.age_growth_per_year,
    ] {
        if let Sigma::Known(value) = sigma
            && !(value.is_finite() && value >= 0.0)
        {
            return Err(invalid("accuracy", alloc::format!("{value}")));
        }
    }
    Ok(())
}

fn check_zoom(zoom: &LayerZoom) -> Result<(), DataError> {
    if zoom.zoom > MAX_ZOOM {
        return Err(invalid("zoom", alloc::format!("{}", zoom.zoom)));
    }
    let side = 1_u64 << zoom.zoom;
    let inside = |x: u32, y: u32| u64::from(x) < side && u64::from(y) < side;
    let tiles_ordered = zoom.tiles.windows(2).all(|w| match w {
        [a, b] => (a.y, a.x) < (b.y, b.x),
        _ => true,
    });
    let gaps_ordered = zoom.gaps.windows(2).all(|w| match w {
        [[ax, ay], [bx, by]] => (ay, ax) < (by, bx),
        _ => true,
    });
    if !tiles_ordered || !gaps_ordered {
        return Err(invalid(
            "tiles",
            alloc::format!("zoom {} is not in strict row, column order", zoom.zoom),
        ));
    }
    if zoom.tiles.iter().any(|t| !inside(t.x, t.y)) || zoom.gaps.iter().any(|&[x, y]| !inside(x, y))
    {
        return Err(invalid(
            "tiles",
            alloc::format!("address outside zoom {}", zoom.zoom),
        ));
    }
    let both = zoom.gaps.iter().any(|&[x, y]| {
        zoom.tiles
            .binary_search_by_key(&(y, x), |t| (t.y, t.x))
            .is_ok()
    });
    if both {
        return Err(invalid("gaps", "an address is both a tile and a gap"));
    }
    Ok(())
}
