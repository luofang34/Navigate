//! The rules of a manifest.

use alloc::string::String;

use crate::{
    Body, DataError, EllipsoidId, Encoding, GeoidModel, GridScheme, HorizontalDatum, LayerKind,
    MAX_ZOOM, SourceKey, VerticalDatum,
    manifest::{Layer, Manifest, PACKAGE_FORMAT, PACKAGE_FORMAT_VERSION},
};

fn invalid(field: &'static str, detail: impl Into<String>) -> DataError {
    DataError::Invalid {
        field,
        detail: detail.into(),
    }
}

const fn unsupported(field: &'static str) -> DataError {
    DataError::Unsupported { field }
}

impl Manifest {
    /// Checks the format, the version, the vocabulary, and the order.
    ///
    /// # Errors
    ///
    /// Refuses another format, another version ([`DataError::Version`]), an
    /// unsupported body, ellipsoid, or grid scheme
    /// ([`DataError::Unsupported`]), a manifest without layers, layers out
    /// of order or repeated, and any layer that [`Layer::validate`]
    /// refuses.
    pub fn validate(&self) -> Result<(), DataError> {
        if self.format != PACKAGE_FORMAT {
            return Err(invalid("format", self.format.clone()));
        }
        if self.version != PACKAGE_FORMAT_VERSION {
            return Err(DataError::Version {
                document: "package manifest",
                found: self.version,
                supported: PACKAGE_FORMAT_VERSION,
            });
        }
        if self.body != Body::Earth {
            return Err(unsupported("body"));
        }
        if self.ellipsoid != EllipsoidId::Wgs84 {
            return Err(unsupported("ellipsoid"));
        }
        if self.grid != GridScheme::WebMercatorXyz {
            return Err(unsupported("grid"));
        }
        if self.layers.is_empty() {
            return Err(invalid("layers", "a package has at least one layer"));
        }
        self.layers.iter().try_for_each(Layer::validate)?;
        if self
            .layers
            .windows(2)
            .any(|w| matches!(w, [a, b] if a.key >= b.key))
        {
            return Err(invalid("layers", "not in strict key order"));
        }
        Ok(())
    }
}

impl Layer {
    /// Checks one layer.
    ///
    /// # Errors
    ///
    /// Refuses an unsupported kind, source, encoding, horizontal datum,
    /// vertical datum, or geoid model; an elevation layer without a
    /// vertical datum and an imagery layer with one; zero tile pixels; a
    /// correlation zoom past the scheme; and an index whose counts do not
    /// match its size.
    pub fn validate(&self) -> Result<(), DataError> {
        if self.key.kind == LayerKind::Unsupported {
            return Err(unsupported("layer kind"));
        }
        if self.key.source == SourceKey::Unsupported {
            return Err(unsupported("source"));
        }
        if self.encoding == Encoding::Unsupported {
            return Err(unsupported("encoding"));
        }
        if self.horizontal_datum == HorizontalDatum::Unsupported {
            return Err(unsupported("horizontal datum"));
        }
        match (self.key.kind, self.vertical_datum) {
            (LayerKind::Elevation, None) => {
                return Err(invalid("vertical_datum", "elevation needs one"));
            }
            (LayerKind::Imagery, Some(_)) => {
                return Err(invalid("vertical_datum", "imagery has none"));
            }
            (_, Some(VerticalDatum::Unsupported)) => return Err(unsupported("vertical datum")),
            (
                _,
                Some(VerticalDatum::Orthometric {
                    geoid: GeoidModel::Unsupported,
                }),
            ) => {
                return Err(unsupported("geoid model"));
            }
            _ => {}
        }
        if self.tile_pixels == 0 || self.correlation_zoom > MAX_ZOOM {
            return Err(invalid(
                "layer",
                alloc::format!(
                    "{} px, correlation zoom {}",
                    self.tile_pixels,
                    self.correlation_zoom
                ),
            ));
        }
        let expected = (crate::index::INDEX_MAGIC.len() as u64).saturating_add(
            self.index
                .records()
                .saturating_mul(crate::index::RECORD_BYTES as u64),
        );
        if self.index.bytes != expected {
            return Err(invalid(
                "index",
                alloc::format!(
                    "{} bytes for {} records",
                    self.index.bytes,
                    self.index.records()
                ),
            ));
        }
        Ok(())
    }
}
