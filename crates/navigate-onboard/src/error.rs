//! Errors of the onboard boundary.

use navigate_camera::SpectralBand;
use navigate_contract::SourceId;
use navigate_data::{DataError, LayerKind, Sha256Digest};
use navigate_fusion::RejectionReason;

/// A refusal of the onboard boundary.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum OnboardError {
    /// The sensor suite or a configuration value is not usable.
    #[error("invalid {field}: {detail}")]
    Config {
        /// The input.
        field: &'static str,
        /// Why it is not usable.
        detail: String,
    },
    /// An input names a sensor that the suite does not declare.
    #[error("sensor {0:?} is not in the declared suite")]
    UnknownSensor(SourceId),
    /// An input does not match the kind of its sensor.
    #[error("sensor {sensor:?} is a {kind}, not a source of {data}")]
    WrongData {
        /// The sensor.
        sensor: SourceId,
        /// Its declared kind.
        kind: &'static str,
        /// The data that came.
        data: &'static str,
    },
    /// The filter refused a step that the navigation cannot continue
    /// without: the alignment or an IMU increment.
    #[error("the filter refused the {stage}: {reason:?}")]
    Refused {
        /// What was refused.
        stage: &'static str,
        /// Navigate's reason.
        reason: RejectionReason,
    },
    /// The star tracker failed outside identification.
    #[error("star tracker: {0}")]
    StarTrack(String),
    /// A terrain camera in a spectral band that the terrain means cannot
    /// match against the visible imagery of a package.
    #[error("sensor {sensor:?} is a {band:?} camera; the terrain means takes a visible camera")]
    UnsupportedBand {
        /// The sensor.
        sensor: SourceId,
        /// Its declared band.
        band: SpectralBand,
    },
    /// A frame whose size is not the size of the declared intrinsics.
    #[error(
        "sensor {sensor:?} declares {declared:?} pixels; the frame states {frame:?} and holds {bytes} bytes"
    )]
    FrameSize {
        /// The sensor.
        sensor: SourceId,
        /// Width and height of the declared intrinsics.
        declared: (u32, u32),
        /// Width and height that the frame states.
        frame: (u32, u32),
        /// Length of the pixel data.
        bytes: usize,
    },
    /// A terrain frame came and the navigator holds no terrain map.
    #[error("sensor {sensor:?} is a terrain camera and no terrain map is loaded")]
    MapNotLoaded {
        /// The sensor.
        sensor: SourceId,
    },
    /// The terrain map was refused at load.
    #[error("terrain map {package} refused: {refusal}")]
    MapRefused {
        /// The package identity that the host expected.
        package: Sha256Digest,
        /// Why the map was refused.
        #[source]
        refusal: MapRefusal,
    },
    /// The store failed, or gave changed data, while the navigator read the
    /// terrain map for a frame.
    #[error("terrain map read for sensor {sensor:?}: {detail}")]
    MapRead {
        /// The sensor whose frame needed the data.
        sensor: SourceId,
        /// What failed.
        detail: String,
    },
}

/// Why a terrain map was refused at load.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum MapRefusal {
    /// The suite declares no terrain camera, so no means reads a map.
    #[error("the suite declares no terrain camera")]
    NoTerrainCamera,
    /// The package is missing, is not the expected one, is of another
    /// format version, or holds a value that the data contract refuses.
    #[error("the package: {0}")]
    Package(#[source] DataError),
    /// The package holds no layer of a kind that the means needs.
    #[error("the package holds no {0:?} layer")]
    MissingLayer(LayerKind),
    /// A needed layer declares no tile.
    #[error("the {0:?} layer holds no tile")]
    EmptyLayer(LayerKind),
    /// A layer is in a form that the reference renderer cannot read.
    #[error("the {layer:?} layer: {detail}")]
    UnsupportedLayer {
        /// The layer kind.
        layer: LayerKind,
        /// What the renderer cannot read.
        detail: String,
    },
    /// A layer states no accuracy and the host declared none. Unknown
    /// error is not zero error, so the means has no covariance for a fix.
    #[error("the {0:?} layer states no accuracy and the request assumes none")]
    UnknownAccuracy(LayerKind),
    /// The product list is missing, changed, or of another package.
    #[error("the product list: {0}")]
    ProductList(#[source] DataError),
    /// A value of the request is not usable.
    #[error("invalid {field}")]
    Request {
        /// The value.
        field: &'static str,
    },
}
