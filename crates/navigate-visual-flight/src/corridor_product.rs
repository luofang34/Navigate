//! The `NVKTILE1` corridor bank as a derived product of a navigation data
//! package (ADR-0022, representation L).

use crate::{corridor_build::CorridorBuildSpec, error::FlightVisualError};
use navigate_data::product::{ProductKind, ProductWriter, Recipe};
use navigate_data::resource::altitude_band;
use navigate_data::{Sha256Digest, WritePort};
use navigate_match::{CORRIDOR_MAGIC, CorridorLandmarkBank};
use std::collections::BTreeMap;

/// What the recipe of a corridor product records beyond the build spec.
#[derive(Clone, Copy, Debug)]
pub struct CorridorProductRecipe {
    /// SHA-256 of the `XFeat` weights file.
    pub weights_sha256: Sha256Digest,
    /// `[lat, lon]` in degrees of the east, north, up frame of the bank
    /// tiles: the centre of the surface.
    pub tile_frame_origin: [f64; 2],
}

/// The complete recipe of a bank built with `spec`.
#[must_use]
pub fn corridor_recipe(
    spec: &CorridorBuildSpec,
    bank: &CorridorLandmarkBank,
    extra: &CorridorProductRecipe,
) -> Recipe {
    let hex = |bytes: &[u8]| bytes.iter().map(|b| format!("{b:02x}")).collect::<String>();
    let priors: Vec<String> = spec
        .priors
        .iter()
        .map(|p| format!("{},{}", p.lat_deg, p.lon_deg))
        .collect();
    let [lat, lon] = extra.tile_frame_origin;
    let parameters = BTreeMap::from([
        ("tile_size_m".into(), spec.tile_size_m.to_string()),
        (
            "corridor_radius_m".into(),
            spec.corridor_radius_m.to_string(),
        ),
        ("nominal_agl_m".into(), spec.nominal_agl_m.to_string()),
        (
            "max_landmarks_per_tile".into(),
            spec.max_landmarks_per_tile.to_string(),
        ),
        ("adaptive_stride".into(), spec.adaptive_stride.to_string()),
        ("include_obliques".into(), spec.include_obliques.to_string()),
        ("xfeat_runtime".into(), format!("{:?}", spec.xfeat_runtime)),
        ("priors".into(), priors.join(";")),
        (
            "render".into(),
            "800x600, fx=fy=600, cx=399.5, cy=299.5".into(),
        ),
        ("encoder_contract".into(), hex(&bank.encoder_contract)),
        ("kashin_seed".into(), bank.seed.to_string()),
        (
            "tile_frame".into(),
            format!("east, north, up at {lat}, {lon}"),
        ),
        (
            "heights".into(),
            "provider-native, not converted to the ellipsoid".into(),
        ),
        ("encoding".into(), "NVKTILE1".into()),
    ]);
    Recipe {
        algorithm: "navigate-visual-flight corridor_build XFeat Kashin".into(),
        version: env!("CARGO_PKG_VERSION").into(),
        weights: Some(extra.weights_sha256),
        precision: "f32".into(),
        parameters,
    }
}

/// Writes a monolithic `NVKTILE1` bank as a product of the core package
/// `core`. The bank must name `core` as its source map.
///
/// # Errors
///
/// Refuses bytes that are not a monolithic bank, a bank made from another
/// map, and a failed write.
pub async fn write_corridor_product<W: WritePort + ?Sized>(
    store: &mut W,
    core: Sha256Digest,
    spec: &CorridorBuildSpec,
    extra: &CorridorProductRecipe,
    bank_bytes: &[u8],
) -> Result<Sha256Digest, FlightVisualError> {
    if !bank_bytes.starts_with(CORRIDOR_MAGIC) {
        return Err(FlightVisualError::Invalid(
            "a corridor product holds one monolithic NVKTILE1 bank".into(),
        ));
    }
    let bank = CorridorLandmarkBank::from_bytes(bank_bytes)?;
    if bank.catalog_sha256 != core.0 {
        return Err(FlightVisualError::Invalid(format!(
            "the bank was made from another map than package {core}"
        )));
    }
    let mut writer = ProductWriter::new(
        ProductKind::CorridorLandmarks,
        core,
        corridor_recipe(spec, &bank, extra),
    )
    .with_altitude_band(altitude_band(spec.nominal_agl_m));
    writer.add_entry(store, "bank", None, bank_bytes).await?;
    Ok(writer.seal(store).await?)
}
