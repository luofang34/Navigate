//! The `NVKTILE1` corridor bank as a derived product chunk (ADR-0022,
//! representation L).
//!
//! The bank tiles are in an east, north, up frame at the centre of the
//! surface, not on the fixed grid, so one bank is one chunk for its whole
//! area (`cell` is `null`) and is not reusable cell for cell.

use crate::{corridor_build::CorridorBuildSpec, error::FlightVisualError};
use navigate_data::product::{
    ChunkBinding, Precision, ProductKind, ProductWriter, Recipe, inputs_digest,
};
use navigate_data::resource::altitude_band;
use navigate_data::{Sha256Digest, WritePort};
use navigate_match::{CORRIDOR_MAGIC, CorridorLandmarkBank};

/// What the recipe of a corridor product records beyond the build spec.
#[derive(Clone, Copy, Debug)]
pub struct CorridorProductRecipe {
    /// SHA-256 of the `XFeat` weights file.
    pub weights_sha256: Sha256Digest,
    /// `[lat, lon]` in degrees of the east, north, up frame of the bank
    /// tiles: the centre of the surface.
    pub tile_frame_origin: [f64; 2],
}

fn round(v: f64, scale: f64) -> i64 {
    (v * scale).round() as i64
}

/// The complete recipe of a bank built with `spec`, in integer units.
#[must_use]
pub fn corridor_recipe(
    spec: &CorridorBuildSpec,
    bank: &CorridorLandmarkBank,
    extra: &CorridorProductRecipe,
) -> Recipe {
    let [lat, lon] = extra.tile_frame_origin;
    let mut recipe = Recipe::new(
        "navigate-visual-flight/corridor-build-xfeat-kashin",
        env!("CARGO_PKG_VERSION"),
        Some(extra.weights_sha256),
        Precision::F32,
    )
    .with("tile_size_mm", round(spec.tile_size_m, 1e3))
    .with("corridor_radius_mm", round(spec.corridor_radius_m, 1e3))
    .with("nominal_agl_mm", round(spec.nominal_agl_m, 1e3))
    .with("max_landmarks_per_tile", spec.max_landmarks_per_tile as i64)
    .with("adaptive_stride_px", i64::from(spec.adaptive_stride))
    .with("include_obliques", i64::from(spec.include_obliques))
    .with("render_width_px", 800)
    .with("render_height_px", 600)
    .with("render_focal_upx", 600_000_000)
    .with("kashin_seed", bank.seed as i64)
    .with("tile_frame_lat_ndeg", round(lat, 1e9))
    .with("tile_frame_lon_ndeg", round(lon, 1e9))
    .with("heights_provider_native", 1)
    .with("encoding_nvktile1", 1);
    let (words, _) = bank.encoder_contract.as_chunks::<8>();
    for (i, word) in words.iter().enumerate() {
        recipe = recipe.with(&format!("encoder_contract_{i}"), i64::from_le_bytes(*word));
    }
    for (i, prior) in spec.priors.iter().enumerate() {
        recipe = recipe
            .with(&format!("prior_{i:04}_lat_ndeg"), round(prior.lat_deg, 1e9))
            .with(&format!("prior_{i:04}_lon_ndeg"), round(prior.lon_deg, 1e9));
    }
    recipe
}

/// Writes a monolithic `NVKTILE1` bank as a `corridor_landmarks` chunk.
/// `inputs` are the tile blobs of the surface that the bank was built on;
/// the bank must name their [`inputs_digest`] as its source map. Returns
/// the chunk digest.
///
/// # Errors
///
/// Refuses bytes that are not a monolithic bank, a bank made from other
/// inputs, and a failed write.
pub async fn write_corridor_product<W: WritePort + ?Sized>(
    store: &mut W,
    inputs: &[Sha256Digest],
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
    if bank.catalog_sha256 != inputs_digest(inputs).0 {
        return Err(FlightVisualError::Invalid(
            "the bank was made from other inputs".into(),
        ));
    }
    let binding = ChunkBinding {
        cell: None,
        altitude_band: Some(altitude_band(spec.nominal_agl_m)),
        inputs: inputs.to_vec(),
        route: None,
        envelope: None,
        calibration: None,
    };
    let mut writer = ProductWriter::new(
        ProductKind::CorridorLandmarks,
        corridor_recipe(spec, &bank, extra),
        binding,
    );
    writer.add_entry(store, 0, None, bank_bytes).await?;
    Ok(writer.seal(store).await?)
}
