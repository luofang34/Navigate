use crate::{BenchError, package::digest, read_blocking};
use nalgebra::{Quaternion, UnitQuaternion};
use navigate_visual::{
    MapRevision,
    place_retrieval::{ReferenceCatalog, ReferenceId},
};
use navigate_visual_onnx::CampIndex;
use serde::{Deserialize, de::DeserializeOwned};
use std::path::{Path, PathBuf};

#[derive(Deserialize)]
struct IndexFile {
    map_release: String,
    map_manifest_sha256: String,
    catalog_manifest_sha256: String,
    model_sha256: String,
    descriptors: PathBuf,
    descriptors_sha256: String,
    ids: Vec<u64>,
}
#[derive(Deserialize)]
pub(super) struct Catalog {
    pub release: String,
    pub gallery: Vec<Entry>,
}
#[derive(Deserialize)]
pub(super) struct Entry {
    pub id: u64,
    pub center_enu_m: [f64; 2],
    pub width_m: f64,
}
pub(super) fn load_blocking(
    path: &Path,
    catalog_path: &Path,
    revision: &MapRevision,
) -> Result<(CampIndex, Catalog), BenchError> {
    let input: IndexFile = json_blocking(path)?;
    let bytes = read_blocking(catalog_path)?;
    if input.map_release != revision.release_id
        || input.map_manifest_sha256 != revision.manifest_sha256
        || digest(&bytes) != input.catalog_manifest_sha256
    {
        return Err(super::invalid(
            "reference index map or catalog identity differs",
        ));
    }
    let catalog: Catalog = serde_json::from_slice(&bytes).map_err(|source| BenchError::Json {
        path: catalog_path.to_owned(),
        source,
    })?;
    if catalog.release != revision.release_id
        || !catalog
            .gallery
            .iter()
            .map(|e| e.id)
            .eq(input.ids.iter().copied())
    {
        return Err(super::invalid(
            "reference catalog rows or map release differ",
        ));
    }
    let descriptor_path = path
        .parent()
        .unwrap_or(Path::new("."))
        .join(input.descriptors);
    let bytes = read_blocking(&descriptor_path)?;
    if digest(&bytes) != input.descriptors_sha256 {
        return Err(super::invalid("reference descriptor checksum differs"));
    }
    let (values, tail) = bytes.as_chunks::<4>();
    if !tail.is_empty() {
        return Err(super::invalid("partial float32 reference descriptor"));
    }
    let index = CampIndex::new(
        ReferenceCatalog {
            map: revision.clone(),
            manifest_sha256: input.catalog_manifest_sha256,
        },
        input.model_sha256,
        input.ids.into_iter().map(ReferenceId).collect(),
        values.iter().map(|v| f32::from_le_bytes(*v)).collect(),
    )?;
    Ok((index, catalog))
}
pub(super) fn json_blocking<T: DeserializeOwned>(path: &Path) -> Result<T, BenchError> {
    serde_json::from_slice(&read_blocking(path)?).map_err(|source| BenchError::Json {
        path: path.to_owned(),
        source,
    })
}
pub(super) fn orientations_blocking(
    path: Option<&Path>,
) -> Result<Vec<UnitQuaternion<f64>>, BenchError> {
    let Some(path) = path else {
        return Ok((0..8)
            .map(|i| {
                UnitQuaternion::from_axis_angle(
                    &nalgebra::Vector3::z_axis(),
                    f64::from(i) * std::f64::consts::FRAC_PI_4,
                )
            })
            .collect());
    };
    let values: Vec<[f64; 4]> = json_blocking(path)?;
    if values.is_empty() || values.len() > 8 {
        return Err(super::invalid("supply one to eight camera orientations"));
    }
    values
        .into_iter()
        .map(|[x, y, z, w]| {
            let q = Quaternion::new(w, x, y, z);
            if !q.coords.iter().all(|v| v.is_finite()) || (q.norm() - 1.0).abs() > 1e-6 {
                return Err(super::invalid(
                    "camera orientation is not a finite unit quaternion",
                ));
            }
            Ok(UnitQuaternion::new_normalize(q))
        })
        .collect()
}

#[cfg(test)]
mod tests;
