//! Query-independent map feature packages use the search window rules.
mod packing;
use crate::{
    Error, assets,
    config::{Backend, Config},
    surface::job::Surface,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{collections::BTreeSet, path::Path, time::Instant};

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Entry {
    pub key: [u8; 32],
    pub sha256: [u8; 32],
    pub bytes: usize,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Catalog {
    pub version: u32,
    pub source: [u8; 32],
    pub contract: [u8; 32],
    pub include_fine: bool,
    pub entries: Vec<Entry>,
}
pub(in crate::surface) fn source_namespace(surface: &Surface) -> [u8; 32] {
    let identity = json!({"image":surface.image.sha256,"heights":surface.heights.sha256,
        "origin":surface.origin,"column":surface.column,"row":surface.row,
        "map_release":surface.map_release,"map_sha256":surface.map_sha256});
    Sha256::digest(identity.to_string().as_bytes()).into()
}
fn windows(size: (u32, u32), fine: bool) -> Result<Vec<Option<[u32; 4]>>, Error> {
    let side = super::selection::SIDE;
    if size.0 < side || size.1 < side || size.0 > 8192 || size.1 > 8192 {
        return Err(Error::Input(
            "feature catalog map dimensions must be 2048..=8192".into(),
        ));
    }
    let axis = |length: u32| {
        let limit = length - side;
        let mut result: BTreeSet<_> = (0..=limit).step_by(1024).collect();
        result.insert(limit);
        result
    };
    let mut parents = BTreeSet::new();
    for x in axis(size.0) {
        for y in axis(size.1) {
            parents.extend(super::selection::neighbors([x, y, side, side], size));
        }
    }
    let mut all = parents.clone();
    if fine {
        for parent in parents {
            all.extend(super::selection::children(parent));
        }
    }
    if all.len().saturating_add(1).saturating_mul(4) > 4096 {
        return Err(Error::Input(
            "feature catalog exceeds 4096 packet bound".into(),
        ));
    }
    Ok(std::iter::once(None)
        .chain(all.into_iter().map(Some))
        .collect())
}
pub(super) fn hex(value: &[u8; 32]) -> String {
    value.iter().map(|b| format!("{b:02x}")).collect()
}
pub(super) fn prepare_blocking(
    config: &Config,
    root: &Path,
    input: &Path,
    output: &Path,
    fine: bool,
) -> Result<(), Error> {
    let started = Instant::now();
    let surface: Surface = assets::read_json_blocking(input)?;
    if surface.window.is_some() {
        return Err(Error::Input(
            "feature catalog requires the full source image".into(),
        ));
    }
    let image = assets::image_verified_blocking(
        &input
            .parent()
            .unwrap_or(Path::new("."))
            .join(&surface.image.path),
        config.limits.image_pixels,
        Some(&surface.image.sha256),
    )?;
    let windows = windows(image.dimensions(), fine)?;
    let Backend::Features(mut backend) = config.load_blocking(root)? else {
        return Err(Error::Input(
            "feature catalog requires an independent encoder".into(),
        ));
    };
    let source = source_namespace(&surface);
    let mut catalog = Catalog {
        version: 1,
        source,
        contract: backend.contract(),
        include_fine: fine,
        entries: Vec::new(),
    };
    std::fs::create_dir(output).map_err(|source| Error::Io {
        operation: "create feature catalog",
        path: output.into(),
        source,
    })?;
    let setup_ms = started.elapsed().as_secs_f64() * 1000.;
    let mut rows = Vec::<Value>::new();
    let mut keys = BTreeSet::new();
    for window in &windows {
        let cropped =
            window.map(|[x, y, w, h]| image::imageops::crop_imm(&image, x, y, w, h).to_image());
        let view = cropped.as_ref().unwrap_or(&image);
        let base = super::cache::fingerprint(source, backend.contract(), view, *window);
        let base_bytes = image.as_raw().len() + cropped.as_ref().map_or(0, |i| i.as_raw().len());
        for rotation in 0..4 {
            let key = super::cache::rotation_key(base, rotation);
            if !keys.insert(key) {
                continue;
            }
            let request = packing::Request {
                image: view,
                rotation,
                key,
                base_bytes,
                byte_limit: config.limits.reference_bytes,
                feature_limit: config.keypoints,
                feature_budget: config.limits.features_per_image,
            };
            let (entry, mut row) = packing::write_blocking(&mut backend, request, output)?;
            row["window"] = json!(window);
            rows.push(row);
            catalog.entries.push(entry);
        }
    }
    catalog.entries.sort_by_key(|e| e.key);
    assets::write_json_blocking(&output.join("catalog.json"), &catalog)?;
    assets::write_json_blocking(
        &output.join("preparation.json"),
        &json!({"source":surface,"dimensions":image.dimensions(),"include_fine":fine,"windows":windows.len(),"packets":catalog.entries.len(),"packet_bytes":catalog.entries.iter().map(|e|e.bytes).sum::<usize>(),"setup_ms":setup_ms,"total_ms":started.elapsed().as_secs_f64()*1000.,"entries":rows,
        "scope":"All coarse and possible local windows come from map dimensions and fixed search rules, without query images or poses. Optional fine coverage includes every child of those local windows. The height digest and map geometry bind the namespace; preparation reads only source image pixels. Each crop retains its own encoder normalization context. Packets preserve descriptor bits and are decoded for exact round-trip verification. Accounted preparation memory includes source/crop/rotation and packet/feature allocations, excluding image decoding peaks, model/runtime scratch, and accumulated report/index metadata. Filesystem cache state is uncontrolled."}),
    )?;
    tracing::info!(path=%output.display(),packets=catalog.entries.len(),"Prepared map feature catalog");
    Ok(())
}
#[cfg(test)]
mod tests;
