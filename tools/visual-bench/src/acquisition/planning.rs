use super::{
    files::{Catalog, Entry},
    terrain::Terrain,
};
use crate::BenchError;
use nalgebra::{UnitQuaternion, Vector3};
use navigate_visual::{
    CameraModel, CameraPose, PosePrior,
    place_retrieval::{ImageRetriever, ReferenceId},
};
use std::collections::{BTreeMap, BTreeSet};

pub(super) struct Plan<'a> {
    entries: BTreeMap<ReferenceId, &'a Entry>,
    groups: Vec<Vec<ReferenceId>>,
}
impl<'a> Plan<'a> {
    pub fn new(
        catalog: &'a Catalog,
        scales: &[f64],
        prior: &PosePrior,
    ) -> Result<Self, BenchError> {
        prior.validate()?;
        if scales.is_empty()
            || scales.len() > 8
            || scales.iter().any(|s| !s.is_finite() || *s <= 0.0)
            || scales
                .iter()
                .enumerate()
                .any(|(i, s)| scales[..i].contains(s))
        {
            return Err(super::invalid("invalid reference scale order"));
        }
        let mut entries = BTreeMap::new();
        for entry in &catalog.gallery {
            if !entry.width_m.is_finite()
                || entry.width_m <= 0.0
                || entry.center_enu_m.iter().any(|v| !v.is_finite())
                || entries.insert(ReferenceId(entry.id), entry).is_some()
            {
                return Err(super::invalid("invalid or duplicate reference entry"));
            }
        }
        let groups = scales
            .iter()
            .map(|scale| {
                catalog
                    .gallery
                    .iter()
                    .filter(|e| {
                        (e.width_m.round() - scale).abs() < 1e-6
                            && (e.center_enu_m[0] - prior.pose.position.x)
                                .hypot(e.center_enu_m[1] - prior.pose.position.y)
                                <= prior.position_radius_m + e.width_m / std::f64::consts::SQRT_2
                    })
                    .map(|e| ReferenceId(e.id))
                    .collect()
            })
            .collect();
        Ok(Self { entries, groups })
    }
    pub fn rank_blocking(
        &self,
        retriever: &mut dyn ImageRetriever,
        rgb: &image::RgbImage,
    ) -> Result<Vec<ReferenceId>, BenchError> {
        let gray = image::RgbImage::from_fn(rgb.width(), rgb.height(), |x, y| {
            let p = rgb.get_pixel(x, y).0;
            let v =
                ((u32::from(p[0]) * 77 + u32::from(p[1]) * 150 + u32::from(p[2]) * 29) >> 8) as u8;
            image::Rgb([v, v, v])
        });
        let mut rankings = Vec::new();
        for input in [&gray, rgb] {
            let mut groups = Vec::new();
            for eligible in &self.groups {
                let ranked = retriever.rank_blocking(input, eligible, 8)?;
                let unique = ranked.iter().copied().collect::<BTreeSet<_>>();
                if ranked.len() > 8
                    || unique.len() != ranked.len()
                    || ranked.iter().any(|id| !eligible.contains(id))
                {
                    return Err(super::invalid(
                        "retriever returned duplicate or ineligible references",
                    ));
                }
                groups.push(ranked);
            }
            rankings.push(groups);
        }
        let mut ids = Vec::new();
        for rank in 0..8 * self.groups.len() {
            for groups in &rankings {
                if let Some(&id) = groups[rank % self.groups.len()].get(rank / self.groups.len())
                    && !ids.contains(&id)
                    && ids.len() < 16
                {
                    ids.push(id);
                }
            }
            if ids.len() == 16 {
                break;
            }
        }
        Ok(ids)
    }
    pub fn candidates(
        &self,
        ids: &[ReferenceId],
        orientations: &[UnitQuaternion<f64>],
        camera: CameraModel,
        terrain: &Terrain,
    ) -> Result<(Vec<CameraPose>, Vec<serde_json::Value>), BenchError> {
        camera.validate()?;
        if ids.len() > 16 || orientations.is_empty() || orientations.len() > 8 {
            return Err(super::invalid("reference candidate budget exceeds 128"));
        }
        let mut poses = Vec::new();
        let mut unsupported = Vec::new();
        for id in ids {
            let entry = self
                .entries
                .get(id)
                .ok_or_else(|| super::invalid("unknown reference entry"))?;
            let [east, north] = entry.center_enu_m;
            let Some(elevation) = terrain.elevation(east, north) else {
                unsupported.push(
                    serde_json::json!({"reference_id":id.0,"reason":"no valid terrain elevation"}),
                );
                continue;
            };
            let position = Vector3::new(
                east,
                north,
                elevation + entry.width_m * camera.fx / f64::from(camera.width),
            );
            for &orientation in orientations {
                poses.push(CameraPose {
                    position,
                    orientation,
                });
            }
        }
        Ok((poses, unsupported))
    }
}

#[cfg(test)]
mod tests;
