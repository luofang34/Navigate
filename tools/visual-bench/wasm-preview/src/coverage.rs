use crate::model::Manifest;
use std::collections::{BTreeMap, HashSet};

type TileLevels = BTreeMap<u32, HashSet<(u32, u32)>>;
pub(crate) struct Coverage {
    imagery: TileLevels,
    terrain: TileLevels,
}
impl Coverage {
    pub fn new(manifest: &Manifest) -> Self {
        let mut value = Self {
            imagery: BTreeMap::new(),
            terrain: BTreeMap::new(),
        };
        for tile in &manifest.tiles {
            let [z, x, y] = tile.xyz;
            if tile.imagery.is_some() {
                value.imagery.entry(z).or_default().insert((x, y));
            }
            if tile.elevation.is_some() {
                value.terrain.entry(z).or_default().insert((x, y));
            }
        }
        value
    }
    pub fn supports(&self, xy: [f64; 2]) -> bool {
        if !xy.iter().all(|v| v.is_finite() && (0.0..1.0).contains(v)) {
            return false;
        }
        let contains = |levels: &TileLevels| {
            levels.iter().any(|(&zoom, tiles)| {
                let n = f64::from(1_u32 << zoom);
                tiles.contains(&((xy[0] * n).floor() as u32, (xy[1] * n).floor() as u32))
            })
        };
        contains(&self.imagery) && contains(&self.terrain)
    }
}
#[cfg(test)]
#[path = "coverage/tests.rs"]
mod tests;
