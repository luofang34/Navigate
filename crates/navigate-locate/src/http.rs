//! Esri World Imagery and Terrarium elevation tiles over HTTP, kept in a disk cache.
//!
//! A run with the same cache directory uses the same map bytes, so a golden case
//! stays fixed after its first run. Browser hosts supply their own source.
use crate::{TileKind, TileSource};
use std::path::PathBuf;

/// Public tile endpoints with a cache directory.
pub struct CachedHttpSource {
    directory: PathBuf,
}

impl CachedHttpSource {
    /// Use `directory` as the cache. It is created when a tile is first written.
    pub fn new(directory: impl Into<PathBuf>) -> Self {
        Self {
            directory: directory.into(),
        }
    }

    fn address(kind: TileKind, z: u32, x: u32, y: u32) -> (String, String) {
        match kind {
            TileKind::Imagery => (
                format!("imagery/{z}/{x}/{y}.jpg"),
                format!(
                    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}?blankTile=false"
                ),
            ),
            TileKind::Terrain => (
                format!("terrain/{z}/{x}/{y}.png"),
                format!("https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"),
            ),
        }
    }
}

impl TileSource for CachedHttpSource {
    async fn fetch(
        &self,
        kind: TileKind,
        z: u32,
        x: u32,
        y: u32,
    ) -> Result<Option<Vec<u8>>, String> {
        let (name, url) = Self::address(kind, z, x, y);
        let path = self.directory.join(name);
        if let Ok(bytes) = std::fs::read(&path) {
            return Ok(Some(bytes));
        }
        let mut response = match ureq::get(&url).call() {
            Ok(response) => response,
            Err(ureq::Error::StatusCode(404)) => return Ok(None),
            Err(error) => return Err(format!("{url}: {error}")),
        };
        let bytes = response
            .body_mut()
            .read_to_vec()
            .map_err(|e| format!("{url}: {e}"))?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
        }
        std::fs::write(&path, &bytes).map_err(|e| format!("{}: {e}", path.display()))?;
        Ok(Some(bytes))
    }

    fn identity(&self) -> String {
        "esri-world-imagery+terrarium".into()
    }
}
