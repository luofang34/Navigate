//! Windowed Hilbert-sharded 3D corridor tile store (`NVKTIDX1` / `NVKTSHD1`).
//!
//! [`CorridorTileStore`] keeps a bounded Least-Recently-Used (LRU) window of active
//! [`CorridorTileEntry`] cells in RAM (`< 10 MiB` regardless of corridor length) while
//! loading missing tiles from Hilbert-bucketed shard files on demand and accepting new
//! tiles synthesized asynchronously by a background worker.

mod disk;

use super::{
    CorridorLandmarkBank, CorridorTileEntry, CorridorTileKey, MAX_TILES, VirtualCorridorView,
    VirtualViewCamera, codec::ensure, project::project_tiles_virtual_view,
};
use crate::{HadamardProjector, MatchError};
pub use disk::hilbert_tile_code;
#[cfg(not(target_arch = "wasm32"))]
use disk::{decode_index_bytes, io_err, read_shard_file, write_index_file, write_shard_file};
use std::{
    collections::{BTreeSet, HashMap, VecDeque},
    sync::{Arc, RwLock},
};

/// Eight-byte magic header of a sharded tile store index (`manifest.nvktidx`).
pub const CORRIDOR_STORE_INDEX_MAGIC: &[u8; 8] = b"NVKTIDX1";
/// Eight-byte magic header of a single sharded [`CorridorTileEntry`] file (`.nvkt`).
pub const CORRIDOR_SHARD_MAGIC: &[u8; 8] = b"NVKTSHD1";
/// Default maximum number of active tiles retained in RAM (`5 x 5` neighborhood).
pub const DEFAULT_WINDOW_TILES: usize = 25;
/// Hard byte ceiling (`8 MiB`) on the resident in-memory LRU tile window (`< 10 MiB`).
const MAX_RESIDENT_WINDOW_BYTES: usize = 8 * 1024 * 1024;

#[derive(Debug)]
enum BackingSource {
    InMemory(HashMap<CorridorTileKey, Arc<CorridorTileEntry>>),
    #[cfg(not(target_arch = "wasm32"))]
    Directory(std::path::PathBuf),
}

#[derive(Debug)]
struct StoreInner {
    indexed_keys: BTreeSet<CorridorTileKey>,
    cache_order: VecDeque<CorridorTileKey>,
    cache_entries: HashMap<CorridorTileKey, Arc<CorridorTileEntry>>,
    backing: BackingSource,
}

/// Thread-safe windowed store of 3D [`CorridorTileEntry`] cells.
#[derive(Debug)]
pub struct CorridorTileStore {
    /// SHA-256 digest of the source terrain and imagery catalog.
    pub catalog_sha256: [u8; 32],
    /// SHA-256 digest of the [`XFeat`](crate::XFeat) encoder contract.
    pub encoder_contract: [u8; 32],
    /// Seed of the [`HadamardProjector`] for all tiles in this store.
    pub seed: u64,
    /// Horizontal ground tile edge length in ENU meters.
    pub tile_size_m: f32,
    max_cached_tiles: usize,
    inner: RwLock<StoreInner>,
}

impl CorridorTileStore {
    /// Construct an empty in-memory [`CorridorTileStore`] with an LRU window of `max_cached_tiles`.
    ///
    /// # Errors
    ///
    /// Returns [`MatchError`] when `tile_size_m` is invalid.
    pub fn in_memory(
        catalog_sha256: [u8; 32],
        encoder_contract: [u8; 32],
        seed: u64,
        tile_size_m: f32,
        max_cached_tiles: usize,
    ) -> Result<Self, MatchError> {
        ensure(
            tile_size_m.is_finite() && tile_size_m >= 10.0,
            "invalid corridor store tile_size_m",
        )?;
        let cap = max_cached_tiles.clamp(1, MAX_TILES);
        Ok(Self {
            catalog_sha256,
            encoder_contract,
            seed,
            tile_size_m,
            max_cached_tiles: cap,
            inner: RwLock::new(StoreInner {
                indexed_keys: BTreeSet::new(),
                cache_order: VecDeque::with_capacity(cap),
                cache_entries: HashMap::with_capacity(cap),
                backing: BackingSource::InMemory(HashMap::new()),
            }),
        })
    }

    /// Wrap an in-memory [`CorridorLandmarkBank`] in a windowed [`CorridorTileStore`].
    #[must_use]
    pub fn from_bank(bank: CorridorLandmarkBank, max_cached_tiles: usize) -> Self {
        let cap = max_cached_tiles.clamp(1, MAX_TILES);
        let mut indexed_keys = BTreeSet::new();
        let mut all = HashMap::with_capacity(bank.tiles.len());
        for tile in bank.tiles {
            let key = CorridorTileKey {
                tile_x: tile.tile_x,
                tile_y: tile.tile_y,
            };
            indexed_keys.insert(key);
            all.insert(key, Arc::new(tile));
        }
        Self {
            catalog_sha256: bank.catalog_sha256,
            encoder_contract: bank.encoder_contract,
            seed: bank.seed,
            tile_size_m: bank.tile_size_m,
            max_cached_tiles: cap,
            inner: RwLock::new(StoreInner {
                indexed_keys,
                cache_order: VecDeque::with_capacity(cap),
                cache_entries: HashMap::with_capacity(cap),
                backing: BackingSource::InMemory(all),
            }),
        }
    }

    /// Write `bank` into a Hilbert-sharded directory `dir` and open it with `max_cached_tiles`.
    ///
    /// # Errors
    ///
    /// Returns [`MatchError`] when directory creation or shard writing fails.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn write_bank_dir(
        bank: &CorridorLandmarkBank,
        dir: &std::path::Path,
        max_cached_tiles: usize,
    ) -> Result<Self, MatchError> {
        bank.check()?;
        std::fs::create_dir_all(dir).map_err(|e| io_err("corridor store mkdir", e))?;
        let mut keys = BTreeSet::new();
        for tile in &bank.tiles {
            let key = CorridorTileKey {
                tile_x: tile.tile_x,
                tile_y: tile.tile_y,
            };
            write_shard_file(dir, key, tile)?;
            keys.insert(key);
        }
        write_index_file(
            &dir.join("manifest.nvktidx"),
            bank.catalog_sha256,
            bank.encoder_contract,
            bank.seed,
            bank.tile_size_m,
            &keys,
        )?;
        Self::open_dir(dir, max_cached_tiles)
    }

    /// Create an empty sharded directory store at `dir`.
    ///
    /// # Errors
    ///
    /// Returns [`MatchError`] when `tile_size_m` is invalid or `dir` cannot be initialized.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn create_dir(
        dir: &std::path::Path,
        catalog_sha256: [u8; 32],
        encoder_contract: [u8; 32],
        seed: u64,
        tile_size_m: f32,
        max_cached_tiles: usize,
    ) -> Result<Self, MatchError> {
        ensure(
            tile_size_m.is_finite() && tile_size_m >= 10.0,
            "invalid corridor store tile_size_m",
        )?;
        std::fs::create_dir_all(dir).map_err(|e| io_err("corridor store mkdir", e))?;
        let keys = BTreeSet::new();
        write_index_file(
            &dir.join("manifest.nvktidx"),
            catalog_sha256,
            encoder_contract,
            seed,
            tile_size_m,
            &keys,
        )?;
        Self::open_dir(dir, max_cached_tiles)
    }

    /// Open either a sharded tile store directory or a monolithic `.nvktile` file at `path`.
    ///
    /// # Errors
    ///
    /// Returns [`MatchError`] when `path` cannot be read or fails checksum validation.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn open(path: &std::path::Path, max_cached_tiles: usize) -> Result<Self, MatchError> {
        if path.is_dir() {
            Self::open_dir(path, max_cached_tiles)
        } else {
            let bank = CorridorLandmarkBank::load_file(path)?;
            Ok(Self::from_bank(bank, max_cached_tiles))
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn open_dir(dir: &std::path::Path, max_cached_tiles: usize) -> Result<Self, MatchError> {
        let idx_bytes = std::fs::read(dir.join("manifest.nvktidx"))
            .map_err(|e| io_err("corridor store index read", e))?;
        let (catalog_sha256, encoder_contract, seed, tile_size_m, indexed_keys) =
            decode_index_bytes(&idx_bytes)?;
        let cap = max_cached_tiles.clamp(1, MAX_TILES);
        Ok(Self {
            catalog_sha256,
            encoder_contract,
            seed,
            tile_size_m,
            max_cached_tiles: cap,
            inner: RwLock::new(StoreInner {
                indexed_keys,
                cache_order: VecDeque::with_capacity(cap),
                cache_entries: HashMap::with_capacity(cap),
                backing: BackingSource::Directory(dir.to_path_buf()),
            }),
        })
    }

    /// Map ENU coordinates `(east_m, north_m)` to their integer [`CorridorTileKey`].
    #[must_use]
    pub fn tile_key_at(&self, east_m: f64, north_m: f64) -> CorridorTileKey {
        let s = f64::from(self.tile_size_m.max(10.0));
        CorridorTileKey {
            tile_x: (east_m / s).floor() as i32,
            tile_y: (north_m / s).floor() as i32,
        }
    }

    /// Return the total number of indexed tiles known to this store.
    #[must_use]
    pub fn indexed_tile_count(&self) -> usize {
        self.inner.read().map(|g| g.indexed_keys.len()).unwrap_or(0)
    }

    /// Return the number of tiles currently resident in the active LRU window.
    #[must_use]
    pub fn cached_tile_count(&self) -> usize {
        self.inner
            .read()
            .map(|g| g.cache_entries.len())
            .unwrap_or(0)
    }

    /// Return the approximate RAM bytes occupied by the active LRU tile window.
    #[must_use]
    pub fn resident_window_bytes(&self) -> usize {
        let Ok(guard) = self.inner.read() else {
            return 0;
        };
        cache_window_bytes(&guard)
    }

    /// Return `true` when `key` is already present in this store.
    #[must_use]
    pub fn has_tile(&self, key: CorridorTileKey) -> bool {
        self.inner
            .read()
            .map(|g| g.indexed_keys.contains(&key))
            .unwrap_or(false)
    }

    /// Insert or replace `tile` in the live LRU window and optionally persist it on disk.
    ///
    /// # Errors
    ///
    /// Returns [`MatchError`] when `tile` is invalid or disk persistence fails.
    pub fn insert_tile(&self, tile: CorridorTileEntry, persist: bool) -> Result<(), MatchError> {
        #[cfg(target_arch = "wasm32")]
        let _ = persist;
        tile.validate()?;
        let key = CorridorTileKey {
            tile_x: tile.tile_x,
            tile_y: tile.tile_y,
        };
        let mut guard = self.inner.write().map_err(|_| MatchError::Device {
            stage: "corridor store write lock",
            message: "poisoned".into(),
        })?;
        let tile_arc = Arc::new(tile);
        match &mut guard.backing {
            BackingSource::InMemory(all) => {
                let _ = all.insert(key, Arc::clone(&tile_arc));
                let _ = guard.indexed_keys.insert(key);
            }
            #[cfg(not(target_arch = "wasm32"))]
            BackingSource::Directory(dir) => {
                let dir_clone = dir.clone();
                if persist {
                    write_shard_file(&dir_clone, key, &tile_arc)?;
                }
                let is_new = guard.indexed_keys.insert(key);
                if persist && is_new {
                    write_index_file(
                        &dir_clone.join("manifest.nvktidx"),
                        self.catalog_sha256,
                        self.encoder_contract,
                        self.seed,
                        self.tile_size_m,
                        &guard.indexed_keys,
                    )?;
                }
            }
        }
        touch_lru(&mut guard, key, tile_arc, self.max_cached_tiles);
        Ok(())
    }

    /// Load only the active spatial window around `cam` into the LRU cache and project `cam`.
    ///
    /// # Errors
    ///
    /// Returns [`MatchError`] when reading a required shard from disk fails.
    pub fn project_virtual_view(
        &self,
        cam: &VirtualViewCamera,
        projector: &HadamardProjector,
    ) -> Result<VirtualCorridorView, MatchError> {
        let candidates = self.select_window_keys(cam);
        let mut active_tiles = Vec::with_capacity(candidates.len());
        let mut guard = self.inner.write().map_err(|_| MatchError::Device {
            stage: "corridor store cache lock",
            message: "poisoned".into(),
        })?;
        for key in candidates {
            if let Some(cached) = guard.cache_entries.get(&key).cloned() {
                promote_key(&mut guard.cache_order, key);
                active_tiles.push(cached);
                continue;
            }
            let loaded = match &guard.backing {
                BackingSource::InMemory(all) => all.get(&key).cloned(),
                #[cfg(not(target_arch = "wasm32"))]
                BackingSource::Directory(dir) => read_shard_file(dir, key)?.map(Arc::new),
            };
            if let Some(tile) = loaded {
                touch_lru(&mut guard, key, Arc::clone(&tile), self.max_cached_tiles);
                active_tiles.push(tile);
            }
        }
        drop(guard);
        Ok(project_tiles_virtual_view(
            active_tiles.iter().map(Arc::as_ref),
            cam,
            projector,
        ))
    }

    fn select_window_keys(&self, cam: &VirtualViewCamera) -> Vec<CorridorTileKey> {
        let Ok(guard) = self.inner.read() else {
            return Vec::new();
        };
        let s = f64::from(self.tile_size_m.max(10.0));
        let fwd = [cam.world_to_cam[6], cam.world_to_cam[7]];
        let fwd_z = cam.world_to_cam[8];
        let fwd_h = (fwd[0] * fwd[0] + fwd[1] * fwd[1]).sqrt();
        let upward_horizon = fwd_h > 0.25 && -fwd_z < 0.05;
        let shallow_horizon = fwd_h > 0.25 && -fwd_z < 0.25;
        let look_dist = if upward_horizon {
            (cam.eye_enu[2].abs() * 8.5).clamp(5_500.0, 13_000.0)
        } else if shallow_horizon {
            (cam.eye_enu[2].abs() * 5.5).clamp(3_500.0, 7_500.0)
        } else if fwd_h > 0.25 {
            (cam.eye_enu[2].abs() * 2.5).clamp(0.0, 3_000.0)
        } else {
            0.0
        };
        let ux = if fwd_h > 1e-6 { fwd[0] / fwd_h } else { 0.0 };
        let uy = if fwd_h > 1e-6 { fwd[1] / fwd_h } else { 0.0 };
        let focus_e = cam.eye_enu[0] + ux * look_dist;
        let focus_n = cam.eye_enu[1] + uy * look_dist;
        let max_range_sq = if upward_horizon {
            (22_000.0_f64 + s).powi(2)
        } else if shallow_horizon {
            (14_000.0_f64 + s).powi(2)
        } else {
            (6_800.0_f64 + s).powi(2)
        };

        let mut ranked: Vec<(f64, CorridorTileKey)> = guard
            .indexed_keys
            .iter()
            .filter_map(|&key| {
                let cx = (f64::from(key.tile_x) + 0.5) * s;
                let cy = (f64::from(key.tile_y) + 0.5) * s;
                let de_eye = cx - cam.eye_enu[0];
                let dn_eye = cy - cam.eye_enu[1];
                if de_eye * de_eye + dn_eye * dn_eye > max_range_sq {
                    return None;
                }
                if fwd_h > 0.5 && (de_eye * ux + dn_eye * uy) < -s * 1.25 {
                    return None;
                }
                let de_foc = cx - focus_e;
                let dn_foc = cy - focus_n;
                Some((de_foc * de_foc + dn_foc * dn_foc, key))
            })
            .collect();
        ranked.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut selected: Vec<CorridorTileKey> = ranked
            .into_iter()
            .take(self.max_cached_tiles)
            .map(|(_, k)| k)
            .collect();
        selected.sort_by_key(|k| (k.tile_y, k.tile_x));
        selected
    }
}

fn promote_key(order: &mut VecDeque<CorridorTileKey>, key: CorridorTileKey) {
    if let Some(pos) = order.iter().position(|&k| k == key) {
        let _ = order.remove(pos);
    }
    order.push_back(key);
}

fn cache_window_bytes(inner: &StoreInner) -> usize {
    inner
        .cache_entries
        .values()
        .map(|t| size_of::<CorridorTileEntry>() + t.len() * 93)
        .sum()
}

fn touch_lru(
    inner: &mut StoreInner,
    key: CorridorTileKey,
    tile: Arc<CorridorTileEntry>,
    cap: usize,
) {
    promote_key(&mut inner.cache_order, key);
    let _ = inner.cache_entries.insert(key, tile);
    while inner.cache_entries.len() > cap
        || (inner.cache_entries.len() > 1 && cache_window_bytes(inner) > MAX_RESIDENT_WINDOW_BYTES)
    {
        let Some(evicted) = inner.cache_order.pop_front() else {
            break;
        };
        let _ = inner.cache_entries.remove(&evicted);
    }
}
