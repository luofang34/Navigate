//! Host fetch adapters for the `package` subcommand. Navigate's library
//! never fetches; this tool is a host.

use crate::{PipelineError, external_tool::run_tool};
use navigate_data::{FetchPort, Fetched, LayerKind, LayerSpec, Revision, TileKey};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// One layer of a URL template source.
#[derive(Clone, Debug)]
pub struct TemplateLayer {
    /// URL template with `{z}`, `{x}`, and `{y}`. `file://` reads local
    /// files; other schemes run `curl`.
    pub url: String,
    /// The layer that the package records: source, revision, datums,
    /// encoding, and accuracy.
    pub spec: LayerSpec,
    /// Attribution text for the descriptor.
    pub attribution: String,
}

/// A fetch port over URL templates, with a cache keyed by the tile key
/// (`<source>/<revision>/<z>/<x>/<y>`), not by the URL.
#[derive(Clone, Debug)]
pub struct TemplateFetch {
    layers: BTreeMap<LayerKind, TemplateLayer>,
    cache: PathBuf,
}

impl TemplateFetch {
    /// A port with a cache directory.
    pub fn new(cache: &Path) -> Self {
        Self {
            layers: BTreeMap::new(),
            cache: cache.to_path_buf(),
        }
    }

    /// Adds a layer.
    #[must_use]
    pub fn with_layer(mut self, kind: LayerKind, layer: TemplateLayer) -> Self {
        self.layers.insert(kind, layer);
        self
    }

    /// The declared layers.
    pub fn specs(&self) -> Vec<LayerSpec> {
        self.layers.values().map(|l| l.spec).collect()
    }

    fn cache_path(&self, key: &TileKey) -> PathBuf {
        let a = key.address;
        let source = serde_json::to_string(&key.source).unwrap_or_default();
        self.cache
            .join(source.trim_matches('"'))
            .join(revision_text(key.revision))
            .join(format!("{}/{}/{}", a.zoom, a.x, a.y))
    }

    /// Gets one tile from the cache, or else from the template.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError`] for an unknown layer, a failed read or
    /// write, and an HTTP status other than 200, 204, and 404.
    pub fn fetch_blocking(&self, key: &TileKey) -> Result<Fetched, PipelineError> {
        let layer = self
            .layers
            .get(&key.layer)
            .ok_or_else(|| PipelineError::Invalid(format!("no {:?} layer", key.layer)))?;
        let cached = self.cache_path(key);
        if cached.is_file() {
            return read(&cached).map(Fetched::Tile);
        }
        let a = key.address;
        let url = layer
            .url
            .replace("{z}", &a.zoom.to_string())
            .replace("{x}", &a.x.to_string())
            .replace("{y}", &a.y.to_string());
        let bytes = match url.strip_prefix("file://") {
            Some(local) if Path::new(local).is_file() => read(Path::new(local))?,
            Some(_) => return Ok(Fetched::Absent),
            None => match download(&url, &cached)? {
                Some(bytes) => bytes,
                None => return Ok(Fetched::Absent),
            },
        };
        write(&cached, &bytes)?;
        Ok(Fetched::Tile(bytes))
    }

    /// Fills the cache on parallel threads, so that the package writer
    /// then reads each tile from the cache.
    ///
    /// # Errors
    ///
    /// Returns the first fetch error.
    pub fn prefetch_blocking(&self, keys: &[TileKey], threads: usize) -> Result<(), PipelineError> {
        let per_thread = keys.len().div_ceil(threads.max(1)).max(1);
        std::thread::scope(|scope| {
            let workers: Vec<_> = keys
                .chunks(per_thread)
                .map(|chunk| {
                    scope.spawn(move || {
                        chunk
                            .iter()
                            .try_for_each(|key| self.fetch_blocking(key).map(drop))
                    })
                })
                .collect();
            workers.into_iter().try_for_each(|worker| {
                worker
                    .join()
                    .map_err(|_| PipelineError::Invalid("tile fetch thread failed".into()))?
            })
        })
    }
}

/// A revision as one path part: `unversioned`, `release-N`, or
/// `collected-D` (day number).
pub fn revision_text(revision: Revision) -> String {
    match revision {
        Revision::Unversioned => "unversioned".into(),
        Revision::Release(n) => format!("release-{n}"),
        Revision::Collected(day) => format!("collected-{}", day.0),
    }
}

impl FetchPort for TemplateFetch {
    async fn fetch(&self, key: &TileKey) -> Result<Fetched, String> {
        self.fetch_blocking(key).map_err(|error| error.to_string())
    }
}

fn read(path: &Path) -> Result<Vec<u8>, PipelineError> {
    std::fs::read(path).map_err(|source| PipelineError::Io {
        path: path.to_path_buf(),
        source,
    })
}

fn write(path: &Path, bytes: &[u8]) -> Result<(), PipelineError> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|source| PipelineError::Io {
            path: parent.to_path_buf(),
            source,
        })?;
    }
    let part = path.with_extension("part");
    std::fs::write(&part, bytes).map_err(|source| PipelineError::Io {
        path: part.clone(),
        source,
    })?;
    std::fs::rename(&part, path).map_err(|source| PipelineError::Io {
        path: path.to_path_buf(),
        source,
    })
}

fn download(url: &str, cached: &Path) -> Result<Option<Vec<u8>>, PipelineError> {
    let part = cached.with_extension("download");
    if let Some(parent) = part.parent() {
        std::fs::create_dir_all(parent).map_err(|source| PipelineError::Io {
            path: parent.to_path_buf(),
            source,
        })?;
    }
    let part_s = part.to_string_lossy().into_owned();
    let status = run_tool(
        "curl",
        &[
            "-sSL",
            "--retry",
            "3",
            "-o",
            &part_s,
            "-w",
            "%{http_code}",
            url,
        ],
    )?;
    let status = String::from_utf8_lossy(&status).trim().to_owned();
    let result = match status.as_str() {
        "200" => read(&part).map(Some),
        "204" | "404" => Ok(None),
        other => Err(PipelineError::Tool {
            tool: "curl".into(),
            message: format!("HTTP status {other} for {url}"),
        }),
    };
    drop(std::fs::remove_file(&part));
    result
}

/// A fetch port over a `photo-reference-catalog` directory, so that an
/// existing catalog becomes a package. It answers by layer kind and
/// address; the host declares the source of each layer.
#[derive(Clone, Debug)]
pub struct CatalogFetch {
    root: PathBuf,
    tiles: BTreeMap<(LayerKind, [u32; 3]), String>,
    attribution: BTreeMap<LayerKind, String>,
}

impl CatalogFetch {
    /// Reads `catalog.json` in `root`.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError`] when the catalog does not read or decode.
    pub fn open_blocking(root: &Path) -> Result<Self, PipelineError> {
        let path = root.join("catalog.json");
        let doc: serde_json::Value = crate::json_file::read_json(&path)?;
        if doc.get("format").and_then(serde_json::Value::as_str) != Some("photo-reference-catalog")
        {
            return Err(PipelineError::Invalid(format!(
                "{} is not a catalog",
                path.display()
            )));
        }
        let mut tiles = BTreeMap::new();
        for tile in doc
            .get("tiles")
            .and_then(serde_json::Value::as_array)
            .into_iter()
            .flatten()
        {
            let kind = match tile.get("kind").and_then(serde_json::Value::as_str) {
                Some("imagery") => LayerKind::Imagery,
                Some("dem" | "terrain") => LayerKind::Elevation,
                _ => continue,
            };
            let xyz: Option<[u32; 3]> = tile
                .get("xyz")
                .cloned()
                .and_then(|v| serde_json::from_value(v).ok());
            let file = tile.get("path").and_then(serde_json::Value::as_str);
            if let (Some(xyz), Some(file)) = (xyz, file) {
                tiles.insert((kind, xyz), file.to_owned());
            }
        }
        let text = |name: &str| {
            doc.pointer(&format!("/attribution/{name}"))
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default()
                .to_owned()
        };
        let attribution = BTreeMap::from([
            (LayerKind::Imagery, text("imagery")),
            (LayerKind::Elevation, text("dem")),
        ]);
        Ok(Self {
            root: root.to_path_buf(),
            tiles,
            attribution,
        })
    }

    /// The catalog tiles of a layer, `[z, x, y]`.
    pub fn tiles(&self, kind: LayerKind) -> impl Iterator<Item = [u32; 3]> + '_ {
        self.tiles
            .keys()
            .filter(move |(k, _)| *k == kind)
            .map(|(_, xyz)| *xyz)
    }

    /// The attribution text of a layer in the catalog.
    pub fn attribution(&self, kind: LayerKind) -> &str {
        self.attribution.get(&kind).map_or("", String::as_str)
    }
}

impl FetchPort for CatalogFetch {
    async fn fetch(&self, key: &TileKey) -> Result<Fetched, String> {
        let a = key.address;
        let Some(file) = self.tiles.get(&(key.layer, [a.zoom, a.x, a.y])) else {
            return Ok(Fetched::Absent);
        };
        let path = self.root.join(file);
        match std::fs::read(&path) {
            Ok(bytes) => Ok(Fetched::Tile(bytes)),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(Fetched::Absent),
            Err(error) => Err(format!("{}: {error}", path.display())),
        }
    }
}
