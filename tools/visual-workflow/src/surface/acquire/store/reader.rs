//! The index is immutable; packet data is read only when a candidate needs it.
use super::{Catalog, Entry, hex};
use crate::{Error, assets};
use navigate_visual_onnx::XFeatFeatures;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    fs::File,
    io::Read,
    path::{Path, PathBuf},
    time::Instant,
};

pub(in crate::surface) struct Reader {
    root: PathBuf,
    source: [u8; 32],
    contract: [u8; 32],
    entries: Vec<Entry>,
    digest: [u8; 32],
    open_ms: f64,
    include_fine: bool,
    file_bytes: usize,
    open_buffer_bytes: usize,
}
pub(in crate::surface) struct Loaded {
    pub features: XFeatFeatures,
    pub buffer_bytes: usize,
    pub file_bytes: usize,
    pub read_ms: f64,
    pub decode_ms: f64,
    pub path: PathBuf,
}
impl Reader {
    pub fn open_blocking(path: &Path, source: [u8; 32], contract: [u8; 32]) -> Result<Self, Error> {
        let started = Instant::now();
        let bytes = assets::read_blocking(path, 4 * 1024 * 1024)?;
        let catalog: Catalog = serde_json::from_slice(&bytes).map_err(|source| Error::Json {
            path: path.into(),
            source,
        })?;
        if catalog.version != 1
            || catalog.entries.len() > 4096
            || catalog.entries.windows(2).any(|p| p[0].key >= p[1].key)
            || catalog
                .entries
                .iter()
                .any(|e| e.bytes < 116 || e.bytes > 116 + 4096 * 284)
        {
            return Err(Error::Input(format!(
                "invalid feature catalog index: {}",
                path.display()
            )));
        }
        let reader = Self {
            root: path.parent().unwrap_or(Path::new(".")).into(),
            source: catalog.source,
            contract: catalog.contract,
            entries: catalog.entries,
            digest: Sha256::digest(&bytes).into(),
            open_ms: started.elapsed().as_secs_f64() * 1000.,
            include_fine: catalog.include_fine,
            file_bytes: bytes.len(),
            open_buffer_bytes: bytes.capacity(),
        };
        reader.check(source, contract)?;
        Ok(reader)
    }
    pub fn check(&self, source: [u8; 32], contract: [u8; 32]) -> Result<(), Error> {
        if self.source != source || self.contract != contract {
            return Err(Error::Input(format!(
                "feature catalog source or encoder mismatch: {}",
                self.root.display()
            )));
        }
        Ok(())
    }
    pub fn resident_bytes(&self) -> usize {
        size_of::<Self>()
            + 2 * size_of::<usize>()
            + self.root.capacity()
            + self.entries.capacity() * size_of::<Entry>()
    }
    pub fn report(&self) -> Value {
        json!({"sha256":hex(&self.digest),"entries":self.entries.len(),"include_fine":self.include_fine,"open_ms":self.open_ms,"index_resident_bytes":self.resident_bytes(),"index_file_bytes":self.file_bytes,"index_open_buffer_bytes":self.open_buffer_bytes,
            "scope":"Index retained once across model reloads. Packets read on demand; missing keys use online encoding. Present unreadable, corrupt or incompatible packets fail. Rotated source pixels are still verified. File bytes are logical reads, not physical storage transfers."})
    }
    pub fn load_blocking(&self, key: [u8; 32], limit: usize) -> Result<Option<Loaded>, Error> {
        let Ok(index) = self.entries.binary_search_by_key(&key, |e| e.key) else {
            return Ok(None);
        };
        let entry = &self.entries[index];
        if entry.bytes > 116 + limit.saturating_mul(284) {
            return Err(Error::Input(
                "feature packet exceeds configured feature limit".into(),
            ));
        }
        let path = self.root.join(format!("{}.navfeat", hex(&key)));
        let started = Instant::now();
        let bytes = read_exact_blocking(&path, entry.bytes)?;
        let read_ms = started.elapsed().as_secs_f64() * 1000.;
        let started = Instant::now();
        if Sha256::digest(&bytes).as_slice() != entry.sha256 {
            return Err(Error::Input(format!(
                "feature packet digest mismatch: {}",
                path.display()
            )));
        }
        let features =
            XFeatFeatures::decode(&bytes, limit).map_err(|source| Error::CatalogPacket {
                path: path.clone(),
                source,
            })?;
        if features.contract() != self.contract {
            return Err(Error::Input(format!(
                "feature packet encoder mismatch: {}",
                path.display()
            )));
        }
        Ok(Some(Loaded {
            features,
            buffer_bytes: bytes.capacity(),
            file_bytes: bytes.len(),
            read_ms,
            decode_ms: started.elapsed().as_secs_f64() * 1000.,
            path,
        }))
    }
}
fn read_exact_blocking(path: &Path, length: usize) -> Result<Vec<u8>, Error> {
    let read = || -> std::io::Result<_> {
        let mut file = File::open(path)?;
        let mut bytes = vec![0; length];
        file.read_exact(&mut bytes)?;
        let mut extra = [0_u8; 1];
        if file.read(&mut extra)? != 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "packet has trailing bytes",
            ));
        }
        Ok(bytes)
    };
    read().map_err(|source| Error::Io {
        operation: "read feature packet",
        path: path.into(),
        source,
    })
}
#[cfg(test)]
mod tests;
