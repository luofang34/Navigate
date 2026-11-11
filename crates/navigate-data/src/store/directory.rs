//! A store in a directory of the native file system.

use alloc::string::ToString;
use alloc::vec::Vec;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use crate::store::checked_path;
use crate::{DataError, RandomAccess, ReadPort, WritePort};

/// A store in a directory. The futures complete at once: each call blocks
/// the caller for the file operation, so an asynchronous host runs them on
/// a thread that can block.
#[derive(Clone, Debug)]
pub struct DirectoryStore {
    root: PathBuf,
}

/// An open file of a [`DirectoryStore`].
#[derive(Clone, Debug)]
pub struct DirectoryFile {
    path: PathBuf,
    len: u64,
}

impl DirectoryStore {
    /// A store at `root`. The directory is made at the first write.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// The root directory.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    fn file(&self, path: &str) -> Result<PathBuf, DataError> {
        checked_path(path).map(|p| self.root.join(p))
    }
}

fn store_error(path: &Path, error: &std::io::Error) -> DataError {
    DataError::Store {
        path: path.display().to_string(),
        message: alloc::format!("{error}"),
    }
}

impl RandomAccess for DirectoryFile {
    fn len(&self) -> u64 {
        self.len
    }

    async fn read_at(&self, offset: u64, length: usize) -> Result<Vec<u8>, DataError> {
        let mut file = std::fs::File::open(&self.path).map_err(|e| store_error(&self.path, &e))?;
        file.seek(SeekFrom::Start(offset))
            .map_err(|e| store_error(&self.path, &e))?;
        let mut out = alloc::vec![0_u8; length];
        file.read_exact(&mut out)
            .map_err(|e| store_error(&self.path, &e))?;
        Ok(out)
    }
}

impl ReadPort for DirectoryStore {
    type File = DirectoryFile;

    async fn open(&self, path: &str) -> Result<Option<DirectoryFile>, DataError> {
        let file = self.file(path)?;
        match std::fs::metadata(&file) {
            Ok(meta) if meta.is_file() => Ok(Some(DirectoryFile {
                path: file,
                len: meta.len(),
            })),
            Ok(_) => Ok(None),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(store_error(&file, &error)),
        }
    }
}

impl WritePort for DirectoryStore {
    /// Writes to a temporary file and renames it, so that a reader never
    /// sees part of a file.
    async fn write(&mut self, path: &str, bytes: &[u8]) -> Result<(), DataError> {
        let file = self.file(path)?;
        if file.is_file() {
            return Ok(());
        }
        if let Some(parent) = file.parent() {
            std::fs::create_dir_all(parent).map_err(|e| store_error(parent, &e))?;
        }
        let part = file.with_extension("part");
        std::fs::write(&part, bytes).map_err(|e| store_error(&part, &e))?;
        std::fs::rename(&part, &file).map_err(|e| store_error(&file, &e))
    }

    async fn contains(&self, path: &str) -> Result<bool, DataError> {
        Ok(self.file(path)?.is_file())
    }
}
