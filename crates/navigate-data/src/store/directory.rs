//! A store in a directory of the native file system.

use alloc::string::String;
use alloc::vec::Vec;
use std::path::{Path, PathBuf};

use crate::store::checked_path;
use crate::{DataError, ReadPort, WritePort};

/// A store in a directory. The futures complete at once: the calls block
/// the caller for the file operation, so an asynchronous host runs them on
/// a thread that can block.
#[derive(Clone, Debug)]
pub struct DirectoryStore {
    root: PathBuf,
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

fn store_error(path: &str, error: &std::io::Error) -> DataError {
    DataError::Store {
        path: String::from(path),
        message: alloc::format!("{error}"),
    }
}

impl ReadPort for DirectoryStore {
    async fn read(&self, path: &str) -> Result<Option<Vec<u8>>, DataError> {
        match std::fs::read(self.file(path)?) {
            Ok(bytes) => Ok(Some(bytes)),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(store_error(path, &error)),
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
            std::fs::create_dir_all(parent).map_err(|e| store_error(path, &e))?;
        }
        let part = file.with_extension("part");
        std::fs::write(&part, bytes).map_err(|e| store_error(path, &e))?;
        std::fs::rename(&part, &file).map_err(|e| store_error(path, &e))
    }

    async fn contains(&self, path: &str) -> Result<bool, DataError> {
        Ok(self.file(path)?.is_file())
    }
}
