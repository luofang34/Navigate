//! Platform roots of the storage classes and a store that routes by class.

use navigate_data::{DataError, DataStore, DataUri, OpenFuture, StorageClass};
use std::path::{Path, PathBuf};

use crate::FileStore;

/// The platform directory of each storage class (Pilotage ADR-0044).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StorageRoots {
    data: PathBuf,
    offline: PathBuf,
    cache: PathBuf,
    config: PathBuf,
    temp: PathBuf,
}

impl StorageRoots {
    /// Roots that the host supplies, for example from Swift on Apple
    /// platforms, where the host also sets the backup exclusion.
    pub fn new(
        data: PathBuf,
        offline: PathBuf,
        cache: PathBuf,
        config: PathBuf,
        temp: PathBuf,
    ) -> Self {
        Self {
            data,
            offline,
            cache,
            config,
            temp,
        }
    }

    /// The standard roots on Linux, Windows, and macOS for one application.
    ///
    /// Linux uses the XDG base directories, Windows uses `AppData`, and macOS
    /// uses `Application Support` and `Caches`. `Offline` is a folder in the
    /// local, non-roaming data directory.
    ///
    /// # Errors
    /// Returns an error when the platform has no home directory.
    pub fn for_platform(organization: &str, application: &str) -> Result<Self, DataError> {
        let dirs =
            directories::ProjectDirs::from("", organization, application).ok_or_else(|| {
                DataError::Backend {
                    uri: application.into(),
                    operation: "resolve platform storage roots",
                    source: Box::new(std::io::Error::new(
                        std::io::ErrorKind::NotFound,
                        "no home directory",
                    )),
                }
            })?;
        Ok(Self {
            data: dirs.data_dir().to_owned(),
            offline: dirs.data_local_dir().join("Offline"),
            cache: dirs.cache_dir().to_owned(),
            config: dirs.config_dir().to_owned(),
            temp: std::env::temp_dir().join(application),
        })
    }

    /// The root directory of a class.
    pub fn root(&self, class: StorageClass) -> &Path {
        match class {
            StorageClass::Data => &self.data,
            StorageClass::Offline => &self.offline,
            StorageClass::Cache => &self.cache,
            StorageClass::Config => &self.config,
            _ => &self.temp,
        }
    }
}

/// A read store that routes each class URI to the root of its class.
///
/// Reading only: the Pilotage package store is the one writer of the
/// `Offline` class (Pilotage ADR-0044).
pub struct ClassStore {
    stores: Vec<(StorageClass, FileStore)>,
}

impl ClassStore {
    /// Create each class root if it is absent, and resolve it.
    ///
    /// # Errors
    /// Returns filesystem context when a root cannot be created or resolved.
    pub async fn new(roots: &StorageRoots) -> Result<Self, DataError> {
        let mut stores = Vec::with_capacity(StorageClass::ALL.len());
        for class in StorageClass::ALL {
            let root = roots.root(class);
            tokio::fs::create_dir_all(root)
                .await
                .map_err(|source| DataError::Backend {
                    uri: root.display().to_string(),
                    operation: "create storage root",
                    source: Box::new(source),
                })?;
            stores.push((class, FileStore::new(root).await?));
        }
        Ok(Self { stores })
    }
}

impl DataStore for ClassStore {
    fn open<'a>(&'a self, uri: &'a DataUri) -> OpenFuture<'a> {
        let store = uri.storage_class().and_then(|class| {
            self.stores
                .iter()
                .find(|(candidate, _)| *candidate == class)
                .map(|(_, store)| store)
        });
        match store {
            Some(store) => store.open(uri),
            None => Box::pin(async move {
                Err(DataError::InvalidUri {
                    uri: uri.as_str().into(),
                })
            }),
        }
    }
}

#[cfg(test)]
mod tests;
