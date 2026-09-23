use navigate_data::{
    DataError, DataStore, DataUri, OpenFuture, RandomAccess, ReadFuture, check_range,
};
use std::{
    fs::File,
    io,
    path::{Path, PathBuf},
    sync::Arc,
};
/// A filesystem adapter with a canonical storage root.
pub struct FileStore {
    root: PathBuf,
}
struct Resource {
    file: Arc<File>,
    uri: DataUri,
    size: u64,
}
fn failure(
    uri: &str,
    operation: &'static str,
    source: impl std::error::Error + 'static,
) -> DataError {
    DataError::Backend {
        uri: uri.into(),
        operation,
        source: Box::new(source),
    }
}
impl FileStore {
    /// Resolve a storage directory before opening logical resources.
    ///
    /// # Errors
    /// Returns filesystem context if the directory cannot be resolved.
    pub async fn new(root: impl AsRef<Path>) -> Result<Self, DataError> {
        let path = root.as_ref();
        let root = tokio::fs::canonicalize(path)
            .await
            .map_err(|e| failure(&path.display().to_string(), "resolve storage root", e))?;
        let metadata = tokio::fs::metadata(&root)
            .await
            .map_err(|e| failure(&root.display().to_string(), "read root metadata", e))?;
        if !metadata.is_dir() {
            return Err(failure(
                &root.display().to_string(),
                "validate storage root",
                io::Error::new(
                    io::ErrorKind::NotADirectory,
                    "storage root is not a directory",
                ),
            ));
        }
        Ok(Self { root })
    }
}
impl DataStore for FileStore {
    fn open<'a>(&'a self, uri: &'a DataUri) -> OpenFuture<'a> {
        Box::pin(async move {
            let path = tokio::fs::canonicalize(self.root.join(uri.relative_path()))
                .await
                .map_err(|e| failure(uri.as_str(), "resolve resource", e))?;
            if !path.starts_with(&self.root) {
                return Err(failure(
                    uri.as_str(),
                    "validate resource root",
                    io::Error::new(
                        io::ErrorKind::PermissionDenied,
                        "resource leaves storage root",
                    ),
                ));
            }
            let file = tokio::fs::File::open(path)
                .await
                .map_err(|e| failure(uri.as_str(), "open resource", e))?;
            let metadata = file
                .metadata()
                .await
                .map_err(|e| failure(uri.as_str(), "read resource metadata", e))?;
            if !metadata.is_file() {
                return Err(failure(
                    uri.as_str(),
                    "validate resource type",
                    io::Error::new(io::ErrorKind::InvalidInput, "resource is not a file"),
                ));
            }
            Ok(Box::new(Resource {
                file: Arc::new(file.into_std().await),
                uri: uri.clone(),
                size: metadata.len(),
            }) as Box<dyn RandomAccess>)
        })
    }
}
impl RandomAccess for Resource {
    fn len(&self) -> u64 {
        self.size
    }
    fn read_at(&self, offset: u64, length: usize) -> ReadFuture<'_> {
        Box::pin(async move {
            check_range(&self.uri, self.size, offset, length)?;
            let file = self.file.clone();
            tokio::task::spawn_blocking(move || read_blocking(&file, offset, length))
                .await
                .map_err(|e| failure(self.uri.as_str(), "join range reader", e))?
                .map_err(|e| failure(self.uri.as_str(), "read resource range", e))
        })
    }
}
fn read_blocking(file: &File, offset: u64, length: usize) -> io::Result<Vec<u8>> {
    let mut bytes = vec![0; length];
    let mut count = 0;
    while count < length {
        #[cfg(unix)]
        let n =
            std::os::unix::fs::FileExt::read_at(file, &mut bytes[count..], offset + count as u64)?;
        #[cfg(windows)]
        let n = std::os::windows::fs::FileExt::seek_read(
            file,
            &mut bytes[count..],
            offset + count as u64,
        )?;
        if n == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "resource changed or ended during read",
            ));
        }
        count += n;
    }
    Ok(bytes)
}
#[cfg(test)]
mod tests;
