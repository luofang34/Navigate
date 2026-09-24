//! Logical resource names have no platform filesystem syntax.
use crate::{DataError, StorageClass};
/// A resource name that a host storage adapter resolves.
///
/// The form is `scheme://path`. The host names the scheme and decides what
/// storage root it selects. This crate gives no meaning to a scheme.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DataUri {
    value: String,
    path_start: usize,
}
impl DataUri {
    /// Parse a logical `scheme://path` resource name.
    ///
    /// # Errors
    /// Rejects a missing or invalid scheme, the `file` scheme, empty
    /// components, relative traversal, and platform path separators.
    pub fn parse(value: impl Into<String>) -> Result<Self, DataError> {
        let value = value.into();
        let parts = value.split_once("://");
        let valid_scheme = parts.is_some_and(|(scheme, _)| {
            scheme != "file"
                && scheme.starts_with(|c: char| c.is_ascii_lowercase())
                && scheme.chars().all(|c| {
                    c.is_ascii_lowercase() || c.is_ascii_digit() || matches!(c, '+' | '-' | '.')
                })
        });
        let valid = valid_scheme
            && parts.is_some_and(|(_, path)| {
                !path.is_empty()
                    && path.split('/').all(|part| {
                        !part.is_empty()
                            && part != "."
                            && part != ".."
                            && part
                                .chars()
                                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
                    })
            });
        let Some((scheme, _)) = parts.filter(|_| valid) else {
            return Err(DataError::InvalidUri { uri: value });
        };
        let path_start = scheme.len() + "://".len();
        Ok(Self { value, path_start })
    }
    /// Name a resource by storage class and logical path.
    ///
    /// # Errors
    /// Rejects the same unsafe paths as [`Self::parse`].
    pub fn in_class(class: StorageClass, path: &str) -> Result<Self, DataError> {
        Self::parse(format!("{}://{path}", class.scheme()))
    }
    /// The storage class that the scheme names, or `None` for a host scheme.
    pub fn storage_class(&self) -> Option<StorageClass> {
        StorageClass::from_scheme(self.scheme())
    }
    /// The complete logical resource name.
    pub fn as_str(&self) -> &str {
        &self.value
    }
    /// The host-defined scheme that selects a storage root.
    pub fn scheme(&self) -> &str {
        &self.value[..self.path_start - "://".len()]
    }
    /// The platform-independent components below the storage root.
    pub fn relative_path(&self) -> &str {
        &self.value[self.path_start..]
    }
}
#[cfg(test)]
mod tests;
