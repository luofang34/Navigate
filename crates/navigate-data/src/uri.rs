//! Logical resource names have no platform filesystem syntax.
use crate::DataError;
/// A resource name that a host storage adapter resolves.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DataUri(String);
impl DataUri {
    /// Parse a logical `pilotage://` resource name.
    ///
    /// # Errors
    /// Rejects empty components, relative traversal, and platform path separators.
    pub fn parse(value: impl Into<String>) -> Result<Self, DataError> {
        let value = value.into();
        let valid = value.strip_prefix("pilotage://").is_some_and(|path| {
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
        if !valid {
            return Err(DataError::InvalidUri { uri: value });
        }
        Ok(Self(value))
    }
    /// The complete logical resource name.
    pub fn as_str(&self) -> &str {
        &self.0
    }
    /// The platform-independent components below the storage root.
    pub fn relative_path(&self) -> &str {
        &self.0[11..]
    }
}
#[cfg(test)]
mod tests;
