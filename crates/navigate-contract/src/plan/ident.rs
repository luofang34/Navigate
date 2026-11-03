//! Fixed-capacity identifiers for plans and waypoints.
//!
//! An [`Ident`] holds up to [`MAX_IDENT_BYTES`] bytes of UTF-8 text in
//! place. It does not allocate, and it is `Copy`. Thus a plan, a
//! waypoint, and a validation error that name one are `Copy` too.

use core::fmt;
use core::ops::Deref;

/// The largest identifier length in UTF-8 bytes.
pub const MAX_IDENT_BYTES: usize = 32;

/// A non-empty UTF-8 identifier of at most [`MAX_IDENT_BYTES`] bytes.
///
/// The bytes after the text are always 0. Thus the derived equality,
/// order, and hash agree with those of the text.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Ident {
    bytes: [u8; MAX_IDENT_BYTES],
    len: u8,
}

/// Why a text cannot be an [`Ident`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum IdentError {
    /// The text is empty.
    #[error("an identifier must not be empty")]
    Empty,
    /// The text is longer than [`MAX_IDENT_BYTES`].
    #[error("an identifier of {len} bytes exceeds the {MAX_IDENT_BYTES}-byte limit")]
    TooLong {
        /// The length of the refused text in UTF-8 bytes.
        len: usize,
    },
}

impl Ident {
    /// Builds an identifier from `text`. The function is `const`, so a
    /// constant identifier costs nothing at run time. The result is
    /// `None` when [`Self::try_new`] refuses `text`.
    #[must_use]
    pub const fn try_from_str(text: &str) -> Option<Self> {
        let source = text.as_bytes();
        if source.is_empty() || source.len() > MAX_IDENT_BYTES {
            return None;
        }
        let mut bytes = [0_u8; MAX_IDENT_BYTES];
        let mut index = 0_usize;
        while index < source.len() {
            bytes[index] = source[index];
            index = index.wrapping_add(1);
        }
        Some(Self {
            bytes,
            // The length check above bounds the length at 32.
            len: source.len() as u8,
        })
    }

    /// Builds an identifier from `text`.
    ///
    /// # Errors
    ///
    /// [`IdentError::Empty`] for empty text, and
    /// [`IdentError::TooLong`] for text longer than [`MAX_IDENT_BYTES`]
    /// bytes. The text is never cut.
    pub const fn try_new(text: &str) -> Result<Self, IdentError> {
        match Self::try_from_str(text) {
            Some(ident) => Ok(ident),
            None if text.is_empty() => Err(IdentError::Empty),
            None => Err(IdentError::TooLong { len: text.len() }),
        }
    }

    /// The identifier text.
    #[must_use]
    pub fn as_str(&self) -> &str {
        let used = self.bytes.get(..usize::from(self.len)).unwrap_or(&[]);
        // Every constructor copies a whole `&str`, so `used` is UTF-8;
        // the fallback cannot occur.
        core::str::from_utf8(used).unwrap_or("")
    }
}

impl fmt::Debug for Ident {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("Ident")
            .field(&self.as_str())
            .finish()
    }
}

impl fmt::Display for Ident {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl Deref for Ident {
    type Target = str;

    fn deref(&self) -> &str {
        self.as_str()
    }
}

impl AsRef<str> for Ident {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl TryFrom<&str> for Ident {
    type Error = IdentError;

    fn try_from(text: &str) -> Result<Self, IdentError> {
        Self::try_new(text)
    }
}

impl PartialEq<str> for Ident {
    fn eq(&self, other: &str) -> bool {
        self.as_str() == other
    }
}

impl PartialEq<&str> for Ident {
    fn eq(&self, other: &&str) -> bool {
        self.as_str() == *other
    }
}

#[cfg(test)]
mod tests {
    use std::format;

    use super::{Ident, IdentError, MAX_IDENT_BYTES};

    #[test]
    fn an_ident_keeps_its_text_and_compares_as_text() {
        let ident = Ident::try_new("RW09L").expect("short");
        assert_eq!(ident.as_str(), "RW09L");
        assert_eq!(ident, "RW09L");
        assert_eq!(&*ident, "RW09L");
        assert_eq!(format!("{ident}"), "RW09L");
        assert_eq!(format!("{ident:?}"), "Ident(\"RW09L\")");
        let shorter = Ident::try_new("RW09").expect("short");
        assert!(shorter < ident, "a prefix orders first, as in str");
        assert_ne!(shorter, ident);
    }

    #[test]
    fn an_ident_is_built_at_compile_time() {
        const END: Option<Ident> = Ident::try_from_str("END");
        assert_eq!(END.map(|ident| ident.len()), Some(3));
    }

    #[test]
    fn empty_and_long_text_is_refused_and_never_cut() {
        assert_eq!(Ident::try_new(""), Err(IdentError::Empty));
        let full = "A".repeat(MAX_IDENT_BYTES);
        assert_eq!(Ident::try_new(&full).map(|ident| ident.len()), Ok(32));
        let long = "A".repeat(MAX_IDENT_BYTES + 1);
        assert_eq!(Ident::try_new(&long), Err(IdentError::TooLong { len: 33 }));
        // Multi-byte text counts bytes: eleven 3-byte characters are 33.
        let wide = "東".repeat(11);
        assert_eq!(
            Ident::try_from(wide.as_str()),
            Err(IdentError::TooLong { len: 33 })
        );
        assert_eq!(
            Ident::try_new(&"東".repeat(10)).map(|i| i.chars().count()),
            Ok(10)
        );
    }
}
