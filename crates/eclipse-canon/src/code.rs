//! The catalogues' one- and two-character codes.

/// A code of one or two ASCII characters, as the catalogue gives it: the
/// eclipse type, such as `T`, `Am` or `H2`, or the code of the eclipses a
/// fortnight before and after, such as `-t` or `p-`. The key of the
/// catalogue pages defines each character.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Code(pub(crate) [u8; 2]);

impl Code {
    /// The code as text, without a trailing space.
    #[must_use]
    pub fn as_str(&self) -> &str {
        let length = if self.0[1] == b' ' { 1 } else { 2 };
        self.0
            .get(..length)
            .and_then(|bytes| core::str::from_utf8(bytes).ok())
            .unwrap_or("")
    }

    /// The first character, which gives the kind of eclipse in a type code.
    #[must_use]
    pub const fn first(&self) -> u8 {
        self.0[0]
    }
}

impl core::fmt::Debug for Code {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        core::fmt::Debug::fmt(self.as_str(), f)
    }
}

impl core::fmt::Display for Code {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.as_str())
    }
}
