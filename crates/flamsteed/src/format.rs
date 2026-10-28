//! Errors of a catalogue file.

use core::fmt;

/// Why [`Catalog::from_bytes`](crate::Catalog::from_bytes) rejects a file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum FormatError {
    /// The file is shorter than its 32-byte header.
    TooShort {
        /// The length of the file in bytes.
        length: usize,
    },
    /// The file does not start with the magic `FLAMSTED`.
    Magic,
    /// The format version or the record size is not one that this release
    /// reads. Format version 1 has 34-byte records and version 2 has 72-byte
    /// records.
    Version {
        /// The format version in the header.
        version: u16,
        /// The record size in the header.
        record_bytes: u16,
    },
    /// The source does not match the format version. Version 1 needs source
    /// 1, the Bright Star Catalogue; version 2 needs source 3, HR identifiers
    /// with the astrometry that each record's provenance names.
    Source {
        /// The source in the header.
        source: u32,
    },
    /// The header count and the length of the file do not agree.
    Length {
        /// The record count in the header.
        count: usize,
        /// The length of the records in bytes.
        record_bytes: usize,
    },
    /// The reference epoch is not a finite number.
    Epoch,
    /// The CRC-32 of the records is not the value in the header.
    Checksum {
        /// The CRC-32 in the header.
        stored: u32,
        /// The CRC-32 of the records.
        computed: u32,
    },
    /// A format version 2 record has a position that is not a unit vector or
    /// a velocity that is not finite.
    Record {
        /// The index of the record.
        index: usize,
    },
    /// A record is brighter than the record before it. The records must be
    /// in magnitude order, brightest first.
    Order {
        /// The index of the record.
        index: usize,
    },
}

impl fmt::Display for FormatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooShort { length } => {
                write!(
                    f,
                    "catalogue file of {length} bytes is shorter than its header"
                )
            }
            Self::Magic => f.write_str("catalogue file does not start with FLAMSTED"),
            Self::Version {
                version,
                record_bytes,
            } => write!(
                f,
                "catalogue format version {version} with {record_bytes}-byte records is not supported"
            ),
            Self::Source { source } => write!(f, "catalogue source {source} is not supported"),
            Self::Length {
                count,
                record_bytes,
            } => write!(
                f,
                "catalogue header gives {count} records but the file has {record_bytes} record bytes"
            ),
            Self::Epoch => f.write_str("catalogue reference epoch is not finite"),
            Self::Checksum { stored, computed } => write!(
                f,
                "catalogue CRC-32 is {computed:#010x}, but the header gives {stored:#010x}"
            ),
            Self::Record { index } => write!(
                f,
                "catalogue record {index} has a position that is not a unit vector or a velocity that is not finite"
            ),
            Self::Order { index } => {
                write!(
                    f,
                    "catalogue record {index} is brighter than the record before it"
                )
            }
        }
    }
}

impl core::error::Error for FormatError {}
