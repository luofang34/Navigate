//! Pure-Rust reader and writer for NumPy `.npy` and compressed `.npz` array archives.

use crate::SegmentError;
use std::{
    collections::BTreeMap,
    fs::File,
    io::{Cursor, Read, Write},
    path::Path,
};
use zip::{CompressionMethod, ZipArchive, ZipWriter, write::SimpleFileOptions};

/// Element data types supported by [`NpyArray`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NpyDType {
    /// Boolean byte (`|b1`).
    Bool,
    /// Unsigned 8-bit integer (`|u1`).
    U8,
    /// Little-endian signed 32-bit integer (`<i4`).
    I32,
    /// Little-endian signed 64-bit integer (`<i8`).
    I64,
    /// Little-endian 32-bit IEEE float (`<f4`).
    F32,
    /// Little-endian 64-bit IEEE float (`<f8`).
    F64,
}

impl NpyDType {
    fn descr(self) -> &'static str {
        match self {
            Self::Bool => "|b1",
            Self::U8 => "|u1",
            Self::I32 => "<i4",
            Self::I64 => "<i8",
            Self::F32 => "<f4",
            Self::F64 => "<f8",
        }
    }

    fn from_descr(descr: &str) -> Result<Self, SegmentError> {
        match descr {
            "|b1" | "b1" | "?" => Ok(Self::Bool),
            "|u1" | "<u1" | ">u1" | "u1" => Ok(Self::U8),
            "<i4" | "|i4" => Ok(Self::I32),
            "<i8" | "|i8" => Ok(Self::I64),
            "<f4" => Ok(Self::F32),
            "<f8" => Ok(Self::F64),
            other => Err(SegmentError::Invalid(format!(
                "unsupported .npy dtype descriptor '{other}'"
            ))),
        }
    }

    fn elem_bytes(self) -> usize {
        match self {
            Self::Bool | Self::U8 => 1,
            Self::I32 | Self::F32 => 4,
            Self::I64 | Self::F64 => 8,
        }
    }
}

/// C-contiguous multidimensional NumPy array stored in host memory.
#[derive(Clone, Debug, PartialEq)]
pub struct NpyArray {
    /// Array dimensions in row-major order.
    pub shape: Vec<usize>,
    /// Element data type.
    pub dtype: NpyDType,
    /// Raw little-endian element bytes.
    pub bytes: Vec<u8>,
}

impl NpyArray {
    /// Construct a `u8` array with `shape`.
    ///
    /// # Errors
    ///
    /// Returns [`SegmentError::Shape`] when `values.len()` does not match `shape`.
    pub fn from_u8(shape: Vec<usize>, values: Vec<u8>) -> Result<Self, SegmentError> {
        check_len("u8", &shape, values.len())?;
        Ok(Self {
            shape,
            dtype: NpyDType::U8,
            bytes: values,
        })
    }

    /// Construct a `bool` array with `shape`.
    ///
    /// # Errors
    ///
    /// Returns [`SegmentError::Shape`] when `values.len()` does not match `shape`.
    pub fn from_bool(shape: Vec<usize>, values: &[bool]) -> Result<Self, SegmentError> {
        check_len("bool", &shape, values.len())?;
        Ok(Self {
            shape,
            dtype: NpyDType::Bool,
            bytes: values.iter().map(|&b| u8::from(b)).collect(),
        })
    }

    /// Construct an `f32` array with `shape`.
    ///
    /// # Errors
    ///
    /// Returns [`SegmentError::Shape`] when `values.len()` does not match `shape`.
    pub fn from_f32(shape: Vec<usize>, values: &[f32]) -> Result<Self, SegmentError> {
        check_len("f32", &shape, values.len())?;
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for &v in values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        Ok(Self {
            shape,
            dtype: NpyDType::F32,
            bytes,
        })
    }

    /// Construct an `f64` array with `shape`.
    ///
    /// # Errors
    ///
    /// Returns [`SegmentError::Shape`] when `values.len()` does not match `shape`.
    pub fn from_f64(shape: Vec<usize>, values: &[f64]) -> Result<Self, SegmentError> {
        check_len("f64", &shape, values.len())?;
        let mut bytes = Vec::with_capacity(values.len() * 8);
        for &v in values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        Ok(Self {
            shape,
            dtype: NpyDType::F64,
            bytes,
        })
    }

    /// Construct an `i64` array with `shape`.
    ///
    /// # Errors
    ///
    /// Returns [`SegmentError::Shape`] when `values.len()` does not match `shape`.
    pub fn from_i64(shape: Vec<usize>, values: &[i64]) -> Result<Self, SegmentError> {
        check_len("i64", &shape, values.len())?;
        let mut bytes = Vec::with_capacity(values.len() * 8);
        for &v in values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        Ok(Self {
            shape,
            dtype: NpyDType::I64,
            bytes,
        })
    }

    /// Decode values as `Vec<u8>` when `dtype` is `U8` or `Bool`.
    ///
    /// # Errors
    ///
    /// Returns [`SegmentError::Invalid`] when `dtype` is not byte-valued.
    pub fn to_u8(&self) -> Result<Vec<u8>, SegmentError> {
        match self.dtype {
            NpyDType::U8 | NpyDType::Bool => Ok(self.bytes.clone()),
            other => Err(SegmentError::Invalid(format!(
                "cannot decode {other:?} array as u8"
            ))),
        }
    }

    /// Decode numeric values as `Vec<f32>`.
    pub fn to_f32(&self) -> Vec<f32> {
        self.to_f64().into_iter().map(|v| v as f32).collect()
    }

    /// Decode numeric values as `Vec<f64>`.
    pub fn to_f64(&self) -> Vec<f64> {
        match self.dtype {
            NpyDType::Bool => self
                .bytes
                .iter()
                .map(|&b| f64::from(u8::from(b != 0)))
                .collect(),
            NpyDType::U8 => self.bytes.iter().map(|&b| f64::from(b)).collect(),
            NpyDType::I32 => self
                .bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f64::from(i32::from_le_bytes(*c)))
                .collect(),
            NpyDType::I64 => self
                .bytes
                .as_chunks::<8>()
                .0
                .iter()
                .map(|c| i64::from_le_bytes(*c) as f64)
                .collect(),
            NpyDType::F32 => self
                .bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f64::from(f32::from_le_bytes(*c)))
                .collect(),
            NpyDType::F64 => self
                .bytes
                .as_chunks::<8>()
                .0
                .iter()
                .map(|c| f64::from_le_bytes(*c))
                .collect(),
        }
    }

    /// Serialize this array into `.npy` v1.0 bytes.
    pub fn to_npy_bytes(&self) -> Vec<u8> {
        let shape_str = match self.shape.as_slice() {
            [] => "()".to_owned(),
            [one] => format!("({one},)"),
            many => format!(
                "({})",
                many.iter()
                    .map(usize::to_string)
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        };
        let mut dict = format!(
            "{{'descr': '{}', 'fortran_order': False, 'shape': {}, }}",
            self.dtype.descr(),
            shape_str
        );
        let prefix_len = 10;
        let rem = (prefix_len + dict.len() + 1) % 64;
        if rem != 0 {
            dict.push_str(&" ".repeat(64 - rem));
        }
        dict.push('\n');
        let header_len = dict.len() as u16;
        let mut out = Vec::with_capacity(prefix_len + dict.len() + self.bytes.len());
        out.extend_from_slice(b"\x93NUMPY\x01\x00");
        out.extend_from_slice(&header_len.to_le_bytes());
        out.extend_from_slice(dict.as_bytes());
        out.extend_from_slice(&self.bytes);
        out
    }

    /// Parse a `.npy` buffer into an [`NpyArray`].
    ///
    /// # Errors
    ///
    /// Returns [`SegmentError`] when the `.npy` header or payload is malformed.
    pub fn from_npy_bytes(raw: &[u8]) -> Result<Self, SegmentError> {
        if raw.len() < 10 || &raw[..6] != b"\x93NUMPY" {
            return Err(SegmentError::Invalid("missing .npy magic header".into()));
        }
        let major = raw[6];
        let (header_len, offset) = match major {
            1 => {
                let len = u16::from_le_bytes([raw[8], raw[9]]) as usize;
                (len, 10)
            }
            2 | 3 => {
                if raw.len() < 12 {
                    return Err(SegmentError::Invalid("truncated .npy v2 header".into()));
                }
                let len = u32::from_le_bytes([raw[8], raw[9], raw[10], raw[11]]) as usize;
                (len, 12)
            }
            other => {
                return Err(SegmentError::Invalid(format!(
                    "unsupported .npy major version {other}"
                )));
            }
        };
        let data_start = offset + header_len;
        if raw.len() < data_start {
            return Err(SegmentError::Invalid("truncated .npy dictionary".into()));
        }
        let header = std::str::from_utf8(&raw[offset..data_start])
            .map_err(|_| SegmentError::Invalid("non-UTF8 .npy header".into()))?;
        if extract_field(header, "fortran_order").is_some_and(|v| v.starts_with("True")) {
            return Err(SegmentError::Invalid(
                "Fortran-order .npy arrays are not supported".into(),
            ));
        }
        let descr_raw = extract_field(header, "descr")
            .ok_or_else(|| SegmentError::Invalid("missing descr in .npy header".into()))?;
        let descr = descr_raw.trim_matches(|c| c == '\'' || c == '"');
        let dtype = NpyDType::from_descr(descr)?;
        let shape = parse_shape(header)?;
        let count: usize = shape.iter().product();
        let expected_bytes = count * dtype.elem_bytes();
        let payload = &raw[data_start..];
        if payload.len() < expected_bytes {
            return Err(SegmentError::Invalid(format!(
                "truncated .npy payload: expected {expected_bytes} bytes, got {}",
                payload.len()
            )));
        }
        Ok(Self {
            shape,
            dtype,
            bytes: payload[..expected_bytes].to_vec(),
        })
    }
}

fn check_len(kind: &str, shape: &[usize], actual: usize) -> Result<(), SegmentError> {
    let expected: usize = shape.iter().product();
    if expected != actual {
        return Err(SegmentError::Shape {
            name: kind.into(),
            expected: shape.to_vec(),
            actual: vec![actual],
        });
    }
    Ok(())
}

fn extract_field<'a>(header: &'a str, key: &str) -> Option<&'a str> {
    for quote in ['\'', '"'] {
        let needle = format!("{quote}{key}{quote}");
        if let Some(pos) = header.find(&needle) {
            let after = &header[pos + needle.len()..];
            let colon = after.find(':')?;
            let rest = after[colon + 1..].trim_start();
            if rest.starts_with('(') {
                let end = rest.find(')')?;
                return Some(&rest[..=end]);
            }
            let end = rest.find([',', '}'])?;
            return Some(rest[..end].trim());
        }
    }
    None
}

fn parse_shape(header: &str) -> Result<Vec<usize>, SegmentError> {
    let raw = extract_field(header, "shape")
        .ok_or_else(|| SegmentError::Invalid("missing shape in .npy header".into()))?;
    let inner = raw
        .strip_prefix('(')
        .and_then(|s| s.strip_suffix(')'))
        .ok_or_else(|| SegmentError::Invalid(format!("invalid .npy shape '{raw}'")))?;
    let mut dims = Vec::new();
    for part in inner.split(',') {
        let trimmed = part.trim();
        if trimmed.is_empty() {
            continue;
        }
        let d = trimmed
            .parse::<usize>()
            .map_err(|_| SegmentError::Invalid(format!("invalid dimension '{trimmed}'")))?;
        dims.push(d);
    }
    Ok(dims)
}

/// Read a single `.npy` file from `path`.
///
/// # Errors
///
/// Returns [`SegmentError`] when reading or decoding fails.
pub fn read_npy(path: &Path) -> Result<NpyArray, SegmentError> {
    let bytes = std::fs::read(path).map_err(|source| SegmentError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    NpyArray::from_npy_bytes(&bytes)
}

/// Write a single `.npy` file to `path`.
///
/// # Errors
///
/// Returns [`SegmentError`] when writing fails.
pub fn write_npy(path: &Path, array: &NpyArray) -> Result<(), SegmentError> {
    std::fs::write(path, array.to_npy_bytes()).map_err(|source| SegmentError::Io {
        path: path.to_path_buf(),
        source,
    })
}

/// Read all arrays from a `.npz` archive at `path`.
///
/// Keys are returned without the trailing `.npy` suffix.
///
/// # Errors
///
/// Returns [`SegmentError`] when opening or decoding any member fails.
pub fn read_npz(path: &Path) -> Result<BTreeMap<String, NpyArray>, SegmentError> {
    let file = File::open(path).map_err(|source| SegmentError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mut archive = ZipArchive::new(file)
        .map_err(|e| SegmentError::Archive(format!("{}: {e}", path.display())))?;
    let mut out = BTreeMap::new();
    for idx in 0..archive.len() {
        let mut entry = archive
            .by_index(idx)
            .map_err(|e| SegmentError::Archive(e.to_string()))?;
        let name = entry
            .name()
            .map_err(|e| SegmentError::Archive(e.to_string()))?
            .trim_end_matches(".npy")
            .to_owned();
        let mut buf = Vec::new();
        entry
            .read_to_end(&mut buf)
            .map_err(|source| SegmentError::Io {
                path: path.to_path_buf(),
                source,
            })?;
        out.insert(name, NpyArray::from_npy_bytes(&buf)?);
    }
    Ok(out)
}

/// Write named [`NpyArray`] entries into a deflate-compressed `.npz` file at `path`.
///
/// # Errors
///
/// Returns [`SegmentError`] when archive creation or disk write fails.
pub fn write_npz(path: &Path, arrays: &BTreeMap<String, NpyArray>) -> Result<(), SegmentError> {
    let mut cursor = Cursor::new(Vec::new());
    {
        let mut writer = ZipWriter::new(&mut cursor);
        let options = SimpleFileOptions::default().compression_method(CompressionMethod::Deflated);
        for (name, array) in arrays {
            let entry_name = if name.ends_with(".npy") {
                name.clone()
            } else {
                format!("{name}.npy")
            };
            writer
                .start_file(entry_name, options)
                .map_err(|e| SegmentError::Archive(e.to_string()))?;
            writer
                .write_all(&array.to_npy_bytes())
                .map_err(|source| SegmentError::Io {
                    path: path.to_path_buf(),
                    source,
                })?;
        }
        writer
            .finish()
            .map_err(|e| SegmentError::Archive(e.to_string()))?;
    }
    std::fs::write(path, cursor.into_inner()).map_err(|source| SegmentError::Io {
        path: path.to_path_buf(),
        source,
    })
}
