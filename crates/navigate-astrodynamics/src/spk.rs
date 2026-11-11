//! Binary NAIF `DAF/SPK` Chebyshev ephemeris parser and evaluator.
//!
//! [`SpkFile`] reads Type 2 and Type 3 Chebyshev segments from a binary `.bsp`
//! slice in memory without file I/O. [`SpkEphemeris`] chains those segments to
//! resolve relative geometric states between any connected pair of NAIF bodies.

mod chebyshev;
mod daf;
mod ephemeris;

pub use chebyshev::{
    ChebyshevError, ChebyshevPva3, eval_chebyshev_pva, eval_type2_record_3d, eval_type3_record_3d,
};
pub use daf::{DafEndian, SpkError};
pub use ephemeris::{SpkDirectedLink, SpkEphemeris};

use std::string::String;
use std::vec::Vec;

use nalgebra::{Matrix3, Vector3};

use daf::{DafContainer, RawDafSummary};

const KM_TO_M: f64 = 1_000.0;
const MARS: i32 = 499;
const MARS_BARYCENTER: i32 = 4;
const J2000_FRAME: i32 = 1;
const ECLIPJ2000_FRAME: i32 = 17;
const SPICE_OBLIQUITY_J2000_RAD: f64 = 84_381.448 * (core::f64::consts::PI / (180.0 * 3_600.0));
type PosVel = (Vector3<f64>, Vector3<f64>);

/// Supported NAIF `SPK` Chebyshev segment representations.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SpkDataType {
    /// SPK Type 2: fixed-interval Chebyshev polynomials for position `(x, y, z)`.
    Type2ChebyshevPosition,
    /// SPK Type 3: fixed-interval Chebyshev polynomials for position and velocity.
    Type3ChebyshevPosVel,
}

/// One validated Chebyshev segment from a binary `DAF/SPK` buffer.
#[derive(Clone, Debug)]
pub struct SpkSegment {
    name: String,
    target: i32,
    center: i32,
    base_frame: i32,
    data_type: SpkDataType,
    start_s: f64,
    end_s: f64,
    init_s: f64,
    intlen_s: f64,
    rsize: usize,
    n_records: usize,
    records: Vec<f64>,
}

impl SpkSegment {
    fn from_raw(container: &DafContainer<'_>, raw: RawDafSummary) -> Result<Self, SpkError> {
        let start_s = raw.dc.first().copied().unwrap_or(0.0);
        let end_s = raw.dc.get(1).copied().unwrap_or(0.0);
        let target = raw.ic.first().copied().unwrap_or(0);
        let center = raw.ic.get(1).copied().unwrap_or(0);
        let base_frame = raw.ic.get(2).copied().unwrap_or(0);
        let dtype_raw = raw.ic.get(3).copied().unwrap_or(0);
        let start_word = raw.ic.get(4).copied().unwrap_or(0);
        let end_word = raw.ic.get(5).copied().unwrap_or(0);

        if base_frame != J2000_FRAME && base_frame != ECLIPJ2000_FRAME {
            return Err(SpkError::UnsupportedBaseFrame(base_frame));
        }
        let data_type = match dtype_raw {
            2 => SpkDataType::Type2ChebyshevPosition,
            3 => SpkDataType::Type3ChebyshevPosVel,
            other => return Err(SpkError::UnsupportedDataType(other)),
        };

        let words = container.read_f64_words(start_word, end_word)?;
        if words.len() < 4 || !start_s.is_finite() || !end_s.is_finite() || end_s < start_s {
            return Err(SpkError::MalformedDirectory(raw.name));
        }
        let footer_start = words.len().wrapping_sub(4);
        let init_s = words.get(footer_start).copied().unwrap_or(0.0);
        let intlen_s = words
            .get(footer_start.wrapping_add(1))
            .copied()
            .unwrap_or(0.0);
        let rsize_f = words
            .get(footer_start.wrapping_add(2))
            .copied()
            .unwrap_or(0.0);
        let n_rec_f = words
            .get(footer_start.wrapping_add(3))
            .copied()
            .unwrap_or(0.0);
        if !init_s.is_finite()
            || !intlen_s.is_finite()
            || intlen_s <= 0.0
            || rsize_f < 5.0
            || n_rec_f < 1.0
        {
            return Err(SpkError::MalformedDirectory(raw.name));
        }
        let rsize = rsize_f as usize;
        let n_records = n_rec_f as usize;
        let expected_payload = rsize.saturating_mul(n_records);
        if expected_payload.saturating_add(4) != words.len() {
            return Err(SpkError::MalformedDirectory(raw.name));
        }
        let mut records = words;
        records.truncate(expected_payload);

        Ok(Self {
            name: raw.name,
            target,
            center,
            base_frame,
            data_type,
            start_s,
            end_s,
            init_s,
            intlen_s,
            rsize,
            n_records,
            records,
        })
    }

    /// Segment summary name (`40` ASCII characters, trimmed).
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Target body or barycentre NAIF ID.
    #[must_use]
    pub const fn target(&self) -> i32 {
        self.target
    }

    /// Center-of-motion body or barycentre NAIF ID.
    #[must_use]
    pub const fn center(&self) -> i32 {
        self.center
    }

    /// Base inertial frame NAIF code (`1` for `J2000`, `17` for `ECLIPJ2000`).
    #[must_use]
    pub const fn base_frame(&self) -> i32 {
        self.base_frame
    }

    /// SPK polynomial representation (`Type 2` or `Type 3`).
    #[must_use]
    pub const fn data_type(&self) -> SpkDataType {
        self.data_type
    }

    /// Segment start and end in TDB seconds past J2000.
    #[must_use]
    pub const fn span_tdb_s(&self) -> (f64, f64) {
        (self.start_s, self.end_s)
    }

    /// Fixed granule interval length `INTLEN` in seconds.
    #[must_use]
    pub const fn granule_length_s(&self) -> f64 {
        self.intlen_s
    }

    /// Initial granule epoch `INIT` in TDB seconds past J2000.
    #[must_use]
    pub const fn init_tdb_s(&self) -> f64 {
        self.init_s
    }

    /// Number of Chebyshev granules in this segment.
    #[must_use]
    pub const fn n_records(&self) -> usize {
        self.n_records
    }

    /// Whether `tdb_s` (TDB seconds past J2000) is inside this segment's interval.
    #[must_use]
    pub fn contains_tdb_s(&self, tdb_s: f64) -> bool {
        tdb_s >= self.start_s && tdb_s <= self.end_s
    }

    /// Evaluate `(position_m, velocity_m_s, acceleration_m_s2)` at split-second `(t_whole_s, t_sub_s)`
    /// in `J2000 / ICRF` axes.
    ///
    /// # Errors
    /// Returns [`SpkError`] if the record layout or Chebyshev evaluation fails.
    pub fn evaluate_pva_split_j2000(
        &self,
        t_whole_s: f64,
        t_sub_s: f64,
    ) -> Result<ChebyshevPva3, SpkError> {
        let rel_s = (t_whole_s - self.init_s) + t_sub_s;
        let raw_idx = (rel_s / self.intlen_s).floor();
        let rec_idx = if raw_idx <= 0.0 {
            0_usize
        } else {
            (raw_idx as usize).min(self.n_records.saturating_sub(1))
        };
        let rec_start = rec_idx.wrapping_mul(self.rsize);
        let rec_end = rec_start.wrapping_add(self.rsize);
        let record = self
            .records
            .get(rec_start..rec_end)
            .ok_or_else(|| SpkError::MalformedDirectory(self.name.clone()))?;

        let (p_arr, v_arr, a_arr) = match self.data_type {
            SpkDataType::Type2ChebyshevPosition => {
                eval_type2_record_3d(record, t_whole_s, t_sub_s, KM_TO_M)
                    .map_err(|_| SpkError::InvalidChebyshevRecord(self.name.clone()))?
            }
            SpkDataType::Type3ChebyshevPosVel => {
                eval_type3_record_3d(record, t_whole_s, t_sub_s, KM_TO_M)
                    .map_err(|_| SpkError::InvalidChebyshevRecord(self.name.clone()))?
            }
        };
        if self.base_frame == ECLIPJ2000_FRAME {
            let rot = eclipj2000_to_j2000();
            let p_rot: [f64; 3] = (rot * Vector3::from(p_arr)).into();
            let v_rot: [f64; 3] = (rot * Vector3::from(v_arr)).into();
            let a_rot: [f64; 3] = (rot * Vector3::from(a_arr)).into();
            return Ok((p_rot, v_rot, a_rot));
        }
        Ok((p_arr, v_arr, a_arr))
    }

    /// Evaluate `(position_m, velocity_m_s)` of `target` relative to `center` in `J2000` axes.
    ///
    /// # Errors
    /// Returns [`SpkError::InvalidChebyshevRecord`] if the record layout or interval radius is invalid.
    pub fn evaluate_j2000(&self, tdb_s: f64) -> Result<(Vector3<f64>, Vector3<f64>), SpkError> {
        let (p, v, _) = self.evaluate_pva_split_j2000(tdb_s, 0.0)?;
        Ok((Vector3::from(p), Vector3::from(v)))
    }

    /// Collects interior Chebyshev granule boundary epochs (in TDB seconds past J2000)
    /// lying strictly inside `(start_s, end_s)`.
    #[must_use]
    pub fn interior_knot_seconds_in(&self, start_s: f64, end_s: f64) -> Vec<f64> {
        if end_s <= start_s {
            return Vec::new();
        }
        let lo_s = start_s.max(self.start_s);
        let hi_s = end_s.min(self.end_s);
        if hi_s <= lo_s {
            return Vec::new();
        }
        let first_k = (((lo_s - self.init_s) / self.intlen_s).floor() as isize).max(1) as usize;
        let last_k = (((hi_s - self.init_s) / self.intlen_s).ceil() as isize)
            .max(0)
            .min(self.n_records as isize) as usize;
        let mut out = Vec::new();
        let mut k = first_k;
        while k < last_k {
            let knot_s = self.init_s + (k as f64) * self.intlen_s;
            if knot_s > start_s && knot_s < end_s && self.contains_tdb_s(knot_s) {
                out.push(knot_s);
            }
            k = k.wrapping_add(1);
        }
        out
    }
}

/// Parsed in-memory `DAF/SPK` file.
#[derive(Clone, Debug)]
pub struct SpkFile {
    segments: Vec<SpkSegment>,
}

impl SpkFile {
    /// Parse a binary `DAF/SPK` byte slice (`Type 2` and `Type 3` Chebyshev segments).
    ///
    /// # Errors
    /// Returns [`SpkError`] if the DAF header, summary chain, or segment directory is invalid.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, SpkError> {
        let container = DafContainer::parse_spk(bytes)?;
        let raw_list = container.summaries()?;
        let mut segments = Vec::with_capacity(raw_list.len());
        for raw in raw_list {
            segments.push(SpkSegment::from_raw(&container, raw)?);
        }
        Ok(Self { segments })
    }

    /// Segments in file summary order.
    #[must_use]
    pub fn segments(&self) -> &[SpkSegment] {
        &self.segments
    }
}

fn eclipj2000_to_j2000() -> Matrix3<f64> {
    let (s, c) = SPICE_OBLIQUITY_J2000_RAD.sin_cos();
    Matrix3::new(1.0, 0.0, 0.0, 0.0, c, -s, 0.0, s, c)
}
