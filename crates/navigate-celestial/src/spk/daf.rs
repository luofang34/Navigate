//! Binary NAIF Double Precision Array File (`DAF/SPK`) container and Chebyshev evaluation.

const RECORD_BYTES: usize = 1024;
const DOUBLES_PER_RECORD: usize = 128;

/// Byte order in a NAIF `DAF` header (`LOCFMT`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DafEndian {
    /// IEEE-754 little-endian (`LTL-IEEE`).
    Little,
    /// IEEE-754 big-endian (`BIG-IEEE`).
    Big,
}

/// A binary `DAF/SPK` parse or evaluation failure.
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum SpkError {
    /// The byte slice ended before the requested record or word range.
    #[error("SPK file ends at {len} bytes, and {needed} bytes are necessary")]
    Truncated {
        /// Byte slice length.
        len: usize,
        /// Minimum necessary byte length.
        needed: usize,
    },
    /// The 8-byte header identifier word is not `DAF/SPK `.
    #[error("SPK header identifier is `{found}`, not `DAF/SPK `")]
    UnexpectedIdWord {
        /// Found 8-byte ASCII identifier.
        found: String,
    },
    /// The binary floating-point format is not IEEE-754 little-endian or big-endian.
    #[error("unsupported DAF binary format `{0}`")]
    UnsupportedBinaryFormat(String),
    /// The summary word counts do not match `ND = 2, NI = 6`.
    #[error("invalid SPK summary counts ND={nd}, NI={ni}")]
    InvalidSummaryFormat {
        /// Double-precision summary count.
        nd: i32,
        /// Integer summary count.
        ni: i32,
    },
    /// The summary record chain points outside the file or loops.
    #[error("corrupt DAF summary record chain at record {record}")]
    CorruptSummaryChain {
        /// 1-based record index.
        record: usize,
    },
    /// A segment double-word range `[start_word, end_word]` is invalid.
    #[error("invalid DAF word range {start_word}..={end_word}")]
    InvalidAddressRange {
        /// 1-based start word index.
        start_word: i32,
        /// 1-based inclusive end word index.
        end_word: i32,
    },
    /// The SPK segment data type is not Type 2 or Type 3.
    #[error("unsupported SPK data type {0}")]
    UnsupportedDataType(i32),
    /// The segment inertial frame is not `J2000 (1)` or `ECLIPJ2000 (17)`.
    #[error("unsupported SPK segment base frame {0}")]
    UnsupportedBaseFrame(i32),
    /// The Chebyshev directory footer `(INIT, INTLEN, RSIZE, N)` is malformed.
    #[error("malformed Chebyshev directory in segment `{0}`")]
    MalformedDirectory(String),
    /// A Chebyshev record has an invalid radius or coefficient count.
    #[error("invalid Chebyshev record in segment `{0}`")]
    InvalidChebyshevRecord(String),
    /// No segment chain connects `target` to `center`.
    #[error("no SPK segment chain connects NAIF body {target} to {center}")]
    UnconnectedBodies {
        /// Requested target NAIF ID.
        target: i32,
        /// Requested center NAIF ID.
        center: i32,
    },
    /// No loaded segment covers `target -> center` at `epoch_tdb_s`.
    #[error("epoch {epoch_tdb_s} s TDB is outside SPK coverage for {target} relative to {center}")]
    OutOfCoverage {
        /// Target NAIF ID.
        target: i32,
        /// Center NAIF ID.
        center: i32,
        /// Requested TDB seconds past J2000.
        epoch_tdb_s: f64,
    },
    /// The observer frame orientation is not supported.
    #[error("unsupported observer frame with origin {center} and orientation {orientation}")]
    UnsupportedObserverFrame {
        /// Frame origin NAIF ID.
        center: i32,
        /// Frame orientation NAIF ID.
        orientation: i32,
    },
}

#[derive(Clone, Debug)]
pub(super) struct RawDafSummary {
    pub name: String,
    pub dc: Vec<f64>,
    pub ic: Vec<i32>,
}

#[derive(Clone, Debug)]
pub(super) struct DafContainer<'a> {
    bytes: &'a [u8],
    endian: DafEndian,
    nd: usize,
    ni: usize,
    fward: usize,
}

impl<'a> DafContainer<'a> {
    pub(super) fn parse_spk(bytes: &'a [u8]) -> Result<Self, SpkError> {
        if bytes.len() < RECORD_BYTES {
            return Err(SpkError::Truncated {
                len: bytes.len(),
                needed: RECORD_BYTES,
            });
        }
        let id_bytes = bytes.get(0..8).ok_or(SpkError::Truncated {
            len: bytes.len(),
            needed: 8,
        })?;
        let found_id = String::from_utf8_lossy(id_bytes).into_owned();
        if found_id != "DAF/SPK " {
            return Err(SpkError::UnexpectedIdWord { found: found_id });
        }
        let endian = detect_endian(bytes)?;
        let nd = read_i32(bytes, 8, endian)?;
        let ni = read_i32(bytes, 12, endian)?;
        if nd != 2 || ni != 6 {
            return Err(SpkError::InvalidSummaryFormat { nd, ni });
        }
        let fward_i32 = read_i32(bytes, 76, endian)?;
        if fward_i32 <= 0 {
            return Err(SpkError::CorruptSummaryChain { record: 0 });
        }
        Ok(Self {
            bytes,
            endian,
            nd: 2,
            ni: 6,
            fward: fward_i32 as usize,
        })
    }

    pub(super) fn summaries(&self) -> Result<Vec<RawDafSummary>, SpkError> {
        let ss_words = self.nd.wrapping_add(self.ni.wrapping_add(1) / 2);
        let entry_bytes = ss_words.wrapping_mul(8);
        let max_records = self.bytes.len() / RECORD_BYTES;
        let mut out = Vec::new();
        let mut rec = self.fward;
        let mut visited = 0_usize;

        while rec != 0 {
            visited = visited.wrapping_add(1);
            if rec > max_records || visited > max_records {
                return Err(SpkError::CorruptSummaryChain { record: rec });
            }
            let sum_off = rec.wrapping_sub(1).wrapping_mul(RECORD_BYTES);
            let name_off = rec.wrapping_mul(RECORD_BYTES);
            let next_f = read_f64(self.bytes, sum_off, self.endian)?;
            let nsum_f = read_f64(self.bytes, sum_off.wrapping_add(16), self.endian)?;
            if !next_f.is_finite() || next_f < 0.0 || !nsum_f.is_finite() || nsum_f < 0.0 {
                return Err(SpkError::CorruptSummaryChain { record: rec });
            }
            let nsum = nsum_f as usize;
            let mut idx = 0_usize;
            while idx < nsum {
                let e_off = sum_off
                    .wrapping_add(24)
                    .wrapping_add(idx.wrapping_mul(entry_bytes));
                let n_off = name_off.wrapping_add(idx.wrapping_mul(entry_bytes));
                out.push(self.parse_one_summary(e_off, n_off, entry_bytes)?);
                idx = idx.wrapping_add(1);
            }
            rec = next_f as usize;
        }
        Ok(out)
    }

    fn parse_one_summary(
        &self,
        e_off: usize,
        n_off: usize,
        entry_bytes: usize,
    ) -> Result<RawDafSummary, SpkError> {
        let mut dc = Vec::with_capacity(self.nd);
        let mut d_idx = 0_usize;
        while d_idx < self.nd {
            dc.push(read_f64(
                self.bytes,
                e_off.wrapping_add(d_idx.wrapping_mul(8)),
                self.endian,
            )?);
            d_idx = d_idx.wrapping_add(1);
        }
        let ic_base = e_off.wrapping_add(self.nd.wrapping_mul(8));
        let mut ic = Vec::with_capacity(self.ni);
        let mut i_idx = 0_usize;
        while i_idx < self.ni {
            ic.push(read_i32(
                self.bytes,
                ic_base.wrapping_add(i_idx.wrapping_mul(4)),
                self.endian,
            )?);
            i_idx = i_idx.wrapping_add(1);
        }
        let name = self
            .bytes
            .get(n_off..n_off.wrapping_add(entry_bytes))
            .map(|s| String::from_utf8_lossy(s).trim().to_owned())
            .unwrap_or_default();
        Ok(RawDafSummary { name, dc, ic })
    }

    pub(super) fn read_f64_words(
        &self,
        start_word: i32,
        end_word: i32,
    ) -> Result<Vec<f64>, SpkError> {
        if start_word <= 0 || end_word < start_word {
            return Err(SpkError::InvalidAddressRange {
                start_word,
                end_word,
            });
        }
        let start_idx = (start_word as usize).wrapping_sub(1);
        let count = (end_word as usize).wrapping_sub(start_idx);
        let byte_start = start_idx.wrapping_mul(8);
        let byte_end = byte_start.wrapping_add(count.wrapping_mul(8));
        if byte_end > self.bytes.len() || count > DOUBLES_PER_RECORD.saturating_mul(2_000_000) {
            return Err(SpkError::Truncated {
                len: self.bytes.len(),
                needed: byte_end,
            });
        }
        let mut out = Vec::with_capacity(count);
        let mut k = 0_usize;
        while k < count {
            out.push(read_f64(
                self.bytes,
                byte_start.wrapping_add(k.wrapping_mul(8)),
                self.endian,
            )?);
            k = k.wrapping_add(1);
        }
        Ok(out)
    }
}

fn detect_endian(bytes: &[u8]) -> Result<DafEndian, SpkError> {
    let fmt_slice = bytes.get(88..96).ok_or(SpkError::Truncated {
        len: bytes.len(),
        needed: 96,
    })?;
    let fmt_str = String::from_utf8_lossy(fmt_slice);
    let trimmed = fmt_str.trim();
    if trimmed.eq_ignore_ascii_case("LTL-IEEE") {
        return Ok(DafEndian::Little);
    }
    if trimmed.eq_ignore_ascii_case("BIG-IEEE") {
        return Ok(DafEndian::Big);
    }
    let nd_le = read_i32(bytes, 8, DafEndian::Little)?;
    if (1..=124).contains(&nd_le) {
        return Ok(DafEndian::Little);
    }
    let nd_be = read_i32(bytes, 8, DafEndian::Big)?;
    if (1..=124).contains(&nd_be) {
        return Ok(DafEndian::Big);
    }
    Err(SpkError::UnsupportedBinaryFormat(trimmed.to_owned()))
}

fn read_i32(bytes: &[u8], offset: usize, endian: DafEndian) -> Result<i32, SpkError> {
    let end = offset.wrapping_add(4);
    let slice = bytes.get(offset..end).ok_or(SpkError::Truncated {
        len: bytes.len(),
        needed: end,
    })?;
    let mut arr = [0_u8; 4];
    arr.copy_from_slice(slice);
    Ok(match endian {
        DafEndian::Little => i32::from_le_bytes(arr),
        DafEndian::Big => i32::from_be_bytes(arr),
    })
}

fn read_f64(bytes: &[u8], offset: usize, endian: DafEndian) -> Result<f64, SpkError> {
    let end = offset.wrapping_add(8);
    let slice = bytes.get(offset..end).ok_or(SpkError::Truncated {
        len: bytes.len(),
        needed: end,
    })?;
    let mut arr = [0_u8; 8];
    arr.copy_from_slice(slice);
    Ok(match endian {
        DafEndian::Little => f64::from_le_bytes(arr),
        DafEndian::Big => f64::from_be_bytes(arr),
    })
}

pub(super) fn eval_chebyshev_pv(
    coeffs: &[f64],
    mid_s: f64,
    radius_s: f64,
    t_s: f64,
) -> Option<(f64, f64)> {
    if !radius_s.is_finite() || radius_s <= 0.0 {
        return None;
    }
    let &c0 = coeffs.first()?;
    let tau_raw = (t_s - mid_s) / radius_s;
    if !tau_raw.is_finite() || tau_raw.abs() > 1.0 + 1e-9 {
        return None;
    }
    let tau = tau_raw.clamp(-1.0, 1.0);
    if coeffs.len() == 1 {
        return Some((c0, 0.0));
    }
    let two_tau = 2.0 * tau;
    let mut t_prev2 = 1.0_f64;
    let mut t_prev1 = tau;
    let mut dt_prev2 = 0.0_f64;
    let mut dt_prev1 = 1.0_f64;

    let c1 = coeffs.get(1).copied().unwrap_or(0.0);
    let mut sum_p = c1 * t_prev1 + c0;
    let mut sum_dp = c1;

    let mut k = 2_usize;
    while let Some(&ck) = coeffs.get(k) {
        let tk = two_tau * t_prev1 - t_prev2;
        let dtk = two_tau * dt_prev1 + 2.0 * t_prev1 - dt_prev2;
        sum_p += ck * tk;
        sum_dp += ck * dtk;
        t_prev2 = t_prev1;
        t_prev1 = tk;
        dt_prev2 = dt_prev1;
        dt_prev1 = dtk;
        k = k.wrapping_add(1);
    }
    Some((sum_p, sum_dp / radius_s))
}
