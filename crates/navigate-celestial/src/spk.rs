//! Binary NAIF `DAF/SPK` Chebyshev ephemeris provider.
//!
//! [`SpkFile`] reads Type 2 and Type 3 Chebyshev segments from a binary `.bsp`
//! slice in memory without file I/O. [`SpkEphemeris`] chains those segments to
//! the solar-system barycentre (`SSB`, NAIF ID `0`) and implements
//! [`Ephemeris`].

mod daf;

pub use daf::{DafEndian, SpkError};

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use nalgebra::{Matrix3, Vector3};

use crate::{Ephemeris, Epoch, FrameId, State, TimeScale, earth_rotation};
use daf::{DafContainer, RawDafSummary, eval_chebyshev_pv};

const KM_TO_M: f64 = 1_000.0;
const SSB: i32 = 0;
const EARTH: i32 = 399;
const MARS: i32 = 499;
const MARS_BARYCENTER: i32 = 4;
const J2000_FRAME: i32 = 1;
const ECLIPJ2000_FRAME: i32 = 17;
const ITRF93_FRAME: i32 = FrameId::EARTH_ITRF93.orientation_naif_id;
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
        let footer_start = words.len() - 4;
        let init_s = words.get(footer_start).copied().unwrap_or(0.0);
        let intlen_s = words.get(footer_start + 1).copied().unwrap_or(0.0);
        let rsize_f = words.get(footer_start + 2).copied().unwrap_or(0.0);
        let n_rec_f = words.get(footer_start + 3).copied().unwrap_or(0.0);
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

    /// Segment start and end in TDB seconds past J2000.
    #[must_use]
    pub const fn span_tdb_s(&self) -> (f64, f64) {
        (self.start_s, self.end_s)
    }

    /// Whether `tdb_s` (TDB seconds past J2000) is inside this segment's interval.
    #[must_use]
    pub fn contains_tdb_s(&self, tdb_s: f64) -> bool {
        tdb_s >= self.start_s && tdb_s <= self.end_s
    }

    /// Evaluate `(position_m, velocity_m_s)` of `target` relative to `center` in `J2000` axes.
    ///
    /// # Errors
    /// Returns [`SpkError::InvalidChebyshevRecord`] if the record layout or interval radius is invalid.
    pub fn evaluate_j2000(&self, tdb_s: f64) -> Result<(Vector3<f64>, Vector3<f64>), SpkError> {
        let rel_s = tdb_s - self.init_s;
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

        let (mut pos, mut vel) = match self.data_type {
            SpkDataType::Type2ChebyshevPosition => eval_type2(record, tdb_s)
                .ok_or_else(|| SpkError::InvalidChebyshevRecord(self.name.clone()))?,
            SpkDataType::Type3ChebyshevPosVel => eval_type3(record, tdb_s)
                .ok_or_else(|| SpkError::InvalidChebyshevRecord(self.name.clone()))?,
        };
        if self.base_frame == ECLIPJ2000_FRAME {
            let rot = eclipj2000_to_j2000();
            pos = rot * pos;
            vel = rot * vel;
        }
        Ok((pos, vel))
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

#[derive(Clone, Copy, Debug)]
struct DirectedLink {
    slot: usize,
    forward: bool,
}

/// Multi-file binary `DAF/SPK` ephemeris provider implementing [`Ephemeris`].
///
/// Later files and later segments take precedence when coverage intervals overlap.
/// If a kernel supplies the Mars system barycentre (`4`) without Mars body centre (`499`),
/// requests for `499` resolve through `4` (offset `< 200 m`).
#[derive(Clone, Debug)]
pub struct SpkEphemeris {
    edge_lists: Vec<Vec<(usize, bool)>>,
    segments: Vec<SpkSegment>,
    adjacency: BTreeMap<i32, Vec<(i32, DirectedLink)>>,
    known_bodies: BTreeSet<i32>,
}

impl SpkEphemeris {
    /// Build an ephemeris from parsed [`SpkFile`] values in load order.
    #[must_use]
    pub fn from_files(files: Vec<SpkFile>) -> Self {
        let mut segments = Vec::new();
        for f in files {
            segments.extend_from_slice(&f.segments);
        }
        let mut pair_map: BTreeMap<(i32, i32), Vec<(usize, bool)>> = BTreeMap::new();
        let mut known_bodies = BTreeSet::new();
        for (idx, seg) in segments.iter().enumerate() {
            let (t, c) = (seg.target(), seg.center());
            known_bodies.insert(t);
            known_bodies.insert(c);
            let (lo, hi, is_lo_to_hi) = if t <= c { (t, c, true) } else { (c, t, false) };
            pair_map
                .entry((lo, hi))
                .or_default()
                .push((idx, is_lo_to_hi));
        }
        let mut edge_lists = Vec::with_capacity(pair_map.len());
        let mut adjacency: BTreeMap<i32, Vec<(i32, DirectedLink)>> = BTreeMap::new();
        for ((lo, hi), mut list) in pair_map {
            list.reverse();
            let slot = edge_lists.len();
            edge_lists.push(list);
            adjacency.entry(lo).or_default().push((
                hi,
                DirectedLink {
                    slot,
                    forward: true,
                },
            ));
            adjacency.entry(hi).or_default().push((
                lo,
                DirectedLink {
                    slot,
                    forward: false,
                },
            ));
        }
        Self {
            edge_lists,
            segments,
            adjacency,
            known_bodies,
        }
    }

    /// Parse one or more binary `.bsp` byte slices in precedence order.
    ///
    /// # Errors
    /// Returns [`SpkError`] if any slice is not a valid `DAF/SPK` Type 2 or Type 3 file.
    pub fn from_slices(slices: &[&[u8]]) -> Result<Self, SpkError> {
        let mut files = Vec::with_capacity(slices.len());
        for &raw in slices {
            files.push(SpkFile::from_bytes(raw)?);
        }
        Ok(Self::from_files(files))
    }

    /// Parse a single binary `.bsp` byte slice.
    ///
    /// # Errors
    /// Returns [`SpkError`] if `bytes` is not a valid `DAF/SPK` Type 2 or Type 3 file.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, SpkError> {
        Self::from_slices(&[bytes])
    }

    fn remap_body(&self, id: i32) -> i32 {
        if self.known_bodies.contains(&id) {
            return id;
        }
        if id == MARS && self.known_bodies.contains(&MARS_BARYCENTER) {
            return MARS_BARYCENTER;
        }
        id
    }

    fn resolve_chain(&self, target: i32, center: i32) -> Result<Vec<DirectedLink>, SpkError> {
        if target == center {
            return Ok(Vec::new());
        }
        let mut visited: BTreeMap<i32, Option<(i32, DirectedLink)>> = BTreeMap::new();
        let mut queue = VecDeque::new();
        visited.insert(target, None);
        queue.push_back(target);

        while let Some(curr) = queue.pop_front() {
            if curr == center {
                let mut chain = Vec::new();
                let mut walk = center;
                while let Some(Some((prev, link))) = visited.get(&walk).copied() {
                    chain.push(link);
                    walk = prev;
                }
                chain.reverse();
                return Ok(chain);
            }
            if let Some(neighbors) = self.adjacency.get(&curr) {
                for &(next, link) in neighbors {
                    if let std::collections::btree_map::Entry::Vacant(e) = visited.entry(next) {
                        e.insert(Some((curr, link)));
                        queue.push_back(next);
                    }
                }
            }
        }
        Err(SpkError::UnconnectedBodies { target, center })
    }

    fn eval_link(&self, link: DirectedLink, tdb_s: f64) -> Result<Option<PosVel>, SpkError> {
        let Some(candidates) = self.edge_lists.get(link.slot) else {
            return Ok(None);
        };
        for &(seg_idx, is_lo_to_hi) in candidates {
            let Some(seg) = self.segments.get(seg_idx) else {
                continue;
            };
            if seg.contains_tdb_s(tdb_s) {
                let (pos, vel) = seg.evaluate_j2000(tdb_s)?;
                return Ok(Some(if link.forward == is_lo_to_hi {
                    (pos, vel)
                } else {
                    (-pos, -vel)
                }));
            }
        }
        Ok(None)
    }

    /// Evaluate the geometric state `(position_m, velocity_m_s)` of `target` relative to `center`
    /// in `J2000` axes at `epoch`.
    ///
    /// # Errors
    /// Returns [`SpkError`] if no segment chain connects `target` to `center` or `epoch` is outside coverage.
    pub fn relative_state_j2000(
        &self,
        target: i32,
        center: i32,
        epoch: Epoch,
    ) -> Result<(Vector3<f64>, Vector3<f64>), SpkError> {
        let t_eff = self.remap_body(target);
        let c_eff = self.remap_body(center);
        if t_eff == c_eff {
            return Ok((Vector3::zeros(), Vector3::zeros()));
        }
        let tdb_s = epoch.to_time_scale(TimeScale::TDB).to_tdb_seconds();
        let chain = self.resolve_chain(t_eff, c_eff)?;
        let mut total_p = Vector3::zeros();
        let mut total_v = Vector3::zeros();
        for link in chain {
            let Some((dp, dv)) = self.eval_link(link, tdb_s)? else {
                return Err(SpkError::OutOfCoverage {
                    target,
                    center,
                    epoch_tdb_s: tdb_s,
                });
            };
            total_p += dp;
            total_v += dv;
        }
        Ok((total_p, total_v))
    }
}

impl Ephemeris for SpkEphemeris {
    type Error = SpkError;

    fn barycentric_state(&self, state: State) -> Result<State, Self::Error> {
        let pos = Vector3::from(state.position_m);
        let vel = Vector3::from(state.velocity_m_s);
        let (rel_p, rel_v) = match (state.frame.center_naif_id, state.frame.orientation_naif_id) {
            (_, J2000_FRAME) => (pos, vel),
            (EARTH, ITRF93_FRAME) => {
                let to_j2000 = earth_rotation::j2000_to_earth_fixed(state.epoch).transpose();
                let v_fixed = vel + earth_rotation::rotation_velocity(pos);
                (to_j2000 * pos, to_j2000 * v_fixed)
            }
            (center, orientation) => {
                return Err(SpkError::UnsupportedObserverFrame {
                    center,
                    orientation,
                });
            }
        };
        let (origin_p, origin_v) =
            self.relative_state_j2000(state.frame.center_naif_id, SSB, state.epoch)?;
        Ok(State {
            position_m: (origin_p + rel_p).into(),
            velocity_m_s: (origin_v + rel_v).into(),
            epoch: state.epoch,
            frame: FrameId::SSB_J2000,
        })
    }

    fn body_state(&self, body: i32, epoch: Epoch) -> Result<State, Self::Error> {
        let (pos, vel) = self.relative_state_j2000(body, SSB, epoch)?;
        Ok(State {
            position_m: pos.into(),
            velocity_m_s: vel.into(),
            epoch,
            frame: FrameId::SSB_J2000,
        })
    }
}

fn eval_type2(record: &[f64], t_s: f64) -> Option<(Vector3<f64>, Vector3<f64>)> {
    if record.len() < 5 || !(record.len() - 2).is_multiple_of(3) {
        return None;
    }
    let mid_s = *record.first()?;
    let radius_s = *record.get(1)?;
    let n_coeff = (record.len() - 2) / 3;
    let mut pos = [0.0_f64; 3];
    let mut vel = [0.0_f64; 3];
    for axis in 0..3 {
        let start = 2 + axis * n_coeff;
        let coeffs = record.get(start..start + n_coeff)?;
        let (p, v) = eval_chebyshev_pv(coeffs, mid_s, radius_s, t_s)?;
        pos[axis] = p * KM_TO_M;
        vel[axis] = v * KM_TO_M;
    }
    Some((Vector3::from(pos), Vector3::from(vel)))
}

fn eval_type3(record: &[f64], t_s: f64) -> Option<(Vector3<f64>, Vector3<f64>)> {
    if record.len() < 8 || !(record.len() - 2).is_multiple_of(6) {
        return None;
    }
    let mid_s = *record.first()?;
    let radius_s = *record.get(1)?;
    let n_coeff = (record.len() - 2) / 6;
    let mut pos = [0.0_f64; 3];
    let mut vel = [0.0_f64; 3];
    for axis in 0..3 {
        let p_start = 2 + axis * n_coeff;
        let v_start = 2 + (axis + 3) * n_coeff;
        let p_coeffs = record.get(p_start..p_start + n_coeff)?;
        let v_coeffs = record.get(v_start..v_start + n_coeff)?;
        let (p, _) = eval_chebyshev_pv(p_coeffs, mid_s, radius_s, t_s)?;
        let (v, _) = eval_chebyshev_pv(v_coeffs, mid_s, radius_s, t_s)?;
        pos[axis] = p * KM_TO_M;
        vel[axis] = v * KM_TO_M;
    }
    Some((Vector3::from(pos), Vector3::from(vel)))
}

fn eclipj2000_to_j2000() -> Matrix3<f64> {
    let (s, c) = SPICE_OBLIQUITY_J2000_RAD.sin_cos();
    Matrix3::new(1.0, 0.0, 0.0, 0.0, c, -s, 0.0, s, c)
}

#[cfg(test)]
mod tests;
