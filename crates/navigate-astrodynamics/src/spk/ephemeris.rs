//! Body-center graph over loaded `SPK` segments and chained relative states.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::vec::Vec;

use nalgebra::Vector3;

use super::{MARS, MARS_BARYCENTER, PosVel, SpkError, SpkFile, SpkSegment};

/// Directed edge in the SPK body-center graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpkDirectedLink {
    /// Edge-list slot index.
    pub slot: usize,
    /// Whether traversal follows the canonical `lo -> hi` pair direction.
    pub forward: bool,
}

/// Multi-file binary `DAF/SPK` ephemeris catalog with last-loaded segment precedence
/// and shortest-chain BFS resolution.
#[derive(Clone, Debug)]
pub struct SpkEphemeris {
    edge_lists: Vec<Vec<(usize, bool)>>,
    segments: Vec<SpkSegment>,
    adjacency: BTreeMap<i32, Vec<(i32, SpkDirectedLink)>>,
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
        let mut idx = 0_usize;
        while let Some(seg) = segments.get(idx) {
            let (t, c) = (seg.target(), seg.center());
            known_bodies.insert(t);
            known_bodies.insert(c);
            let (lo, hi, is_lo_to_hi) = if t <= c { (t, c, true) } else { (c, t, false) };
            pair_map
                .entry((lo, hi))
                .or_default()
                .push((idx, is_lo_to_hi));
            idx = idx.wrapping_add(1);
        }
        let mut edge_lists = Vec::with_capacity(pair_map.len());
        let mut adjacency: BTreeMap<i32, Vec<(i32, SpkDirectedLink)>> = BTreeMap::new();
        for ((lo, hi), mut list) in pair_map {
            list.reverse();
            let slot = edge_lists.len();
            edge_lists.push(list);
            adjacency.entry(lo).or_default().push((
                hi,
                SpkDirectedLink {
                    slot,
                    forward: true,
                },
            ));
            adjacency.entry(hi).or_default().push((
                lo,
                SpkDirectedLink {
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

    /// All loaded segments in load order.
    #[must_use]
    pub fn segments(&self) -> &[SpkSegment] {
        &self.segments
    }

    /// Set of all NAIF body or barycentre IDs appearing in any loaded segment.
    #[must_use]
    pub const fn known_bodies(&self) -> &BTreeSet<i32> {
        &self.known_bodies
    }

    /// Candidate `(segment_index, is_lo_to_hi)` pairs for `link` in high-to-low precedence order.
    #[must_use]
    pub fn link_candidates(&self, link: SpkDirectedLink) -> &[(usize, bool)] {
        self.edge_lists.get(link.slot).map_or(&[], Vec::as_slice)
    }

    /// Shortest directed segment chain connecting `target` to `center`.
    ///
    /// # Errors
    /// Returns [`SpkError::UnconnectedBodies`] if no segment chain connects `target` to `center`.
    pub fn resolve_chain_links(
        &self,
        target: i32,
        center: i32,
    ) -> Result<Vec<SpkDirectedLink>, SpkError> {
        if target == center {
            return Ok(Vec::new());
        }
        let mut visited: BTreeMap<i32, Option<(i32, SpkDirectedLink)>> = BTreeMap::new();
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

    fn remap_body(&self, id: i32) -> i32 {
        if self.known_bodies.contains(&id) {
            return id;
        }
        if id == MARS && self.known_bodies.contains(&MARS_BARYCENTER) {
            return MARS_BARYCENTER;
        }
        id
    }

    fn eval_link(&self, link: SpkDirectedLink, tdb_s: f64) -> Result<Option<PosVel>, SpkError> {
        for &(seg_idx, is_lo_to_hi) in self.link_candidates(link) {
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
    /// in `J2000` axes at `tdb_s` (TDB seconds past J2000).
    ///
    /// # Errors
    /// Returns [`SpkError`] if no segment chain connects `target` to `center` or `tdb_s` is outside coverage.
    pub fn relative_state_tdb_j2000(
        &self,
        target: i32,
        center: i32,
        tdb_s: f64,
    ) -> Result<(Vector3<f64>, Vector3<f64>), SpkError> {
        let t_eff = self.remap_body(target);
        let c_eff = self.remap_body(center);
        if t_eff == c_eff {
            return Ok((Vector3::zeros(), Vector3::zeros()));
        }
        let chain = self.resolve_chain_links(t_eff, c_eff)?;
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
