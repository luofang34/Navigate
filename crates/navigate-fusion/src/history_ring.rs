//! A ring of history entries with a fixed inline capacity (ADR-0016).
//!
//! The kinematic filter and the inertial filter keep their bounded
//! delayed-measurement histories in this ring. The ring stores the
//! entries in place and does not allocate. It keeps a head index and a
//! length, so that the removal of the oldest entry does not move the
//! other entries.

/// Up to `N` entries, oldest first. A caller bound below `N` reduces the
/// capacity further.
#[derive(Debug, Clone, Copy)]
pub(crate) struct HistoryRing<T: Copy, const N: usize> {
    slots: [Option<T>; N],
    head: usize,
    len: usize,
}

impl<T: Copy, const N: usize> HistoryRing<T, N> {
    /// An empty ring.
    pub(crate) const fn new() -> Self {
        Self {
            slots: [const { None }; N],
            head: 0,
            len: 0,
        }
    }

    /// The capacity for the caller bound `bound`.
    const fn capacity(bound: usize) -> usize {
        if bound < N { bound } else { N }
    }

    /// The slot of the entry at logical position `index`.
    const fn slot(&self, index: usize) -> usize {
        self.head.wrapping_add(index) % N
    }

    /// Forgets every entry.
    pub(crate) fn clear(&mut self) {
        self.slots = [const { None }; N];
        self.head = 0;
        self.len = 0;
    }

    /// The number of entries.
    pub(crate) const fn len(&self) -> usize {
        self.len
    }

    /// The entry at logical position `index`, oldest first.
    pub(crate) fn get(&self, index: usize) -> Option<&T> {
        if index >= self.len {
            return None;
        }
        self.slots.get(self.slot(index))?.as_ref()
    }

    /// The entry at logical position `index`, for a change in place.
    pub(crate) fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        if index >= self.len {
            return None;
        }
        let slot = self.slot(index);
        self.slots.get_mut(slot)?.as_mut()
    }

    /// The oldest entry.
    pub(crate) fn front(&self) -> Option<&T> {
        self.get(0)
    }

    /// The entries, oldest first.
    pub(crate) fn iter(&self) -> impl Iterator<Item = &T> {
        (0..self.len).filter_map(move |index| self.get(index))
    }

    /// The number of leading entries for which `before` is true. The
    /// entries must be partitioned: every entry for which `before` is
    /// true comes before every other entry.
    pub(crate) fn partition_point(&self, before: impl Fn(&T) -> bool) -> usize {
        self.iter().take_while(|entry| before(entry)).count()
    }

    /// Removes the oldest entry.
    fn pop_front(&mut self) {
        if self.len == 0 {
            return;
        }
        if let Some(slot) = self.slots.get_mut(self.head) {
            *slot = None;
        }
        self.head = self.slot(1);
        self.len = self.len.wrapping_sub(1);
    }

    /// Adds `entry` as the newest entry. The oldest entries go when the
    /// ring holds more than `bound` entries. A zero bound keeps nothing.
    pub(crate) fn push_back(&mut self, entry: T, bound: usize) {
        self.insert(self.len, entry, bound);
    }

    /// Puts `entry` at logical position `index` and moves the later
    /// entries back by one position. The oldest entries go when the ring
    /// holds more than `bound` entries. When `index` is zero in a full
    /// ring, `entry` is the oldest entry, so it goes at once.
    pub(crate) fn insert(&mut self, index: usize, entry: T, bound: usize) {
        let capacity = Self::capacity(bound);
        let mut index = index.min(self.len);
        while self.len > 0 && self.len >= capacity {
            if index == 0 {
                return;
            }
            self.pop_front();
            index = index.wrapping_sub(1);
        }
        if capacity == 0 {
            return;
        }
        let mut position = self.len;
        while position > index {
            let from = self.slot(position.wrapping_sub(1));
            let to = self.slot(position);
            let moved = self.slots.get(from).copied().flatten();
            if let Some(slot) = self.slots.get_mut(to) {
                *slot = moved;
            }
            position = position.wrapping_sub(1);
        }
        let target = self.slot(index);
        if let Some(slot) = self.slots.get_mut(target) {
            *slot = Some(entry);
        }
        self.len = self.len.wrapping_add(1);
    }
}

#[cfg(test)]
mod tests;
