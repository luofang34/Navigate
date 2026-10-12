//! Bounded owned entries use exact input keys and least-recently-used eviction.
use std::collections::VecDeque;

struct Entry<T> {
    key: [u8; 32],
    value: T,
    bytes: usize,
}
pub(super) struct Cache<T> {
    entries: VecDeque<Entry<T>>,
    pub limit: usize,
    pub bytes: usize,
    pub evictions: usize,
}
impl<T> Cache<T> {
    pub fn new(limit: usize) -> Self {
        Self {
            entries: VecDeque::new(),
            limit,
            bytes: 0,
            evictions: 0,
        }
    }
    pub fn take_cached(&mut self, key: &[u8; 32]) -> Option<(T, usize)> {
        let index = self.entries.iter().position(|e| &e.key == key)?;
        let entry = self.entries.remove(index)?;
        self.bytes -= entry.bytes;
        Some((entry.value, entry.bytes))
    }
    pub fn trim(&mut self, available: usize) {
        while self.bytes > available {
            let Some(entry) = self.entries.pop_front() else {
                break;
            };
            self.bytes -= entry.bytes;
            self.evictions = self.evictions.wrapping_add(1);
        }
    }
    pub fn entry_bytes(payload: usize) -> usize {
        payload.saturating_add(size_of::<Entry<T>>() - size_of::<T>())
    }
    pub fn insert(&mut self, key: [u8; 32], value: T, bytes: usize) {
        if bytes > self.limit {
            return;
        }
        self.trim(self.limit - bytes);
        self.entries.push_back(Entry { key, value, bytes });
        self.bytes += bytes;
    }
    pub fn len(&self) -> usize {
        self.entries.len()
    }
}
pub(super) fn fingerprint(
    namespace: [u8; 32],
    contract: [u8; 32],
    image: &image::GrayImage,
    window: Option<[u32; 4]>,
) -> [u8; 32] {
    use sha2::{Digest, Sha256};
    let mut hash = Sha256::new();
    hash.update(namespace);
    hash.update(contract);
    for value in window.unwrap_or([0, 0, image.width(), image.height()]) {
        hash.update(value.to_le_bytes());
    }
    hash.update(image.width().to_le_bytes());
    hash.update(image.height().to_le_bytes());
    hash.update(image.as_raw());
    hash.finalize().into()
}
pub(super) fn rotation_key(base: [u8; 32], rotation: u8) -> [u8; 32] {
    use sha2::{Digest, Sha256};
    let mut hash = Sha256::new();
    hash.update(base);
    hash.update([rotation]);
    hash.finalize().into()
}
#[cfg(test)]
mod tests;
