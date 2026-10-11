use crate::Error;
use std::{collections::VecDeque, sync::Arc};

struct Entry<T> {
    key: String,
    value: Arc<T>,
    bytes: usize,
}

pub(super) struct Cache<T> {
    entries: VecDeque<Entry<T>>,
    bytes: usize,
    max_entries: usize,
    max_bytes: usize,
}

impl<T> Cache<T> {
    pub fn new(max_entries: usize, max_bytes: usize) -> Self {
        Self {
            entries: VecDeque::new(),
            bytes: 0,
            max_entries,
            max_bytes,
        }
    }
    pub fn bytes(&self) -> usize {
        self.bytes
    }
    pub fn get_cached(&mut self, key: &str) -> Option<Arc<T>> {
        let position = self.entries.iter().position(|e| e.key == key)?;
        let entry = self.entries.remove(position)?;
        let value = Arc::clone(&entry.value);
        self.entries.push_back(entry);
        Some(value)
    }
    pub fn insert(
        &mut self,
        key: String,
        value: T,
        bytes: usize,
        protected: &[String],
    ) -> Result<Arc<T>, Error> {
        if bytes > self.max_bytes || self.max_entries == 0 {
            return Err(Error::Input(format!(
                "reference needs {bytes} bytes; cache limit is {}",
                self.max_bytes
            )));
        }
        while self.entries.len() >= self.max_entries
            || self.bytes.saturating_add(bytes) > self.max_bytes
        {
            let position = self
                .entries
                .iter()
                .position(|e| !protected.contains(&e.key))
                .ok_or_else(|| {
                    Error::Input("reference cache budget exhausted by current frame".into())
                })?;
            let entry = self
                .entries
                .remove(position)
                .ok_or_else(|| Error::Input("reference cache entry missing".into()))?;
            self.bytes = self.bytes.saturating_sub(entry.bytes);
        }
        let value = Arc::new(value);
        self.bytes = self.bytes.saturating_add(bytes);
        self.entries.push_back(Entry {
            key,
            value: Arc::clone(&value),
            bytes,
        });
        Ok(value)
    }
}

#[cfg(test)]
mod tests;
