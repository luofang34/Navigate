//! Share one model and one validated query feature packet across map windows.
use super::{
    super::matching::{Matches, restore, rotate},
    cache::Cache,
};
use crate::{
    Error,
    config::{Backend, Config},
};
use image::GrayImage;
use navigate_visual_onnx::{XFeatFeatures, XFeatStages};
use navigate_visual_workflow::FeatureBackend;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::time::Instant;

const ENTRY_RESERVE: usize = 4 * 1024 * 1024;
#[derive(Clone, Copy)]
pub(in crate::surface) struct CachePolicy {
    pub bytes: usize,
    pub source: [u8; 32],
}
pub(super) struct PreparedMatch {
    pub candidate: Matches,
    pub preparation_ms: f64,
    pub extraction_ms: f64,
    pub cache_hit: bool,
}
pub(super) struct ReferenceInput<'a> {
    image: &'a GrayImage,
    key: Option<[u8; 32]>,
}
#[derive(Default)]
struct Counts {
    candidates: usize,
    hits: usize,
    rotations: usize,
    key_validations: usize,
}

pub(in crate::surface) struct Engine {
    pub backend: Box<XFeatStages>,
    query: Option<XFeatFeatures>,
    cache: Cache<XFeatFeatures>,
    namespace: [u8; 32],
    counts: Counts,
    pub query_bytes: usize,
    pub reference_encodings: usize,
    pub peak_reference_bytes: usize,
}
impl Engine {
    pub fn new(backend: Backend) -> Result<Self, Error> {
        let Backend::Features(backend) = backend else {
            return Err(Error::Input(
                "acquisition requires a feature backend".into(),
            ));
        };
        Ok(Self {
            backend,
            query: None,
            cache: Cache::new(0),
            namespace: [0; 32],
            counts: Counts::default(),
            query_bytes: 0,
            reference_encodings: 0,
            peak_reference_bytes: 0,
        })
    }
    pub fn configure_cache(&mut self, policy: CachePolicy) {
        self.cache = Cache::new(policy.bytes);
        self.namespace = policy.source;
    }
    pub fn query_blocking(&mut self, query: &GrayImage, config: &Config) -> Result<(), Error> {
        self.query = None;
        self.query_bytes = 0;
        self.reference_encodings = 0;
        self.counts = Counts::default();
        self.cache.evictions = 0;
        self.peak_reference_bytes = 0;
        let features = self.backend.extract_features_blocking(query)?;
        let info = self.backend.validate_features(&features, query)?;
        if info.count > config.limits.features_per_image
            || info.resident_bytes > config.limits.query_feature_bytes
        {
            return Err(Error::Input("acquisition query feature budget".into()));
        }
        self.query = Some(features);
        self.query_bytes = info.resident_bytes;
        Ok(())
    }
    pub fn prepare_raster(&mut self, bytes: usize, config: &Config) -> Result<(), Error> {
        if bytes > config.limits.reference_bytes {
            return Err(Error::Input("acquisition raster budget".into()));
        }
        self.cache.trim(config.limits.reference_bytes - bytes);
        self.peak_reference_bytes = self.peak_reference_bytes.max(bytes + self.cache.bytes);
        Ok(())
    }
    pub fn cache_report(&self) -> Value {
        json!({"limit_bytes":self.cache.limit,"resident_bytes":self.cache.bytes,"entries":self.cache.len(),
            "hits":self.counts.hits,"misses":self.reference_encodings,"candidates":self.counts.candidates,
            "evictions":self.cache.evictions,"rotations":self.counts.rotations,"source_pixel_key_validations":self.counts.key_validations,
            "source_namespace":format!("{:x}",Sha256::digest(self.namespace)),
            "scope":"Exact host feature handles, keyed by source map identity, encoder contract, crop, rotation and actual grayscale pixels. Hits skip rotation and extraction. Matching still uses its existing tensor adapter. Counts reset for each acquisition; entries persist within one model session lifetime. Metadata for live entries is included; allocator and container spare capacity are excluded."})
    }
    pub(super) fn match_blocking(
        &mut self,
        input: &ReferenceInput<'_>,
        rotation: u8,
        resident_raster_bytes: usize,
        config: &Config,
    ) -> Result<PreparedMatch, Error> {
        let image = input.image;
        if self.counts.candidates >= config.limits.candidates {
            return Err(Error::Input(
                "acquisition candidate budget exhausted".into(),
            ));
        }
        self.counts.candidates = self.counts.candidates.wrapping_add(1);
        self.prepare_raster(resident_raster_bytes, config)?;
        let start = Instant::now();
        let key = input
            .key
            .map(|base| super::cache::rotation_key(base, rotation));
        let cached = key.as_ref().and_then(|k| self.cache.take_cached(k));
        let hit = cached.is_some();
        let (features, bytes, extraction_ms) = if let Some((features, bytes)) = cached {
            if resident_raster_bytes.saturating_add(bytes) > config.limits.reference_bytes {
                return Err(Error::Input(
                    "cached reference handle exceeds total budget".into(),
                ));
            }
            self.counts.hits = self.counts.hits.wrapping_add(1);
            self.cache.trim(
                config
                    .limits
                    .reference_bytes
                    .saturating_sub(resident_raster_bytes + bytes),
            );
            self.peak_reference_bytes = self
                .peak_reference_bytes
                .max(resident_raster_bytes + self.cache.bytes + bytes);
            (features, bytes, 0.0)
        } else {
            self.extract_blocking(image, rotation, resident_raster_bytes, config)?
        };
        let preparation_ms = start.elapsed().as_secs_f64() * 1000.0;
        let start = Instant::now();
        let mut pairs = self.backend.match_features_blocking(
            &features,
            self.query
                .as_ref()
                .ok_or_else(|| Error::Input("acquisition query is not encoded".into()))?,
        )?;
        for pair in &mut pairs {
            restore(pair, rotation, image.dimensions());
        }
        let matching_ms = start.elapsed().as_secs_f64() * 1000.0;
        if let Some(key) = key {
            self.cache.insert(key, features, bytes);
        }
        Ok(PreparedMatch {
            candidate: Matches {
                rotation,
                pairs,
                matching_ms,
            },
            preparation_ms,
            extraction_ms,
            cache_hit: hit,
        })
    }
    pub(super) fn reference_input<'a>(
        &mut self,
        image: &'a GrayImage,
        window: Option<[u32; 4]>,
    ) -> ReferenceInput<'a> {
        let key = if self.cache.limit == 0 {
            None
        } else {
            self.counts.key_validations = self.counts.key_validations.wrapping_add(1);
            Some(super::cache::fingerprint(
                self.namespace,
                self.backend.contract(),
                image,
                window,
            ))
        };
        ReferenceInput { image, key }
    }
    fn extract_blocking(
        &mut self,
        image: &GrayImage,
        rotation: u8,
        raster_bytes: usize,
        config: &Config,
    ) -> Result<(XFeatFeatures, usize, f64), Error> {
        let overlap = raster_bytes.saturating_add(image.as_raw().len());
        if overlap > config.limits.reference_bytes {
            return Err(Error::Input("acquisition rotated raster budget".into()));
        }
        if self.cache.limit > 0 {
            // Reserve a bounded host handle before extraction. Runtime scratch is separate.
            if overlap.saturating_add(ENTRY_RESERVE) > config.limits.reference_bytes {
                return Err(Error::Input(
                    "acquisition feature reservation budget".into(),
                ));
            }
            self.cache
                .trim(config.limits.reference_bytes - overlap - ENTRY_RESERVE);
        }
        let started = Instant::now();
        let rotated = rotate(image, rotation);
        self.counts.rotations = self.counts.rotations.wrapping_add(1);
        let features = self.backend.extract_features_blocking(&rotated)?;
        let info = self.backend.validate_features(&features, &rotated)?;
        let bytes = if self.cache.limit > 0 {
            Cache::<XFeatFeatures>::entry_bytes(info.resident_bytes)
        } else {
            info.resident_bytes
        };
        let peak = overlap
            .saturating_add(self.cache.bytes)
            .saturating_add(bytes);
        if peak > config.limits.reference_bytes
            || info.count > config.limits.features_per_image
            || (self.cache.limit > 0 && bytes > ENTRY_RESERVE)
        {
            return Err(Error::Input("acquisition reference feature budget".into()));
        }
        self.peak_reference_bytes = self.peak_reference_bytes.max(peak);
        self.reference_encodings = self.reference_encodings.wrapping_add(1);
        Ok((features, bytes, started.elapsed().as_secs_f64() * 1000.0))
    }
}
