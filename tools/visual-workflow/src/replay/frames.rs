use super::{cache::Cache, stream::Record};
use crate::{
    Error, assets,
    config::Config,
    job::{Job, References},
    run,
};
use navigate_visual_onnx::{XFeatFeatures, XFeatStages};
use navigate_visual_workflow::{FeatureBackend, FeatureLocalizer, PreparedReference};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{path::Path, time::Instant};

pub(super) struct Frames<'a> {
    workflow: FeatureLocalizer<XFeatStages>,
    cache: Cache<PreparedReference<XFeatFeatures>>,
    source: Option<(String, References)>,
    config: &'a Config,
}

impl<'a> Frames<'a> {
    pub fn new(workflow: FeatureLocalizer<XFeatStages>, config: &'a Config) -> Self {
        Self {
            workflow,
            cache: Cache::new(config.limits.candidates, config.limits.reference_bytes),
            source: None,
            config,
        }
    }
    pub fn process_blocking(&mut self, record: &Record, sequence: u64) -> Result<Value, Error> {
        let job: Job = assets::read_json_blocking(&record.job)?;
        if job.references.len() > self.config.limits.candidates {
            return Err(Error::Input(
                "candidate count exceeds the configured limit".into(),
            ));
        }
        let root = record
            .job
            .parent()
            .ok_or_else(|| Error::Input("job parent".into()))?;
        let mut frame = job.query_blocking(root, self.config.limits.image_pixels)?;
        frame.stamp.sequence = sequence;
        frame.stamp.capture_time_ns = record.capture_time_ns;
        let prior = job.prior()?;
        let start = Instant::now();
        let context = context_key(&job, root)?;
        if self.source.as_ref().is_some_and(|(key, _)| key != &context) {
            self.source = None;
        }
        if let Some((_, source)) = &mut self.source {
            source.render_ms.clear();
        }
        let keys = job
            .references
            .iter()
            .map(|r| digest(&json!([context, r])))
            .collect::<Result<Vec<_>, _>>()?;
        let mut selected = Vec::new();
        let mut hits = 0_usize;
        for (entry, key) in job.references.iter().zip(&keys) {
            let value = if let Some(value) = self.cache.get_cached(key) {
                hits = hits.wrapping_add(1);
                value
            } else {
                self.prepare_blocking(&job, root, entry, key, &context, &keys)?
            };
            selected.push(value);
        }
        let prepare_ms = start.elapsed().as_secs_f64() * 1000.0;
        let borrowed: Vec<_> = selected.iter().map(|r| r.as_ref()).collect();
        let report = self
            .workflow
            .estimate_selected_blocking(&frame, &prior, &borrowed)?;
        Ok(
            json!({"status":"processed", "backend":self.workflow.backend().identity(),
            "reference_ids":job.references.iter().map(|r|&r.id).collect::<Vec<_>>(),
            "reference_cache_hits":hits,"reference_cache_bytes":self.cache.bytes(),
            "reference_prepare_ms":prepare_ms,"observation_sha256":frame.evidence_sha256(),
            "render":self.source.as_ref().map(|(_, source)|run::render_report(source)),
            "report":run::report(report),"adapter_tensor_io":self.workflow.backend().tensor_statistics()}),
        )
    }
    fn prepare_blocking(
        &mut self,
        job: &Job,
        root: &Path,
        entry: &crate::job::Reference,
        key: &str,
        context: &str,
        protected: &[String],
    ) -> Result<std::sync::Arc<PreparedReference<XFeatFeatures>>, Error> {
        if entry.image.is_none()
            && self
                .source
                .as_ref()
                .is_none_or(|(_, source)| source.identity().is_none())
        {
            self.source = Some((
                context.into(),
                job.references_blocking(root, self.config.limits.image_pixels)?,
            ));
        }
        let mut file_source = References::default();
        let source = self
            .source
            .as_mut()
            .map_or(&mut file_source, |(_, source)| source);
        let view = job.reference_blocking(entry, root, self.config.limits.image_pixels, source)?;
        let prepared = if let Some(path) = &entry.features {
            let bytes = assets::read_blocking(&root.join(path), 2 * 1024 * 1024)?;
            self.workflow
                .import_reference(view, XFeatFeatures::decode(&bytes, self.config.keypoints)?)?
        } else {
            self.workflow.prepare_reference_blocking(view)?
        };
        let bytes = prepared.resident_bytes();
        self.cache.insert(key.into(), prepared, bytes, protected)
    }
}
fn context_key(job: &Job, root: &Path) -> Result<String, Error> {
    digest(&json!([
        root,
        job.camera,
        job.anchor_lat_lon,
        job.coordinate_frame,
        job.map_release,
        job.map_sha256,
        job.render_scene
    ]))
}
fn digest(value: &Value) -> Result<String, Error> {
    let bytes = serde_json::to_vec(value).map_err(|source| Error::Json {
        path: "reference cache key".into(),
        source,
    })?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}
