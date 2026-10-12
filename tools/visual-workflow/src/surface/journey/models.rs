//! A reload control isolates model residency from search and tracking decisions.
use crate::{
    Error,
    config::{Config, Runtime},
    surface::acquire::engine::{CachePolicy, Engine},
};
use navigate_visual::Frame;
use std::{
    path::{Path, PathBuf},
    time::Instant,
};

pub(super) struct Models {
    engine: Option<Engine>,
    root: PathBuf,
    runtime: Runtime,
    reload: bool,
    cache_policy: CachePolicy,
    pub statistics_before: Option<navigate_visual_onnx::TensorIoStatistics>,
    pub loads: usize,
    pub queries: usize,
}
impl Models {
    pub fn new(
        mut engine: Engine,
        runtime: Runtime,
        root: &Path,
        reload: bool,
        cache_policy: CachePolicy,
    ) -> Self {
        engine.configure_cache(cache_policy);
        Self {
            engine: Some(engine),
            runtime,
            root: root.into(),
            reload,
            cache_policy,
            statistics_before: None,
            loads: 1,
            queries: 0,
        }
    }
    pub fn prepare_blocking(
        &mut self,
        frame: &Frame,
        config: &Config,
    ) -> Result<(f64, f64), Error> {
        let mut reload_ms = 0.0;
        if self.reload && self.queries > 0 {
            let started = Instant::now();
            // Release the active model sessions before loading so the control does not retain two models.
            self.engine = None;
            self.engine = Some(Engine::new(
                self.runtime.load_blocking(config, &self.root)?,
            )?);
            let policy = self.cache_policy;
            self.engine_mut()?.configure_cache(policy);
            self.loads = self.loads.wrapping_add(1);
            reload_ms = started.elapsed().as_secs_f64() * 1000.0;
        }
        self.statistics_before = self.engine_mut()?.backend.tensor_statistics();
        let started = Instant::now();
        self.engine_mut()?.query_blocking(&frame.image, config)?;
        self.queries = self.queries.wrapping_add(1);
        Ok((reload_ms, started.elapsed().as_secs_f64() * 1000.0))
    }
    pub fn engine_mut(&mut self) -> Result<&mut Engine, Error> {
        self.engine
            .as_mut()
            .ok_or_else(|| Error::Input("acquisition model is unavailable".into()))
    }
}
