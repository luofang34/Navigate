//! Opt-in host timings; logical payloads do not measure physical memory traffic.
use std::time::Instant;

pub(super) struct Timer(Option<Instant>);
#[derive(Debug)]
struct Stage {
    name: &'static str,
    host_us: f64,
    logical_bytes: usize,
    gpu_buffers: usize,
}
pub(super) struct Profile {
    enabled: bool,
    stages: Vec<Stage>,
}
impl Profile {
    pub fn new() -> Self {
        let enabled =
            tracing::enabled!(target: "navigate_visual::gpu::timings", tracing::Level::DEBUG);
        Self {
            enabled,
            stages: Vec::with_capacity(if enabled { 16 } else { 0 }),
        }
    }

    pub fn timer(&self) -> Timer {
        Timer(self.enabled.then(Instant::now))
    }
    pub fn record(
        &mut self,
        timer: Timer,
        name: &'static str,
        logical_bytes: usize,
        gpu_buffers: usize,
    ) {
        if let Some(start) = timer.0 {
            self.stages.push(Stage {
                name,
                host_us: start.elapsed().as_secs_f64() * 1_000_000.0,
                logical_bytes,
                gpu_buffers,
            });
        }
    }
    pub fn finish(self, kind: &'static str) {
        if self.enabled {
            // Emit only after alignment; stage I/O must not interrupt device submission.
            let stages: Vec<_> = self
                .stages
                .iter()
                .map(|s| (s.name, s.host_us, s.logical_bytes, s.gpu_buffers))
                .collect();
            tracing::debug!(target: "navigate_visual::gpu::timings", kind, ?stages, "GPU host stages");
        }
    }
}
