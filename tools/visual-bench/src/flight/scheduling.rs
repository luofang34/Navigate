//! File replay uses the same budget controller as a serialized live worker.
use super::FlightArgs;
use crate::BenchError;
use navigate_visual::{AdaptiveSampler, SamplingConfig, SamplingStatus};
use std::time::{Duration, Instant};
pub(super) struct Schedule {
    sampler: AdaptiveSampler,
    last: Option<Duration>,
    fixed_interval: Duration,
    adaptive: bool,
}
impl Schedule {
    pub fn new(args: &FlightArgs) -> Result<Self, BenchError> {
        let interval = Duration::from_secs_f64(1.0 / args.fps);
        Ok(Self {
            sampler: AdaptiveSampler::new(
                SamplingConfig {
                    minimum_interval: interval,
                    maximum_interval: interval.max(Duration::from_secs(1)),
                    utilization: args.utilization,
                },
                if args.adaptive {
                    Duration::from_millis(100)
                } else {
                    Duration::from_nanos(1)
                },
            )?,
            last: None,
            fixed_interval: interval,
            adaptive: args.adaptive,
        })
    }
    pub fn admit_blocking(
        &mut self,
        pts: u64,
        next: Option<u64>,
        wall: Instant,
        args: &FlightArgs,
    ) -> Result<bool, BenchError> {
        let seconds = pts as f64 / 1e9;
        if seconds < args.start || args.duration.is_some_and(|d| seconds >= args.start + d) {
            return Ok(false);
        }
        let capture = Duration::from_secs_f64(seconds - args.start);
        if args.realtime {
            if next.is_some_and(|n| n as f64 / 1e9 - args.start <= wall.elapsed().as_secs_f64()) {
                return Ok(false);
            }
            if let Some(delay) = capture.checked_sub(wall.elapsed()) {
                std::thread::sleep(delay)
            }
        }
        if let Some(path) = &args.budget_file {
            let bytes = crate::read_blocking(path)?;
            let fraction =
                serde_json::from_slice::<f64>(&bytes).map_err(|source| BenchError::Json {
                    path: path.clone(),
                    source,
                })?;
            self.sampler.set_utilization(fraction)?;
        }
        let now = if args.realtime {
            wall.elapsed()
        } else {
            capture
        };
        if !args.adaptive
            && self
                .last
                .is_some_and(|last| now.saturating_sub(last) < self.fixed_interval)
        {
            return Ok(false);
        }
        let admitted = self.sampler.begin(now)?;
        if admitted {
            self.last = Some(now)
        }
        Ok(admitted)
    }
    pub fn finish_work(
        &mut self,
        cost: Duration,
        warmup: bool,
    ) -> Result<SamplingStatus, BenchError> {
        if warmup {
            Ok(self.sampler.status())
        } else {
            self.complete(cost)
        }
    }
    pub fn complete(&mut self, cost: Duration) -> Result<SamplingStatus, BenchError> {
        let mut status = self.sampler.complete(if self.adaptive {
            cost
        } else {
            Duration::from_nanos(1)
        })?;
        if !self.adaptive {
            status.estimated_cost = cost;
            status.deadline_unattainable = cost > self.fixed_interval;
        }
        Ok(status)
    }
}
pub(super) fn validate(args: &FlightArgs) -> Result<(), BenchError> {
    if args.surface_tracks && (args.fixed_tilt || args.keyframe_interval != 0.0 || args.dense_only)
    {
        return Err(BenchError::Record { reason: "surface tracks require free attitude, adjacent image tracking, and a point-tracking backend".into() });
    }
    if args.budget_file.is_some() && !args.adaptive {
        return Err(BenchError::Record {
            reason: "a live budget requires --adaptive".into(),
        });
    }
    if !args.fps.is_finite()
        || !(0.01..=60.0).contains(&args.fps)
        || !(64..=1920).contains(&args.width)
        || !args.start.is_finite()
        || args.start < 0.0
        || args.duration.is_some_and(|d| !d.is_finite() || d <= 0.0)
        || !args.map_interval.is_finite()
        || args.map_interval < 0.0
        || !args.keyframe_interval.is_finite()
        || !(0.0..=30.0).contains(&args.keyframe_interval)
    {
        return Err(BenchError::Record {
            reason: "invalid flight sampling or timing settings".into(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests;
