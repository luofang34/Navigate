//! A bounded queue uses capture timestamps and measured wall time to admit frames.
use crate::{
    Error,
    surface::sequence::{input::Sample, tracks::MAX_TRACK_GAP_NS},
};
use serde_json::{Value, json};
use std::{
    collections::VecDeque,
    time::{Duration, Instant},
};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Policy {
    Offline,
    Fifo,
    Latest,
    CatchUp,
}
#[derive(Clone, Copy)]
pub struct Options {
    policy: Policy,
    capacity: usize,
    during_setup: bool,
}
impl Options {
    pub fn new(policy: &str, capacity: usize, start: &str) -> Result<Self, Error> {
        let policy = match policy {
            "offline" => Policy::Offline,
            "fifo" => Policy::Fifo,
            "latest" => Policy::Latest,
            "catch-up" => Policy::CatchUp,
            _ => return Err(Error::Input("unknown arrival policy".into())),
        };
        if !(1..=128).contains(&capacity) {
            return Err(Error::Input(
                "arrival queue capacity must be 1..=128".into(),
            ));
        }
        let during_setup = match start {
            "setup" => true,
            "ready" => false,
            _ => return Err(Error::Input("capture start must be setup or ready".into())),
        };
        Ok(Self {
            policy,
            capacity: if policy == Policy::Latest {
                1
            } else {
                capacity
            },
            during_setup,
        })
    }
    fn name(self) -> &'static str {
        match self.policy {
            Policy::Offline => "offline",
            Policy::Fifo => "fifo",
            Policy::Latest => "latest",
            Policy::CatchUp => "catch-up",
        }
    }
}
struct Queue {
    next: usize,
    pending: VecDeque<usize>,
    dropped: Vec<Value>,
    peak: usize,
    options: Options,
}
impl Queue {
    fn new(options: Options) -> Self {
        Self {
            next: 0,
            pending: VecDeque::new(),
            dropped: Vec::new(),
            peak: 0,
            options,
        }
    }
    fn advance(&mut self, samples: &[Sample], elapsed_ns: u128) {
        let epoch = samples.first().map_or(0, |s| s.capture_time_ns);
        while let Some(sample) = samples.get(self.next) {
            if u128::from(sample.capture_time_ns - epoch) > elapsed_ns {
                break;
            }
            if self.pending.len() == self.options.capacity
                && let Some(index) = self.pending.pop_front()
            {
                self.drop_record(
                    samples,
                    index,
                    self.next,
                    elapsed_ns,
                    if self.options.policy == Policy::Latest {
                        "superseded"
                    } else {
                        "overflow"
                    },
                );
            }
            self.pending.push_back(self.next);
            self.next = self.next.wrapping_add(1);
            self.peak = self.peak.max(self.pending.len());
        }
    }
    fn drop_record(
        &mut self,
        samples: &[Sample],
        index: usize,
        replacement: usize,
        elapsed_ns: u128,
        reason: &str,
    ) {
        let dropped = &samples[index];
        self.dropped.push(json!({"frame_id":dropped.frame_id,"capture_time_ns":dropped.capture_time_ns,
            "replaced_by_frame_id":samples[replacement].frame_id,"detected_at_ms":elapsed_ns as f64 / 1e6,"reason":reason}));
    }
    fn select(
        &mut self,
        samples: &[Sample],
        anchor: Option<u64>,
        elapsed_ns: u128,
    ) -> Option<usize> {
        if self.options.policy != Policy::CatchUp {
            return self.pending.pop_front();
        }
        let eligible = anchor.and_then(|stamp| {
            self.pending.iter().rposition(|&index| {
                samples[index].capture_time_ns > stamp
                    && samples[index].capture_time_ns <= stamp.saturating_add(MAX_TRACK_GAP_NS)
            })
        });
        let position = eligible.unwrap_or(self.pending.len().saturating_sub(1));
        let selected = *self.pending.get(position)?;
        let reason = if eligible.is_some() {
            "catch_up_stride"
        } else if anchor.is_none() {
            "no_anchor"
        } else {
            "history_gap"
        };
        for _ in 0..position {
            if let Some(index) = self.pending.pop_front() {
                self.drop_record(samples, index, selected, elapsed_ns, reason);
            }
        }
        self.pending.pop_front()
    }
}
pub(super) struct Arrivals {
    queue: Queue,
    clock: Instant,
    initial_elapsed_ms: f64,
}
pub(super) struct Selection {
    pub index: usize,
    started: Instant,
    queue_depth: usize,
    dropped_before: usize,
}
impl Arrivals {
    pub fn new(options: Options, setup_start: Instant) -> Self {
        let clock = if options.during_setup {
            setup_start
        } else {
            Instant::now()
        };
        Self {
            queue: Queue::new(options),
            clock,
            initial_elapsed_ms: clock.elapsed().as_secs_f64() * 1000.0,
        }
    }
    pub fn next_blocking(&mut self, samples: &[Sample], anchor: Option<u64>) -> Option<Selection> {
        if self.queue.options.policy == Policy::Offline {
            let index = self.queue.next;
            samples.get(index)?;
            self.queue.next = self.queue.next.wrapping_add(1);
            return Some(Selection {
                index,
                started: Instant::now(),
                queue_depth: 0,
                dropped_before: 0,
            });
        }
        let epoch = samples.first()?.capture_time_ns;
        loop {
            self.queue.advance(samples, self.clock.elapsed().as_nanos());
            let queue_depth = self.queue.pending.len();
            if let Some(index) = self
                .queue
                .select(samples, anchor, self.clock.elapsed().as_nanos())
            {
                return Some(Selection {
                    index,
                    started: Instant::now(),
                    queue_depth,
                    dropped_before: self.queue.dropped.len(),
                });
            }
            let sample = samples.get(self.queue.next)?;
            let due = Duration::from_nanos(sample.capture_time_ns - epoch);
            // Pace capture arrivals; no recorded inference duration controls this clock.
            std::thread::sleep(due.saturating_sub(self.clock.elapsed()));
        }
    }
    pub fn completed(&mut self, selection: Selection, samples: &[Sample]) -> Value {
        if self.queue.options.policy == Policy::Offline {
            return Value::Null;
        }
        let completed = Instant::now();
        let elapsed = completed.duration_since(self.clock);
        self.queue.advance(samples, elapsed.as_nanos());
        let epoch = samples.first().map_or(0, |s| s.capture_time_ns);
        let capture_offset_ms = (samples[selection.index].capture_time_ns - epoch) as f64 / 1e6;
        let start_ms = selection.started.duration_since(self.clock).as_secs_f64() * 1000.0;
        let completion_ms = elapsed.as_secs_f64() * 1000.0;
        json!({"start_ms":start_ms,"completion_ms":completion_ms,"capture_offset_ms":capture_offset_ms,
            "input_age_ms":start_ms-capture_offset_ms,"result_age_ms":completion_ms-capture_offset_ms,
            "queue_depth_before_take":selection.queue_depth,"queue_depth_at_completion":self.queue.pending.len(),
            "arrived_total":self.queue.next,"dropped_before":selection.dropped_before,"dropped_total":self.queue.dropped.len()})
    }
    pub fn report(self, samples: &[Sample]) -> Value {
        if self.queue.options.policy == Policy::Offline {
            return json!({"policy":"offline"});
        }
        let duration_ms = self.clock.elapsed().as_secs_f64() * 1000.0;
        let capture_span_ms = samples
            .last()
            .zip(samples.first())
            .map_or(0.0, |(last, first)| {
                (last.capture_time_ns - first.capture_time_ns) as f64 / 1e6
            });
        json!({"policy":self.queue.options.name(),"queue_capacity":self.queue.options.capacity,
            "capture_start":if self.queue.options.during_setup {"workflow_setup"} else {"models_and_map_ready"},
            "initial_elapsed_ms":self.initial_elapsed_ms,"capture_span_ms":capture_span_ms,"duration_ms":duration_ms,
            "drain_ms":(duration_ms-capture_span_ms).max(0.0),"arrived":self.queue.next,"peak_queue_depth":self.queue.peak,
            "pending":self.queue.pending.len(),"dropped":self.queue.dropped,"tracking_gap_limit_ns":MAX_TRACK_GAP_NS,
            "scope":"Capture timestamps are shifted by the first sample timestamp onto a measured monotonic clock. Queue admission advances at worker boundaries using actual elapsed time. Catch-up selects the latest retained frame within the tracking gap from the current accepted anchor; without retained eligible history it selects the latest arrival for recovery. Pending items are file records, not decoded frames; selected images are verified and decoded within processing time. The finite recording ends at the last capture timestamp and the remaining queue is drained. Output serialization and live camera costs are excluded. Result timestamps remain the original capture timestamps; no extrapolation or freshness guarantee."})
    }
}
#[cfg(test)]
mod tests;
