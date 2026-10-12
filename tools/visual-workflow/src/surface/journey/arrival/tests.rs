use super::*;
fn samples() -> Vec<Sample> {
    (0..5)
        .map(|i| Sample {
            frame_id: 100 + i,
            capture_time_ns: 1_000_000_000 + i * 33_333_333,
            image: crate::job::Image {
                path: format!("{i}.png").into(),
                sha256: "a".repeat(64),
            },
        })
        .collect()
}
#[test]
fn fifo_preserves_order_drops_oldest_and_drains_after_last_arrival() {
    let samples = samples();
    let mut queue = Queue::new(Options::new("fifo", 2, "ready").unwrap());
    queue.advance(&samples, 0);
    assert_eq!(queue.pending.pop_front(), Some(0));
    queue.advance(&samples, 100_000_000);
    assert_eq!(
        queue.pending.iter().copied().collect::<Vec<_>>(),
        vec![2, 3]
    );
    assert_eq!(queue.dropped[0]["frame_id"], 101);
    assert_eq!(queue.dropped[0]["capture_time_ns"], 1_033_333_333_u64);
    assert_eq!(queue.pending.pop_front(), Some(2));
    queue.advance(&samples, 1_000_000_000);
    assert_eq!(queue.pending.pop_front(), Some(3));
    assert_eq!(queue.pending.pop_front(), Some(4));
    assert_eq!(queue.next, samples.len());
    assert_eq!(queue.peak, 2);
    assert!(queue.pending.is_empty());
}
#[test]
fn latest_never_selects_a_future_frame_and_keeps_only_latest_arrival() {
    let samples = samples();
    let mut queue = Queue::new(Options::new("latest", 128, "setup").unwrap());
    queue.advance(&samples, 66_666_665);
    assert_eq!(queue.pending.pop_front(), Some(1));
    assert_eq!(queue.dropped.len(), 1);
    assert_eq!(queue.dropped[0]["reason"], "superseded");
    queue.advance(&samples, 66_666_666);
    assert_eq!(queue.pending.pop_front(), Some(2));
    assert_eq!(queue.peak, 1);
    assert_eq!(samples[2].capture_time_ns, 1_066_666_666);
}
#[test]
fn invalid_queue_options_are_rejected() {
    for capacity in [0, 129, usize::MAX] {
        assert!(Options::new("fifo", capacity, "ready").is_err());
    }
    assert!(Options::new("random", 2, "ready").is_err());
    assert!(Options::new("fifo", 2, "later").is_err());
}

#[test]
fn catch_up_selects_the_farthest_retained_frame_inside_the_tracking_gap() {
    let mut samples = samples();
    for i in 5..12 {
        samples.push(Sample {
            frame_id: 100 + i,
            capture_time_ns: 1_000_000_000 + i * 33_333_333,
            image: crate::job::Image {
                path: format!("{i}.png").into(),
                sha256: "a".repeat(64),
            },
        });
    }
    let mut queue = Queue::new(Options::new("catch-up", 12, "ready").unwrap());
    queue.advance(&samples, 0);
    assert_eq!(queue.select(&samples, None, 0), Some(0));
    queue.advance(&samples, 1_000_000_000);
    assert_eq!(
        queue.select(&samples, Some(samples[0].capture_time_ns), 1_000_000_000),
        Some(6)
    );
    assert_eq!(queue.dropped.len(), 5);
    assert!(
        queue
            .dropped
            .iter()
            .all(|d| d["reason"] == "catch_up_stride")
    );
    assert_eq!(
        queue.select(&samples, Some(samples[6].capture_time_ns), 1_000_000_000),
        Some(11)
    );
    assert!(queue.pending.is_empty());
}
#[test]
fn missing_history_and_missing_anchor_choose_latest_for_recovery() {
    let samples = samples();
    let mut queue = Queue::new(Options::new("catch-up", 2, "ready").unwrap());
    queue.advance(&samples, 1_000_000_000);
    assert_eq!(queue.select(&samples, Some(0), 1_000_000_000), Some(4));
    assert_eq!(queue.dropped.last().unwrap()["reason"], "history_gap");
    let mut queue = Queue::new(Options::new("catch-up", 2, "ready").unwrap());
    queue.advance(&samples, 1_000_000_000);
    assert_eq!(queue.select(&samples, None, 1_000_000_000), Some(4));
    assert_eq!(queue.dropped.last().unwrap()["reason"], "no_anchor");
}
