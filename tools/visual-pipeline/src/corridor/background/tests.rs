use super::BackgroundWorker;
use std::{sync::mpsc, time::Duration};

const SHUTDOWN: Duration = Duration::from_secs(5);

#[test]
fn results_return_in_submission_order() {
    let worker = BackgroundWorker::spawn("test-order", SHUTDOWN, |job: u32| job * 2).unwrap();
    for job in 0..4 {
        worker.submit(job).unwrap();
    }
    let results: Vec<u32> = (0..4).map(|_| worker.recv_blocking().unwrap()).collect();
    assert_eq!(results, [0, 2, 4, 6]);
}

#[test]
fn polling_does_not_wait_for_a_blocked_job() {
    let (gate_tx, gate_rx) = mpsc::channel::<()>();
    let (started_tx, started_rx) = mpsc::channel::<()>();
    let worker = BackgroundWorker::spawn("test-no-stall", SHUTDOWN, move |job: u32| {
        started_tx.send(()).unwrap();
        gate_rx.recv().unwrap();
        job + 1
    })
    .unwrap();
    worker.submit(41).unwrap();
    started_rx.recv().unwrap();
    let mut frames = 0;
    for _ in 0..100 {
        assert!(worker.try_recv().unwrap().is_none());
        frames += 1;
    }
    assert_eq!(frames, 100);
    gate_tx.send(()).unwrap();
    assert_eq!(worker.recv_blocking().unwrap(), 42);
}

#[test]
fn drop_joins_an_idle_worker() {
    let (alive_tx, alive_rx) = mpsc::channel::<()>();
    let worker = BackgroundWorker::spawn("test-join", SHUTDOWN, move |job: u32| {
        alive_tx.send(()).unwrap();
        job
    })
    .unwrap();
    worker.submit(1).unwrap();
    alive_rx.recv().unwrap();
    drop(worker);
    // The handler owned `alive_tx`; a closed channel shows the thread has exited.
    assert!(alive_rx.recv().is_err());
}

#[test]
fn drop_returns_when_a_job_outlives_the_shutdown_timeout() {
    let (gate_tx, gate_rx) = mpsc::channel::<()>();
    let (started_tx, started_rx) = mpsc::channel::<()>();
    let worker =
        BackgroundWorker::spawn("test-timeout", Duration::from_millis(20), move |_: u32| {
            started_tx.send(()).unwrap();
            gate_rx.recv().ok();
        })
        .unwrap();
    worker.submit(0).unwrap();
    started_rx.recv().unwrap();
    drop(worker);
    gate_tx.send(()).unwrap();
}
