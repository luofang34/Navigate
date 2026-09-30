use super::*;

fn ms(value: u64) -> Duration {
    Duration::from_millis(value)
}
fn frame(now: u64) -> FrameStamp {
    FrameStamp {
        sequence: now,
        capture_time_ns: now * 1_000_000,
    }
}
fn profile(id: u64, device: u64, cost: u64) -> ExecutionProfile {
    ExecutionProfile {
        id: ProfileId(id),
        device: DeviceId(device),
        work: WorkKind::MapCheck,
        initial_cost: ms(cost),
        initial_call: ms(cost),
        peak_bytes: 100,
    }
}
fn grant(device: u64, until: u64, call: u64) -> ResourceGrant {
    ResourceGrant {
        device: DeviceId(device),
        until: ms(until),
        maximum_call: ms(call),
        memory_bytes: 100,
    }
}
fn demand(due: Option<u64>) -> WorkDemand {
    WorkDemand {
        work: WorkKind::MapCheck,
        due_by: due.map(ms),
    }
}
fn controller(profiles: Vec<ExecutionProfile>) -> VisualController {
    VisualController::new(
        ControllerConfig {
            maximum_capture_age: ms(100),
            maximum_result_age: ms(2000),
            reserve: ms(50),
            minimum_interval: ms(20),
        },
        profiles,
    )
    .unwrap_or_else(|error| panic!("{error}"))
}
fn started(decision: Admission) -> WorkTicket {
    match decision {
        Admission::Start { ticket, .. } => ticket,
        other => panic!("expected work, got {other:?}"),
    }
}

#[test]
fn sub_hertz_checks_do_not_require_intermediate_frames() {
    let mut policy = controller(vec![profile(1, 0, 700)]);
    let grants = [grant(0, 6000, 700)];
    assert_eq!(
        policy
            .consider(ms(0), frame(0), demand(Some(5000)), &grants)
            .expect("request"),
        Admission::NotDue(ms(4250))
    );
    let ticket = started(
        policy
            .consider(ms(4250), frame(4250), demand(Some(5000)), &grants)
            .expect("request"),
    );
    assert!(
        policy
            .complete(
                ticket,
                ms(4950),
                WorkCost {
                    total: ms(700),
                    longest_call: ms(700)
                }
            )
            .expect("completion")
    );
    assert_eq!(
        policy
            .consider(
                ms(5000),
                frame(5000),
                demand(Some(10000)),
                &[grant(0, 11000, 700)]
            )
            .expect("request"),
        Admission::NotDue(ms(9250))
    );
}

#[test]
fn reserved_gpu_selects_a_validated_cpu_profile() {
    let mut policy = controller(vec![profile(1, 1, 100), profile(2, 0, 300)]);
    let grants = [grant(1, 1000, 20), grant(0, 1000, 500)];
    let ticket = started(
        policy
            .consider(ms(0), frame(0), demand(None), &grants)
            .expect("request"),
    );
    assert_eq!(ticket.profile(), ProfileId(2));
}

#[test]
fn revoked_resources_never_queue_work() {
    let mut policy = controller(vec![profile(1, 1, 100)]);
    assert_eq!(
        policy
            .consider(ms(0), frame(0), demand(None), &[])
            .expect("request"),
        Admission::NoResources
    );
    let ticket = started(
        policy
            .consider(ms(1000), frame(1000), demand(None), &[grant(1, 2000, 100)])
            .expect("request"),
    );
    assert_eq!(ticket.observation(), frame(1000));
    assert_eq!(
        policy
            .consider(ms(1001), frame(1001), demand(None), &[])
            .expect("request"),
        Admission::Busy
    );
}

#[test]
fn failed_work_still_raises_cost_and_non_preemptible_budget() {
    let mut policy = controller(vec![profile(1, 1, 100)]);
    let ticket = started(
        policy
            .consider(ms(0), frame(0), demand(None), &[grant(1, 3000, 1000)])
            .expect("request"),
    );
    assert!(
        policy
            .complete(
                ticket,
                ms(800),
                WorkCost {
                    total: ms(800),
                    longest_call: ms(600)
                }
            )
            .expect("failed job accounted")
    );
    assert_eq!(
        policy
            .consider(ms(801), frame(801), demand(None), &[grant(1, 3000, 500)])
            .expect("request"),
        Admission::NoResources
    );
    assert_eq!(
        policy
            .consider(ms(802), frame(802), demand(None), &[grant(1, 1500, 600)])
            .expect("request"),
        Admission::NoResources
    );
}

#[test]
fn stale_frames_and_short_grants_are_rejected() {
    let mut policy = controller(vec![profile(1, 0, 100)]);
    assert_eq!(
        policy
            .consider(ms(101), frame(0), demand(None), &[grant(0, 1000, 100)])
            .expect("request"),
        Admission::StaleObservation
    );
    assert_eq!(
        policy
            .consider(ms(102), frame(102), demand(None), &[grant(0, 250, 100)])
            .expect("request"),
        Admission::NoResources
    );
    let mut memory = grant(0, 1000, 100);
    memory.memory_bytes = 99;
    assert_eq!(
        policy
            .consider(ms(103), frame(103), demand(None), &[memory])
            .expect("request"),
        Admission::NoResources
    );
}

#[test]
fn a_wrong_ticket_cannot_clear_active_work() {
    let mut policy = controller(vec![profile(1, 0, 100)]);
    let ticket = started(
        policy
            .consider(ms(0), frame(0), demand(None), &[grant(0, 1000, 100)])
            .expect("request"),
    );
    let mut wrong = ticket;
    wrong.generation = wrong.generation.wrapping_add(1);
    let cost = WorkCost {
        total: ms(100),
        longest_call: ms(100),
    };
    assert_eq!(
        policy.complete(wrong, ms(100), cost),
        Err(ControllerError::CompletionMismatch)
    );
    assert_eq!(
        policy
            .consider(ms(100), frame(100), demand(None), &[])
            .expect("request"),
        Admission::Busy
    );
    assert!(policy.complete(ticket, ms(100), cost).expect("completion"));
}

#[test]
fn late_results_do_not_gain_fresh_timestamps() {
    let mut policy = controller(vec![profile(1, 0, 100)]);
    let ticket = started(
        policy
            .consider(ms(0), frame(0), demand(None), &[grant(0, 1000, 100)])
            .expect("request"),
    );
    assert!(
        !policy
            .complete(
                ticket,
                ms(2100),
                WorkCost {
                    total: ms(2100),
                    longest_call: ms(2100)
                }
            )
            .expect("late completion")
    );
    assert_eq!(ticket.observation().capture_time_ns, 0);
}

#[test]
fn tracking_cost_does_not_replace_map_check_cost() {
    let mut tracking = profile(2, 0, 10);
    tracking.work = WorkKind::Tracking;
    let mut policy = controller(vec![profile(1, 0, 700), tracking]);
    assert_eq!(
        policy
            .consider(ms(0), frame(0), demand(None), &[grant(0, 500, 700)])
            .expect("request"),
        Admission::NoResources
    );
}
