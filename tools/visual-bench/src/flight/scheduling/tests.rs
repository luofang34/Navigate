use super::*;
fn schedule(adaptive: bool) -> Result<Schedule, BenchError> {
    Ok(Schedule {
        sampler: AdaptiveSampler::new(
            SamplingConfig {
                minimum_interval: Duration::from_millis(200),
                maximum_interval: Duration::from_secs(1),
                utilization: 0.5,
            },
            Duration::from_nanos(1),
        )?,
        last: None,
        fixed_interval: Duration::from_millis(200),
        adaptive,
    })
}
#[test]
fn fixed_sampling_reports_overload_without_changing_its_requested_rate() -> Result<(), BenchError> {
    let mut s = schedule(false)?;
    assert!(s.sampler.begin(Duration::ZERO)?);
    let result = s.complete(Duration::from_millis(800))?;
    assert!(result.deadline_unattainable);
    assert_eq!(result.estimated_cost, Duration::from_millis(800));
    assert!(s.sampler.begin(Duration::from_millis(200))?);
    Ok(())
}
#[test]
fn adaptive_sampling_uses_actual_rejected_or_successful_work_cost() -> Result<(), BenchError> {
    let mut s = schedule(true)?;
    assert!(s.sampler.begin(Duration::ZERO)?);
    let result = s.complete(Duration::from_millis(800))?;
    assert!(result.deadline_unattainable);
    assert_eq!(result.interval, Duration::from_millis(1600));
    assert!(!s.sampler.begin(Duration::from_millis(200))?);
    assert!(s.sampler.begin(Duration::from_millis(1600))?);
    Ok(())
}

#[test]
fn warm_acquisition_is_explicitly_outside_the_paced_tracking_budget() -> Result<(), BenchError> {
    let mut s = schedule(true)?;
    let result = s.finish_work(Duration::from_secs(4), true)?;
    assert_eq!(result.interval, Duration::from_millis(200));
    assert!(s.sampler.begin(Duration::ZERO)?);
    let result = s.finish_work(Duration::from_millis(150), false)?;
    assert_eq!(result.interval, Duration::from_millis(300));
    assert!(s.finish_work(Duration::from_millis(150), false).is_err());
    Ok(())
}

#[test]
fn sub_hertz_rates_remain_valid_and_budget_reductions_can_go_slower() -> Result<(), BenchError> {
    struct Command {
        args: FlightArgs,
    }
    let mut matches = FlightArgs::args(clap::Command::new("flight")).try_get_matches_from([
        "flight",
        "map.json",
        "video.mp4",
        "--prior",
        "prior.json",
        "--output",
        "output.jsonl",
        "--model",
        "model.onnx",
        "--runtime",
        "runtime.dylib",
        "--fps",
        "0.2",
        "--adaptive",
        "--utilization",
        "0.1",
    ])?;
    let command = Command {
        args: FlightArgs::from_matches(&mut matches)?,
    };
    validate(&command.args)?;
    let mut s = Schedule::new(&command.args)?;
    assert!(s.sampler.begin(Duration::ZERO)?);
    let status = s.complete(Duration::from_secs(1))?;
    assert_eq!(status.interval, Duration::from_secs(10));
    assert!(status.deadline_unattainable);
    assert!(!s.sampler.begin(Duration::from_secs(5))?);
    assert!(s.sampler.begin(Duration::from_secs(10))?);
    Ok(())
}
