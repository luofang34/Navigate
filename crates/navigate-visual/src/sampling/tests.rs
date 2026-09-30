use super::*;
fn ms(value: u64) -> Duration {
    Duration::from_millis(value)
}
fn sampler() -> Result<AdaptiveSampler, VisualError> {
    AdaptiveSampler::new(
        SamplingConfig {
            minimum_interval: ms(20),
            maximum_interval: ms(500),
            utilization: 0.5,
        },
        ms(50),
    )
}
#[test]
fn serialized_admission_has_no_catch_up_burst() -> Result<(), VisualError> {
    let mut s = sampler()?;
    assert!(s.begin(ms(0))?);
    assert!(!s.begin(ms(100))?);
    s.complete(ms(50))?;
    assert!(s.begin(ms(1000))?);
    s.complete(ms(50))?;
    assert!(!s.begin(ms(1000))?);
    assert!(!s.begin(ms(1099))?);
    assert!(s.begin(ms(1100))?);
    Ok(())
}
#[test]
fn adapts_to_contention_and_reports_impossible_deadline() -> Result<(), VisualError> {
    let mut s = sampler()?;
    assert!(s.begin(ms(0))?);
    let slow = s.complete(ms(400))?;
    assert_eq!(slow.interval, ms(800));
    assert!(slow.deadline_unattainable);
    assert!(!s.begin(ms(500))?);
    s.set_utilization(1.0)?;
    assert!(s.begin(ms(800))?);
    let recovered = s.complete(ms(50))?;
    assert!(recovered.estimated_cost < slow.estimated_cost);
    assert!(recovered.estimated_cost > ms(50));
    assert!(!recovered.deadline_unattainable);
    Ok(())
}
#[test]
fn live_host_budget_can_pause_resume_and_release_more_time() -> Result<(), VisualError> {
    let mut s = sampler()?;
    s.set_utilization(0.0)?;
    assert!(!s.begin(ms(0))?);
    assert!(s.status().paused);
    s.set_utilization(0.25)?;
    assert!(s.begin(ms(10))?);
    s.complete(ms(50))?;
    assert!(!s.begin(ms(100))?);
    s.set_utilization(1.0)?;
    assert!(s.begin(ms(100))?);
    assert!(s.begin(ms(99)).is_err());
    Ok(())
}
#[test]
fn invalid_inputs_and_unmatched_completion_fail() -> Result<(), VisualError> {
    let mut s = sampler()?;
    assert!(s.complete(ms(50)).is_err());
    for fraction in [f64::NAN, -0.1, 1.1, f64::INFINITY] {
        assert!(s.set_utilization(fraction).is_err());
    }
    assert!(
        AdaptiveSampler::new(
            SamplingConfig {
                minimum_interval: ms(0),
                maximum_interval: ms(1),
                utilization: 0.5,
            },
            ms(1)
        )
        .is_err()
    );
    Ok(())
}
