//! The deterministic end-to-end scenario: a scripted three-waypoint
//! flight driven through fusion, plan execution, guidance, and the
//! EGPWS availability seam.
//!
//! Nothing here reads a clock or a random source (ADR-0002): simulated
//! time advances one second per step as plain
//! [`navigate_contract::MonotonicNanos`] arithmetic, and measurement
//! "noise" is a fixed four-entry offset table, so every run publishes
//! field-identical solutions.

use navigate_contract::{
    ClockDomainId, FaultDetection, FlightPlan, GeodeticPosition, GuidanceSetpoint, MonotonicNanos,
    NavigationSolution, ObservationStamp, PlanRole, PlanValidationError, Redundancy, SensorClass,
    SolutionQuality, SourceComposition, SourceEpoch, SourceId, SymmetricCov3, Waypoint,
    WrappingSequence,
};
use navigate_egpws::{EgpwsAssessment, EgpwsConfig, EgpwsUnavailable, assess};
use navigate_fpl::{ExecutionConfig, PlanExecution, SequenceEvent};
use navigate_fusion::{
    FusionConfig, IngestOutcome, NavigationFilter, Observation, ObservationValue,
};
use navigate_geodesy::{
    GeodesyError, LocalTangentPlane, NedOffset, distance_m, initial_bearing_rad,
};
use navigate_guidance::{GuidanceConfig, GuidanceRefusal, guide};

/// Ground speed of the truth model in meters per second; with one-second
/// steps, also the distance flown per step.
const GROUND_SPEED_MPS: f64 = 30.0;

/// Simulated time advance per step: one second.
const STEP_NANOS: u64 = 1_000_000_000;

/// Step budget: the roughly six-kilometer mission takes about two
/// hundred steps at [`GROUND_SPEED_MPS`], so this bound only catches a
/// scenario that stalls.
const MAX_STEPS: u32 = 300;

/// Fixed cyclic north/east measurement offsets in meters — the
/// deterministic stand-in for GNSS noise. Every entry is under 3 m in
/// magnitude, which bounds the lateral deviation the run can exhibit.
const FIX_OFFSETS_NE_M: [(f64, f64); 4] = [(1.5, -0.5), (-1.0, 2.0), (0.5, 1.0), (-2.0, -1.5)];

/// Per-axis GNSS fix variance in m² (3 m 1-sigma), honest about the
/// offset table's magnitude so the innovation gate stays quiet.
const FIX_VARIANCE_M2: f64 = 9.0;

/// The single clock domain of the run: truth, stamps, and admission all
/// read the same simulated clock.
const CLOCK: ClockDomainId = ClockDomainId::new(0);

/// The synthesized GNSS receiver.
const GNSS_SOURCE: SourceId = SourceId::new(1);

/// Cruise altitude above the WGS84 ellipsoid in meters.
const CRUISE_ALTITUDE_M: f64 = 500.0;

/// Typed outcome of one scenario run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScenarioSummary {
    /// Steps executed at 1 Hz, one synthesized observation each.
    pub steps_run: u32,
    /// Whether the final waypoint captured within the step budget.
    pub plan_completed: bool,
    /// Solutions the filter published across the run.
    pub solutions_published: u64,
    /// Observations the filter refused admission.
    pub observations_rejected: u64,
    /// Largest absolute cross-track deviation any guidance command
    /// reported, in meters.
    pub max_lateral_dev_m: f64,
    /// Quality classification of the final published solution.
    pub final_quality: SolutionQuality,
    /// Redundancy of the final published solution.
    pub final_redundancy: Redundancy,
    /// Fault-detection statement of the final published solution.
    pub final_fault_detection: FaultDetection,
}

/// Why the scripted scenario could not finish.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ScenarioError {
    /// The scripted flight plan failed structural validation.
    #[error(transparent)]
    InvalidPlan(#[from] PlanValidationError),
    /// A truth-model tangent plane could not be anchored.
    #[error("truth geometry failed at step {step}")]
    TruthGeometry {
        /// Step at which the anchoring failed.
        step: u32,
        /// The geodesy constructor's refusal.
        #[source]
        source: GeodesyError,
    },
    /// Guidance refused a solution the scenario expected it to accept.
    #[error("guidance refused at step {step}")]
    GuidanceRefused {
        /// Step at which guidance refused.
        step: u32,
        /// The typed refusal.
        #[source]
        source: GuidanceRefusal,
    },
    /// The EGPWS seam did not refuse with
    /// [`EgpwsUnavailable::NoDatabase`] despite no terrain database
    /// being bound.
    #[error("terrain seam breached at step {step}: expected NoDatabase, got {outcome:?}")]
    TerrainSeamBreached {
        /// Step at which the seam misbehaved.
        step: u32,
        /// What `assess` returned instead.
        outcome: Result<EgpwsAssessment, EgpwsUnavailable>,
    },
    /// The filter never published a solution.
    #[error("no navigation solution was published in {steps} steps")]
    NoSolutionPublished {
        /// Steps run without a publication.
        steps: u32,
    },
}

/// Runs the scripted flight to completion or the step budget and
/// returns its typed summary.
///
/// # Errors
///
/// Returns a [`ScenarioError`] when the plan fails validation, truth
/// geometry cannot be anchored, guidance refuses a solution, the EGPWS
/// seam answers anything but `NoDatabase`, or no solution is ever
/// published. An incomplete plan is not an error: the summary reports
/// `plan_completed: false` and the caller judges it.
pub fn run_scenario() -> Result<ScenarioSummary, ScenarioError> {
    let mut run = ScenarioRun::new()?;
    let mut steps_run = 0_u32;
    for step in 0..MAX_STEPS {
        run.step(step)?;
        steps_run = step.wrapping_add(1);
        if run.plan_completed {
            break;
        }
    }
    run.into_summary(steps_run)
}

/// Mutable state of one scenario run: the systems under test plus the
/// truth model driving them.
struct ScenarioRun {
    filter: NavigationFilter,
    execution: PlanExecution,
    truth: GeodeticPosition,
    next_sequence: WrappingSequence,
    now: MonotonicNanos,
    solutions_published: u64,
    observations_rejected: u64,
    max_lateral_dev_m: f64,
    last_solution: Option<NavigationSolution>,
    plan_completed: bool,
}

impl ScenarioRun {
    /// Builds the run: truth starts at waypoint 0's position.
    fn new() -> Result<Self, ScenarioError> {
        let waypoints = mission_waypoints();
        let truth = waypoints[0].position;
        let plan = FlightPlan::new("navrun-mission".into(), PlanRole::Mission, waypoints.into());
        let execution = PlanExecution::new(plan, ExecutionConfig::default())?;
        Ok(Self {
            filter: NavigationFilter::new(FusionConfig::default(), CLOCK),
            execution,
            truth,
            next_sequence: WrappingSequence::new(0),
            now: MonotonicNanos::from_nanos(STEP_NANOS),
            solutions_published: 0,
            observations_rejected: 0,
            max_lateral_dev_m: 0.0,
            last_solution: None,
            plan_completed: false,
        })
    }

    /// One 1 Hz step: synthesize a fix, ingest it, tick for a solution,
    /// sequence the plan on the solution position, guide toward the
    /// active leg, probe the terrain seam, then fly the truth point and
    /// advance simulated time for the next step.
    fn step(&mut self, step: u32) -> Result<(), ScenarioError> {
        let observation = self.synthesize_fix(step)?;
        self.next_sequence = self.next_sequence.next();
        match self.filter.ingest(&observation, self.now) {
            IngestOutcome::Accepted => {}
            IngestOutcome::Rejected(_) => {
                self.observations_rejected = self.observations_rejected.wrapping_add(1);
            }
        }
        if let Some(solution) = self.filter.tick(self.now) {
            self.solutions_published = self.solutions_published.wrapping_add(1);
            if self.execution.advance(&solution.position) == SequenceEvent::PlanComplete {
                self.plan_completed = true;
            }
            self.guide_active_leg(&solution, step)?;
            check_terrain_seam(&solution, self.now, step)?;
            self.last_solution = Some(solution);
        }
        self.propagate_truth(step)?;
        self.now = MonotonicNanos::from_nanos(self.now.as_nanos().wrapping_add(STEP_NANOS));
        Ok(())
    }

    /// Truth plus the step's cyclic offset, stamped and declared as a
    /// GNSS position fix.
    fn synthesize_fix(&self, step: u32) -> Result<Observation, ScenarioError> {
        let plane = tangent_plane(&self.truth, step)?;
        let (north_m, east_m) = FIX_OFFSETS_NE_M[step as usize % FIX_OFFSETS_NE_M.len()];
        let measured = plane.from_ned(&NedOffset::new(north_m, east_m, 0.0));
        let stamp = ObservationStamp::new(
            GNSS_SOURCE,
            SourceEpoch::new(0),
            self.next_sequence,
            self.now,
            CLOCK,
        );
        Ok(Observation::new(
            stamp,
            ObservationValue::PositionFix {
                position: measured,
                covariance: SymmetricCov3::from_diagonal(
                    FIX_VARIANCE_M2,
                    FIX_VARIANCE_M2,
                    FIX_VARIANCE_M2,
                ),
            },
            SourceComposition::of(SensorClass::Gnss),
        ))
    }

    /// Guides toward the active leg and records the largest lateral
    /// deviation. Inert once the plan is complete.
    fn guide_active_leg(
        &mut self,
        solution: &NavigationSolution,
        step: u32,
    ) -> Result<(), ScenarioError> {
        let Some(leg) = self.execution.active_leg() else {
            return Ok(());
        };
        let command = guide(
            solution,
            leg.from.map(|waypoint| &waypoint.position),
            leg.to,
            self.now,
            CLOCK,
            &GuidanceConfig::default(),
        )
        .map_err(|source| ScenarioError::GuidanceRefused { step, source })?;
        if let GuidanceSetpoint::DeviationTracking { lateral_m, .. } = command.setpoint {
            self.max_lateral_dev_m = self.max_lateral_dev_m.max(lateral_m.abs());
        }
        Ok(())
    }

    /// Flies the truth point one second along the great circle toward
    /// the active waypoint, never overshooting it. Inert once the plan
    /// is complete.
    fn propagate_truth(&mut self, step: u32) -> Result<(), ScenarioError> {
        let Some(leg) = self.execution.active_leg() else {
            return Ok(());
        };
        let target = leg.to.position;
        let travel_m = distance_m(&self.truth, &target).min(GROUND_SPEED_MPS);
        let bearing_rad = initial_bearing_rad(&self.truth, &target);
        let plane = tangent_plane(&self.truth, step)?;
        self.truth = plane.from_ned(&NedOffset::new(
            travel_m * bearing_rad.cos(),
            travel_m * bearing_rad.sin(),
            0.0,
        ));
        Ok(())
    }

    /// Folds the run into its summary; the final solution supplies the
    /// integrity fields.
    fn into_summary(self, steps_run: u32) -> Result<ScenarioSummary, ScenarioError> {
        let last = self
            .last_solution
            .ok_or(ScenarioError::NoSolutionPublished { steps: steps_run })?;
        Ok(ScenarioSummary {
            steps_run,
            plan_completed: self.plan_completed,
            solutions_published: self.solutions_published,
            observations_rejected: self.observations_rejected,
            max_lateral_dev_m: self.max_lateral_dev_m,
            final_quality: last.integrity.quality,
            final_redundancy: last.integrity.redundancy,
            final_fault_detection: last.integrity.fault_detection,
        })
    }
}

/// The honest-seam check: with no database bound, [`assess`] must
/// refuse with [`EgpwsUnavailable::NoDatabase`] rather than invent a
/// terrain clearance (ADR-0004).
fn check_terrain_seam(
    solution: &NavigationSolution,
    now: MonotonicNanos,
    step: u32,
) -> Result<(), ScenarioError> {
    match assess(solution, now, CLOCK, None, &EgpwsConfig::default()) {
        Err(EgpwsUnavailable::NoDatabase) => Ok(()),
        outcome => Err(ScenarioError::TerrainSeamBreached { step, outcome }),
    }
}

/// Three eastbound waypoints along 47° N, roughly 3 km apart — a
/// straight-ish track, so leg transitions carry no heading transient.
fn mission_waypoints() -> [Waypoint; 3] {
    [
        waypoint("W0", 47.0, 8.00),
        waypoint("W1", 47.0, 8.04),
        waypoint("W2", 47.0, 8.08),
    ]
}

fn waypoint(ident: &str, latitude_deg: f64, longitude_deg: f64) -> Waypoint {
    Waypoint::new(
        ident.to_owned(),
        GeodeticPosition::new(
            latitude_deg.to_radians(),
            longitude_deg.to_radians(),
            CRUISE_ALTITUDE_M,
        ),
    )
}

/// Anchors a tangent plane at `origin`, naming the step on refusal.
fn tangent_plane(origin: &GeodeticPosition, step: u32) -> Result<LocalTangentPlane, ScenarioError> {
    LocalTangentPlane::new(*origin).map_err(|source| ScenarioError::TruthGeometry { step, source })
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::panic)]

    use super::*;

    #[test]
    fn identical_runs_publish_identical_summaries() {
        let first = run_scenario().expect("scenario runs");
        let second = run_scenario().expect("scenario runs");
        assert_eq!(first, second, "replay determinism is a property");
    }

    #[test]
    fn synthesized_fixes_stay_within_the_offset_bound() {
        let run = ScenarioRun::new().expect("scenario constructs");
        for step in 0..8 {
            let observation = run.synthesize_fix(step).expect("fix synthesizes");
            let ObservationValue::PositionFix { position, .. } = observation.value else {
                panic!("scenario synthesizes position fixes");
            };
            let displacement_m = distance_m(&run.truth, &position);
            assert!(
                displacement_m < 3.0,
                "step {step}: fix displaced {displacement_m} m from truth"
            );
        }
    }

    #[test]
    fn truth_flies_the_ground_speed_toward_the_active_waypoint() {
        let mut run = ScenarioRun::new().expect("scenario constructs");
        // Truth starts at W0, so the first advance captures it and the
        // active waypoint becomes W1, about 3 km east.
        let start = run.truth;
        assert_eq!(
            run.execution.advance(&start),
            SequenceEvent::LegAdvanced { to_index: 1 }
        );
        let target = run.execution.active_leg().expect("active leg").to.position;
        let before_m = distance_m(&run.truth, &target);
        run.propagate_truth(0).expect("truth propagates");
        let after_m = distance_m(&run.truth, &target);
        // The step is 30 tangent-plane meters on the ellipsoid while
        // `distance_m` measures on the mean-radius sphere; the two
        // metrics diverge by up to ~0.5% (see `navigate_geodesy`).
        assert!(
            (before_m - after_m - GROUND_SPEED_MPS).abs() < GROUND_SPEED_MPS * 0.005,
            "one step closes about {GROUND_SPEED_MPS} m: {before_m} -> {after_m}"
        );
    }
}
