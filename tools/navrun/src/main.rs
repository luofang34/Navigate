//! Deterministic scripted scenario runner. All diagnostics go through
//! `tracing` to stderr; the exit code is the machine-readable verdict.

use std::process::ExitCode;

use navrun::{ScenarioSummary, run_scenario};

fn main() -> ExitCode {
    if tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .try_init()
        .is_err()
    {
        // Installation only fails when a global subscriber already
        // exists, so this warning has somewhere to go.
        tracing::warn!("global tracing subscriber was already installed");
    }
    match run_scenario() {
        Ok(summary) => report(&summary),
        Err(error) => {
            tracing::error!(error = ?error, "scenario failed");
            ExitCode::FAILURE
        }
    }
}

fn report(summary: &ScenarioSummary) -> ExitCode {
    tracing::info!(
        steps_run = summary.steps_run,
        plan_completed = summary.plan_completed,
        solutions_published = summary.solutions_published,
        observations_rejected = summary.observations_rejected,
        max_lateral_dev_m = summary.max_lateral_dev_m,
        final_quality = ?summary.final_quality,
        final_redundancy = ?summary.final_redundancy,
        final_fault_detection = ?summary.final_fault_detection,
        "scenario complete"
    );
    if summary.plan_completed {
        ExitCode::SUCCESS
    } else {
        tracing::error!("plan did not complete within the step budget");
        ExitCode::FAILURE
    }
}
