//! Deterministic scripted scenario runner exercising the Navigate
//! crates end to end: fusion, plan execution, guidance, and the EGPWS
//! availability seam, with no clocks and no randomness (ADR-0002).

pub mod scenario;

pub use scenario::{ScenarioError, ScenarioSummary, run_scenario};
