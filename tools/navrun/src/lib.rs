//! Deterministic scripted scenario runner exercising the Navigate
//! crates end to end: fusion, route execution, guidance, and the
//! terrain-clearance availability seam, with no clocks and no randomness
//! (ADR-0002).

pub mod scenario;

pub use scenario::{ScenarioError, ScenarioSummary, run_scenario};
