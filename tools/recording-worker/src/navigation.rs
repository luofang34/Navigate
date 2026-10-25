//! JSON Lines worker for offline media preparation, backed by the shared session library.

use navigate_recording::{Configuration, Observation, Processor};
use std::{
    ffi::OsString,
    io::{BufRead, Write},
    path::PathBuf,
};

pub(crate) fn run_blocking(args: &[OsString]) -> Result<(), Box<dyn std::error::Error>> {
    let [root, config] = args else {
        return Err("navigate-worker requires EVIDENCE_DIRECTORY CONFIGURATION_JSON".into());
    };
    let config: Configuration = serde_json::from_slice(&std::fs::read(config)?)?;
    let mut processor = Processor::new(config)?;
    let input = std::io::stdin().lock();
    let mut output = std::io::stdout().lock();
    for line in input.lines() {
        let line = line?;
        let result = if line == "{\"finish\":true}" {
            serde_json::to_value(processor.results()?)?
        } else if line == "{\"relocate\":true}" {
            serde_json::to_value(processor.relocation_candidates()?)?
        } else if line == "{\"revisits\":true}" {
            serde_json::to_value(processor.revisit_candidates()?)?
        } else if let Some(candidate) = line.strip_prefix("assess ") {
            let observation: Observation = serde_json::from_str(candidate)?;
            serde_json::to_value(processor.assess_blocking(&PathBuf::from(root), &observation)?)?
        } else if let Some(terrain) = line.strip_prefix("terrain ") {
            let terrain: (u64, f64) = serde_json::from_str(terrain)?;
            processor.update_terrain(terrain.0, terrain.1)?;
            serde_json::Value::Null
        } else if let Some(proposal) = line.strip_prefix("propose ") {
            let observation: Observation = serde_json::from_str(proposal)?;
            serde_json::to_value(processor.propose_blocking(&PathBuf::from(root), &observation)?)?
        } else {
            let observation: Observation = serde_json::from_str(&line)?;
            serde_json::to_value(processor.process_blocking(&PathBuf::from(root), observation)?)?
        };
        serde_json::to_writer(&mut output, &result)?;
        output.write_all(b"\n")?;
        output.flush()?;
    }
    Ok(())
}
