//! External command-line tools that the pipeline runs as child processes.

use crate::PipelineError;
use std::process::Command;

/// Run an external tool (`ffmpeg`, `ffprobe`, `curl`, `magick`) and return its stdout bytes.
///
/// # Errors
///
/// Returns [`PipelineError::Tool`] when execution fails or exits non-zero.
pub fn run_tool(tool: &str, args: &[&str]) -> Result<Vec<u8>, PipelineError> {
    let output = Command::new(tool)
        .args(args)
        .output()
        .map_err(|e| PipelineError::Tool {
            tool: tool.to_owned(),
            message: e.to_string(),
        })?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
        return Err(PipelineError::Tool {
            tool: tool.to_owned(),
            message: format!("status {}: {stderr}", output.status),
        });
    }
    Ok(output.stdout)
}
