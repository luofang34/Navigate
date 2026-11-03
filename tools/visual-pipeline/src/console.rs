//! Product output lines on standard output.

use crate::PipelineError;
use std::{io::Write, path::PathBuf};

/// Write one line to standard output without invoking `println!`.
///
/// # Errors
///
/// Returns [`PipelineError::Io`] when writing to stdout fails.
pub fn write_stdout_line(line: &str) -> Result<(), PipelineError> {
    let mut stdout = std::io::stdout().lock();
    writeln!(stdout, "{line}").map_err(|source| PipelineError::Io {
        path: PathBuf::from("stdout"),
        source,
    })
}
