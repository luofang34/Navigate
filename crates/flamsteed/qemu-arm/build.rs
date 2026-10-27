//! Put `memory.x` on the linker search path.

use std::{io::Write as _, path::PathBuf};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out = PathBuf::from(std::env::var("OUT_DIR")?);
    std::fs::copy("memory.x", out.join("memory.x"))?;
    let mut stdout = std::io::stdout();
    writeln!(stdout, "cargo:rustc-link-search={}", out.display())?;
    writeln!(stdout, "cargo:rerun-if-changed=memory.x")?;
    Ok(())
}
