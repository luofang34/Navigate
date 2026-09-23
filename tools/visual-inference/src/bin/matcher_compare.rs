//! Compare native matchers on identical image and depth evidence.
#[path = "../comparison.rs"]
mod comparison;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .init();
    comparison::run_blocking()
}
