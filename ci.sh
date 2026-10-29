#!/usr/bin/env bash
# Full local quality gate. CI runs the same steps; run this before pushing.
set -euo pipefail
cd "$(dirname "$0")"

echo "== structure checks =="
./scripts/check-structure.sh

echo "== publishable crates =="
./scripts/check-publishable-crates.sh

echo "== cargo fmt --check =="
cargo fmt --all --check

echo "== cargo clippy =="
cargo clippy --all-targets -- -D warnings

echo "== wasm32 (navigate-celestial) =="
cargo clippy -p navigate-celestial --target wasm32-unknown-unknown -- -D warnings
cargo clippy -p navigate-celestial --no-default-features --target wasm32-unknown-unknown -- -D warnings

echo "== flamsteed (wasm32, no_std targets, crates.io package) =="
cargo clippy -p flamsteed --target wasm32-unknown-unknown -- -D warnings
cargo clippy -p flamsteed --no-default-features --features libm --all-targets -- -D warnings
for features in libm libm,magnitude-5 libm,all-stars std std,magnitude-5; do
    cargo test -p flamsteed --no-default-features --features "$features"
done
for target in thumbv8m.main-none-eabihf riscv32imac-unknown-none-elf thumbv6m-none-eabi; do
    cargo clippy -p flamsteed --no-default-features --features libm --target "$target" -- -D warnings
done
for harness in qemu-riscv qemu-arm; do
    cargo fmt --manifest-path "crates/flamsteed/$harness/Cargo.toml" --check
    (cd "crates/flamsteed/$harness" && cargo clippy --release --locked -- -D warnings)
done
# The QEMU runs and the nightly feature are in the weekly hardware workflow
# (.github/workflows/flamsteed-hardware.yml); run the script directly when
# flamsteed changes.
cargo package -p flamsteed --allow-dirty

echo "== eclipse-canon without the standard library =="
for target in thumbv8m.main-none-eabihf riscv32imac-unknown-none-elf wasm32-unknown-unknown; do
    cargo clippy -p eclipse-canon --target "$target" -- -D warnings
done
cargo package -p eclipse-canon --allow-dirty

echo "== navigate-celestial without long-range (1800 to 2200 series) =="
cargo clippy -p navigate-celestial --no-default-features --all-targets -- -D warnings
cargo test -p navigate-celestial --no-default-features

echo "== navigate-celestial against every eclipse of the Canon (release) =="
# cargo test checks a sample in the debug build; this checks every eclipse.
cargo test --release -p navigate-celestial --test eclipse_canon

echo "== cargo test =="
cargo test --all-targets

echo "== cargo doc =="
RUSTDOCFLAGS="-D missing_docs -D rustdoc::broken_intra_doc_links" \
    cargo doc --no-deps --workspace

echo "== visual tools =="
./scripts/check-visual-tools.sh

echo "== visual pipeline and segmentation (Rust) =="
cargo test -p navigate-segment -p navigate-visual-pipeline
cargo clippy -p navigate-locate-web --target wasm32-unknown-unknown -- -D warnings

echo "== native tools =="
./scripts/check-native-tools.sh

echo "== cargo build --release =="
cargo build --release --workspace

echo "ALL GATES GREEN"
