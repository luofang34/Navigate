#!/usr/bin/env bash
# Full local quality gate. CI runs the same steps; run this before pushing.
set -euo pipefail
cd "$(dirname "$0")"
export RUST_MIN_STACK="${RUST_MIN_STACK:-8388608}"

echo "== structure checks =="
./scripts/check-structure.sh

echo "== publishable crates =="
./scripts/check-publishable-crates.sh

echo "== cargo fmt --check =="
cargo fmt --all --check

echo "== cargo clippy =="
cargo clippy --all-targets -- -D warnings
# The workspace run has navigate-onboard without its terrain-camera means.
cargo clippy -p navigate-onboard --features terrain-camera --all-targets -- -D warnings

echo "== bare-metal no_std (core navigation crates) =="
for target in thumbv8m.main-none-eabihf riscv32imac-unknown-none-elf; do
    for crate in \
        navigate-contract \
        navigate-camera \
        navigate-data \
        navigate-geodesy \
        navigate-astrodynamics \
        navigate-fusion \
        navigate-magnav \
        navigate-route \
        navigate-trajectory \
        navigate-guidance \
        navigate-terrain-safety; do
        cargo clippy -p "$crate" --no-default-features --features libm --target "$target" -- -D warnings
    done
done

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

echo "== navigate-fusion Monte Carlo consistency (release) =="
# cargo test runs a small sample in the debug build; this runs the full one.
# The names are exact, and the step fails unless all of them ran.
monte_carlo_tests="inertial::tests::baro_consistency::monte_carlo_a_barometric_altitude_improves_the_horizontal_estimate_consistently
inertial::tests::baro_maneuver::monte_carlo_an_altimeter_in_turns_keeps_the_horizontal_estimate_consistent
inertial::tests::star_distribution::monte_carlo_stars_do_not_make_the_horizontal_estimate_worse
inertial::tests::star_distribution::monte_carlo_a_constant_mount_error_needs_its_correlated_declaration
inertial::tests::correlated_gnss::monte_carlo_a_correlated_gnss_error_declared_as_white_makes_the_position_over_confident
inertial::tests::correlated_gnss::monte_carlo_a_declared_correlated_gnss_error_with_a_block_keeps_the_position_nees_in_band
inertial::tests::correlated_gnss::monte_carlo_a_declared_correlated_gnss_error_without_a_block_is_inflated_into_band
inertial::tests::correlated_gnss::monte_carlo_a_correlation_time_declared_three_times_too_short_or_too_long_stays_in_band"
monte_carlo_count=$(echo "$monte_carlo_tests" | wc -l | tr -d ' ')
# shellcheck disable=SC2086
monte_carlo_out=$(cargo test --release -p navigate-fusion --lib -- --exact $monte_carlo_tests 2>&1) \
    || { echo "$monte_carlo_out"; exit 1; }
echo "$monte_carlo_out" | grep "test result"
echo "$monte_carlo_out" | grep -q "test result: ok. $monte_carlo_count passed" \
    || { echo "expected $monte_carlo_count Monte Carlo tests to run"; exit 1; }

echo "== navigate-onboard Monte Carlo consistency (release) =="
onboard_tests="reference::tests::correlated_gnss::monte_carlo_a_declared_correlated_gnss_error_at_10_hz_keeps_the_position_nees_in_band
reference::tests::correlated_gnss::monte_carlo_a_correlated_gnss_error_declared_as_white_at_10_hz_is_over_confident
reference::terrain::tests::anchors::monte_carlo_a_cruise_across_cells_with_bound_map_errors_keeps_the_position_nees_in_band
reference::terrain::tests::anchors::monte_carlo_a_loiter_in_one_cell_with_its_bound_map_error_keeps_the_position_nees_in_band"
onboard_count=$(echo "$onboard_tests" | wc -l | tr -d ' ')
# shellcheck disable=SC2086
onboard_out=$(cargo test --release -p navigate-onboard --features terrain-camera --lib -- --exact $onboard_tests 2>&1) \
    || { echo "$onboard_out"; exit 1; }
echo "$onboard_out" | grep "test result"
echo "$onboard_out" | grep -q "test result: ok. $onboard_count passed" \
    || { echo "expected $onboard_count onboard Monte Carlo tests to run"; exit 1; }

echo "== cargo test =="
cargo test --all-targets
cargo test -p navigate-onboard --features terrain-camera

echo "== cargo doc =="
RUSTDOCFLAGS="-D missing_docs -D rustdoc::broken_intra_doc_links" \
    cargo doc --no-deps --workspace
RUSTDOCFLAGS="-D missing_docs -D rustdoc::broken_intra_doc_links" \
    cargo doc --no-deps -p navigate-onboard --features terrain-camera

echo "== wasm32 (visual locator, matchers, and segmentation) =="
cargo clippy -p navigate-locate-web --target wasm32-unknown-unknown -- -D warnings
cargo clippy -p navigate-match --target wasm32-unknown-unknown -- -D warnings
cargo clippy -p navigate-match --no-default-features --features webgpu --target wasm32-unknown-unknown -- -D warnings
cargo clippy -p navigate-data --target wasm32-unknown-unknown -- -D warnings

echo "== cargo build --release =="
cargo build --release --workspace

echo "ALL GATES GREEN"
