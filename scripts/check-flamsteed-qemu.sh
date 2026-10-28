#!/usr/bin/env bash
# Run flamsteed without the standard library in QEMU on two cores, and compare
# each checksum with the host:
# - RV32IMAC with no FPU, as in the RP2350 Hazard3 cores;
# - Arm Cortex-M33 with a single-precision FPU, as in the RP2350 Arm cores.
set -euo pipefail
cd "$(dirname "$0")/.."

host=$(cargo run -q -p flamsteed --example checksum)
fail=0
for target in riscv:qemu-system-riscv32 arm:qemu-system-arm; do
    name=${target%%:*}
    emulator=${target#*:}
    if ! command -v "$emulator" >/dev/null 2>&1; then
        echo "SKIPPED: flamsteed on $name — $emulator is not installed"
        continue
    fi
    output=$(cd "crates/flamsteed/qemu-$name" && timeout 300 cargo run -q --release --locked)
    echo "$output"
    if [ "$host" != "$(grep '^checksum' <<< "$output")" ]; then
        echo "FAIL: flamsteed on $name differs from the host: $host" >&2
        fail=1
    else
        echo "flamsteed on $name matches the host: $host"
    fi
done
# The nightly feature takes the f32 square root from core, which must compile to
# the Cortex-M33 hardware instruction. CI installs this toolchain with
# llvm-tools and the thumbv8m.main-none-eabihf target.
nightly=nightly-2026-08-28
if ! rustup toolchain list 2>/dev/null | grep -q "^$nightly" || ! command -v qemu-system-arm >/dev/null 2>&1; then
    echo "SKIPPED: flamsteed nightly feature — $nightly or qemu-system-arm is not installed"
    exit "$fail"
fi
cargo "+$nightly" clippy -p flamsteed --no-default-features --features libm,nightly \
    --target thumbv8m.main-none-eabihf -- -D warnings
output=$(cd crates/flamsteed/qemu-arm && timeout 300 cargo "+$nightly" run -q --release --locked --features nightly)
echo "$output"
if [ "$host" != "$(grep '^checksum' <<< "$output")" ]; then
    echo "FAIL: flamsteed on arm with the nightly feature differs from the host: $host" >&2
    fail=1
fi
sysroot=$(rustc "+$nightly" --print sysroot)
objdump=$(find "$sysroot/lib/rustlib" -name llvm-objdump -type f | head -1)
binary=crates/flamsteed/qemu-arm/target/thumbv8m.main-none-eabihf/release/flamsteed-qemu-arm
# grep -c reads all the input, so the disassembler does not stop on a closed pipe.
roots=0
if [ -n "$objdump" ]; then
    roots=$("$objdump" -d --mattr=+fp-armv8d16sp "$binary" | grep -c 'vsqrt.f32' || true)
fi
if [ "$roots" -eq 0 ]; then
    echo "FAIL: the nightly feature does not give the hardware vsqrt.f32 on the Cortex-M33" >&2
    fail=1
else
    echo "flamsteed nightly feature uses vsqrt.f32 on the Cortex-M33"
fi
exit "$fail"
