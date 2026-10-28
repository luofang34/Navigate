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
exit "$fail"
