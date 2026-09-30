#!/bin/sh
set -eu

# The runtime reads this before its telemetry API can disable background work.
export ORT_DISABLE_TELEMETRY=1
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec "$script_dir/target/release/visual-bench" "$@"
