#!/usr/bin/env bash
# End-to-end corridor test suite runner.
# Executes all 299 tests across Tiers 1 through 4.
set -euo pipefail

echo "=================================================="
echo "Running End-to-End Visual Corridor Test Suite"
echo "Package: e2e-corridor-tests (299 test cases)"
echo "=================================================="

cargo test -p e2e-corridor-tests "$@"

echo "=================================================="
echo "All 299 End-to-End Corridor Tests Passed"
echo "=================================================="
