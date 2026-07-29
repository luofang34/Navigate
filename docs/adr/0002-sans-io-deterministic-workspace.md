# ADR-0002: One Cargo workspace with sans-IO deterministic core crates

- Status: Accepted
- Date: 2026-07-29

## Context

Navigate's domain logic — filtering, integrity, sequencing, guidance — must
be testable without hardware, replayable deterministically, and embeddable
both in-process inside a host platform and behind a future sidecar process.
The sibling systems already prove the discipline: the FC's kernel is a pure
state machine driven by an external world, and the host platform's core
crates are sans-IO by decision.

## Decision

- The repository is a single Cargo workspace. All domain logic lives in
  sans-IO core crates: inputs are typed values and explicit `now`
  timestamps; outputs are typed values. Core crates MUST NOT depend on
  tokio, sockets, threads, system clocks, random number generators, or
  sensor SDKs.
- Cores are `std` crates (collections and `f64` math are used freely) but
  exclude nondeterminism, not the standard library. A `no_std` refactor is
  a non-goal; revisit if a flight-computer deployment without an OS
  becomes concrete.
- Numeric work uses `nalgebra` on `f64`. The dependency is confined to
  crates that need matrix algebra; the contract crate stays
  dependency-light so pinning it never drags a math stack.
- Workspace-enforced gates: `unsafe_code = "forbid"`,
  `missing_docs = "deny"`, clippy `unwrap_used`/`expect_used`/`panic`
  denied, disallowed `anyhow::Error` and print macros, `fmt`/`clippy`/
  `test`/`doc`/release CI, structure checks (500-line file limit, no
  `mod.rs`, `lib.rs` under 100 lines with a crate-level doc comment).
- Errors are typed `thiserror` enums carrying the context their messages
  need; `Result` propagation, never `unwrap`/`expect` in library code.

## Consequences

- Every filter, sequencer, and guidance behavior is a plain unit test with
  synthetic inputs; replay determinism is a property, not a feature.
- Platform integration (transports, shared memory, sensor drivers) lives
  outside this repository, in the host platform's adapters or future
  binder crates here that wrap the cores without leaking I/O into them.
- Time originates outside: whoever drives the cores supplies monotonic
  timestamps, which is what makes staleness bounds and sequencing
  decisions reproducible in tests.
