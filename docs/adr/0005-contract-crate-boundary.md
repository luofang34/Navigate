# ADR-0005: External consumers pin a pure-types contract crate; aiding stays draft until a joint RFC

- Status: Accepted
- Date: 2026-07-29

## Context

The host platform consumes Navigate under a contract-first rule: in-process
linking today, a sidecar process split later, with no contract change. The
FC side consumes (and may reject) aiding observations whose schema is owed
to a joint RFC between the two projects. Both need a boundary they can pin
that does not drag Navigate's internals — the same pattern as an FC
exposing a pinned contract crate for its shared-memory layout.

## Decision

- **`navigate-contract` is the boundary.** It contains pure types only:
  solutions, integrity assessments, stamps, source compositions, flight
  plans, guidance setpoints, and the draft aiding module. No I/O, no
  behavioral traits, no math dependencies. External consumers pin this
  crate (path or git revision); nothing else in the workspace is a
  contract.
- **Canonical units are SI-canonical**: radians for angles, meters for
  distance, meters per second for speed, nanoseconds for monotonic time.
  Display formatting is a consumer concern.
- **Guidance setpoints mirror the FC's declared command surface** —
  position, velocity, deviation tracking — and the vocabulary is
  `#[non_exhaustive]` so richer surfaces (trajectory-level handoff, if
  ever adopted FC-side) arrive additively.
- **The aiding module is DRAFT.** Its types exist so the shape is
  reserved and testable, and its documentation says so; the wire schema,
  covariance encoding, integrity terms, and source-composition encoding
  are owed to a joint RFC with the FC side before any aiding flows to a
  real controller. Draft status is lifted by a superseding record, not by
  silent use.
- **Behavioral traits are deferred deliberately.** The embedding surface
  (how a host drives fusion, plans, and guidance) stays in the
  implementation crates until the first real integration teaches its
  shape; freezing an engine trait before that would be speculation.

## Consequences

- A host can display solutions and load plans by pinning one small crate.
- Contract evolution is additive: `#[non_exhaustive]` types with
  constructors, new variants over breaking edits; a breaking change is a
  new major decision, not a patch.
- The sidecar split, when it comes, serializes contract types; choosing
  the encoding then is a new record and does not reshape the vocabulary.
- Nothing in the contract names a transport, a session, or an authority
  concept: commanding remains the host platform's fenced path, and the
  contract cannot be used to bypass it.
