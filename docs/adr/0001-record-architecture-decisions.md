# ADR-0001: Record architecture decisions as versioned files in this repository

- Status: Accepted
- Date: 2026-07-29

## Context

Navigate spans estimation, integrity, flight-plan execution, and guidance —
decisions that will be re-examined by contributors who need the reasoning,
not just the outcome, and by the sibling systems (flight controller, host
platform) whose boundaries these records define from Navigate's side.

## Decision

- Every significant architectural commitment is recorded as one file under
  `docs/adr/`, named `NNNN-short-slug.md`, numbered in acceptance order.
- Records use a compact MADR-style template: Status, Date, Context,
  Decision, Consequences, and — where a real choice was weighed —
  Alternatives considered.
- Statuses: `Proposed`, `Accepted`, `Deprecated`, `Superseded by ADR-NNNN`.
- An Accepted record is immutable except for status changes and factual
  errata. A changed decision gets a new record that supersedes the old one;
  history is never rewritten.
- Requirement words `MUST`, `SHOULD`, `MAY` follow RFC 2119 usage.
- Open questions listed inside a record are commitments to decide, not
  decisions.

## Consequences

- Design review happens per decision, matching one-issue-per-PR discipline.
- The index in `docs/adr/README.md` is the entry point and MUST stay
  current.
