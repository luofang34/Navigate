# ADR-0004: Every solution carries an integrity assessment that never overstates redundancy

- Status: Accepted
- Date: 2026-07-29

## Context

Consumers of the navigation solution make decisions of very different
criticality: a moving map can tolerate degraded accuracy; guidance toward
terrain cannot. The filter always has *some* estimate; the danger is an
estimate presented with confidence the sensor set cannot support —
especially in single-source operation, which is an explicitly supported
mode, not a failure.

## Decision

- Every published `NavigationSolution` embeds an `IntegrityAssessment`:
  - a quality classification (`Good`, `Degraded`, `Unusable`);
  - the set of source classes that actually contributed within the
    assessment window;
  - a redundancy level derived from that set (none, single, multiple
    independent classes);
  - 1-sigma horizontal and vertical uncertainties derived from the state
    covariance;
  - a fault-detection statement: with no redundant information the
    assessment MUST say fault detection is unavailable — it MUST NOT
    infer health from a single source agreeing with itself;
  - reserved, explicitly optional protection-level fields (RAIM-class
    bounds), absent until an implementation earns them.
- Quality is derived, not asserted: covariance beyond configured bounds
  or source silence degrades it mechanically.
- Consumers that require integrity fail closed on absence: guidance
  refuses maneuvers whose integrity requirement exceeds the assessment,
  while display consumers render the quality rather than hiding the
  solution.
- Solution age is the consumer's judgment: the stamp carries acquisition
  time, and republication never refreshes it.

## Consequences

- Single-source honesty is testable: the same trajectory with one source
  and with three yields the same state math but different assessments —
  pinned by tests.
- The assessment vocabulary lives in the contract crate, so the host
  platform's displays and the FC-side aiding consumers read one integrity
  language.
- Protection levels stay absent rather than approximate; an optimistic
  bound is worse than none.
