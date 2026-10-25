# ADR-0011: Matchers and the locate pipeline have an asynchronous form

- Status: Proposed
- Date: 2026-10-25

## Context

The locate pipeline must run in a browser and on vehicle hardware with the same
code (luofang34/Navigate#134). The pipeline renders reference views, matches each
view with the query, and fits the pose.

In a browser, a GPU result is read back only after the browser returns control to
its event loop. A call that waits for the GPU cannot complete. `device.poll(Wait)`
does not block on the web. `Atomics.wait` in a Worker needs cross-origin isolation
headers, and GitHub Pages cannot send them.

`ImageMatcher::match_images_blocking` and `PairLocalizer::estimate_calibrated_blocking`
block the caller. `GpuRenderer::render` is already asynchronous (ADR-0010 and #55
ask for an asynchronous renderer port).

`std::time::Instant` panics on `wasm32-unknown-unknown`.

## Decision

- `navigate-visual` defines `AsyncImageMatcher`. It has the same contract as
  `ImageMatcher`; only the call form is asynchronous.
- `PairLocalizer::estimate_calibrated` is the asynchronous form of
  `estimate_calibrated_blocking`. Both forms use one validation and one
  pose-verification step, so they give the same decisions.
- The blocking forms stay. Native callers that do not need the asynchronous form
  keep them. A native caller of the asynchronous form drives it with `pollster` or
  Tokio.
- Crates on the locate path measure time with `web_time::Instant`. On native
  targets it is `std::time::Instant`.

## Consequences

- A matcher that runs on a GPU in a browser implements `AsyncImageMatcher`. A CPU
  matcher can implement both traits.
- The locate pipeline is written as `async` functions and runs unchanged in a Web
  Worker and on native hosts.
- A test drives the asynchronous form with the same test matcher as the blocking
  form and compares the decisions.
