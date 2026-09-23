# ADR-0010: Reference views come from a renderer port that Navigate owns

- Status: Proposed
- Date: 2026-09-23

## Context

The MapLibre fork has two jobs. It shows map layers to the operator, and it
renders reference views for visual positioning. The two jobs have different
needs:

| | Display | Reference |
|---|---|---|
| Content | A style with symbols, labels, fades, and atmosphere | Imagery and terrain only |
| Tile detail | Chosen for the display size | Chosen from the source data |
| Completion | Frames can refine over time | A view is returned only when it is complete |
| Output | Pixels on a screen | Colour, optical-axis depth in metres, map and renderer identity |

The visual tools used MapLibre directly. They rendered a fixed count of
settle frames and did not record which style made a view. A small render
target selected a coarser tile zoom than the supplied tiles, and no view
appeared.

## Decision

### The port

`navigate-visual` defines the `ReferenceRenderer` trait:

- `identity` gives the renderer revision and the SHA-256 of the reference
  style.
- `render_blocking` renders one candidate pose.
- `render_batch_blocking` renders candidates in order. The default renders
  them one at a time. A renderer that draws several views in one pass
  overrides it.

The navigation core depends only on the trait. The MapLibre fork, a
precomputed orthophoto, or a test double can implement it. The visual bench
implements it on the fork through `maplibre::headless::map::reference`.

### Requirements on an implementation

The trait documentation states them:

1. Draw only imagery and terrain.
2. Select tile detail from the source data, not from the display size.
3. Return a view only when it is complete, not after a fixed frame count.
4. Report optical-axis depth in metres, and zero where no surface or imagery
   is present.

The fork meets requirement 4 now. Requirements 2 and 3, and a batched
multi-view render, are tracked as fork work. Until they land, the fork
renders settle frames, and its documentation states the tile-detail limit.

### One package, one identity

The display and the reference renderer read the same installed package. The
package `pack_id` is the map identity for both (the visual bench records it
with every result). With the renderer identity, a fix names every input that
made its reference.

### Scheduling

On a device with a display, reference renders run on their own queue at a
lower priority than the display, with a time budget for each frame. On an
onboard host, the reference renderer runs headless without a display.

## Consequences

- The navigation core has no MapLibre dependency. A faster reference source,
  such as a warped orthophoto for near-nadir cameras, replaces the renderer
  behind the same trait.
- Batched rendering is an optimization inside an implementation. Callers
  already use `render_batch_blocking`.
- The WebAssembly preview renders asynchronously. An asynchronous form of the
  port is added when the browser pipeline moves to it.
