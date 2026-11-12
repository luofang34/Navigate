# ADR-0009: The navigation estimate narrows the visual search and never admits a fix

- Status: Proposed
- Date: 2026-09-23

## Context

Visual positioning renders reference views at candidate poses and matches
them with the camera frame (ADR-0007). Each candidate costs a render and a
match. The number of candidates sets the time to a fix.

The fused navigation solution already knows where the vehicle probably is.
Its position covariance gives the size of the area to search. The flight
controller gives the attitude. With this information, most frames need one
candidate or a few, not a global search.

There is one danger. If the same estimate also sets the admission bound of a
visual fix, a wrong estimate can accept a wrong fix. The filter then fuses
that fix and becomes more confident in the wrong position. ADR-0007 already
refuses a fix whose independence from the state is not validated.

## Decision

### The search prior

`navigate-visual` defines `SearchPrior`: a camera pose projected to the
frame capture time, a position covariance, and an attitude uncertainty.
`navigate_visual_fusion::search_prior` builds it from a
`NavigationSolution`:

- It refuses a frame and a solution that use different clock domains.
- It moves the position with the solution velocity over the time step.
- It adds the velocity covariance over the time step to the position
  covariance.
- The caller supplies the camera attitude from the flight controller
  attitude and the camera mounting.

### Three search tiers

`SearchPrior::tier` compares the sigma radius of the horizontal error
ellipse with `SearchConfig` thresholds:

| Tier | Condition | Work |
|---|---|---|
| `Local` | Radius at or below `local_radius_m` | Refine from the projected pose |
| `Region` | Radius at or below `region_radius_m` | Render and match grid candidates inside the ellipse, most likely first, at most `max_candidates` |
| `Global` | Larger radius | Global retrieval against precomputed descriptors |

The host stops at the first candidate that the verifier accepts. The
candidate order puts the most likely pose first, so the expected cost is low
when the estimate is good. Global retrieval is tracked as separate work. Until
it lands, a global tier gives no candidates, and the host reports that the
frame could not be located.

### Acquisition by the terrain camera

The onboard terrain camera does not render a grid of candidates for the
`Region` and `Global` tiers. One reference view reaches a few hundred metres,
so a grid over an ellipse kilometres wide costs hundreds of renders. The
camera acquires instead (`navigate_visual::acquisition`):

- It lays the frame on flat ground below the believed eye, at a few headings
  about the believed one, as a raster of ground cells. It correlates that
  raster with a raster of the map about the belief, within the search
  radius of the tier, capped by `max_acquisition_radius_m`.
- The places whose correlation is within a margin of the best are the
  candidates, at most `max_candidates`. More of them means that the ground
  repeats, and the frame is not located.
- Each candidate is rendered and matched. `CandidateResults` decides: one
  accepted candidate gives a pose; several give none.

Building the map raster and the correlation are counted work. The host
spreads them over frames, so one frame takes a bounded time.

### Search and admission are separate

### Search and admission are separate

The search prior chooses where to look. It never sets the admission bound.
`PoseVerifier` accepts a pose only against a `PosePrior` that the host builds
from evidence independent of the fused estimate, for example the prior before
the last visual fixes, or a bound from another means. There is no conversion
from `SearchPrior` to `PosePrior`.

An acquired pose has no such independent bound. Its `PosePrior` is the place
that the correlation found, and inside a belief kilometres wide the filter
gate cannot refuse a wrong place either. The evidence that admits it is a
second acquisition: the terrain camera holds the fix until the next
acquisition, from another frame and another reference view, corrects the
belief by the same amount. The two corrections may differ by the search
multiple of the one-sigma of their difference: the two fix covariances and
what the inertial solution added to the belief between the frames. Only the
confirming fix goes to the filter. The confirmation does not catch a
repeated pattern that both frames mistake alike. The candidate margin and the
match of every candidate guard against that.

### Performance

- Candidate generation is a bounded grid. It does not allocate beyond
  `max_candidates` poses.
- A renderer can draw a batch of candidates in one pass (ADR-0010).
- A pack for a planned route can include reference descriptors along the
  corridor. The host then matches features instead of rendering for most
  frames.

## Consequences

- A good estimate gives a fix from one render. A poor estimate costs at most
  `max_candidates` renders, and the host knows this before it starts.
- A wrong estimate can make the search miss. It cannot make a wrong fix pass
  admission.
- After a long gap, the first terrain fix needs two acquisitions. At the
  default budget of a frame, that is about a dozen frames when the map
  raster has to be built.
- Global retrieval and route descriptor precomputation are tracked as
  separate work.
