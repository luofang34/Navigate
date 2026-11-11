# ADR-0022: One navigation data contract, content-addressed mission packages, and one onboard boundary

- Status: Proposed
- Date: 2026-11-10

## Context

A host that flies a route with Navigate needs map data along the route.
Visual terrain-relative navigation (TRN) uses imagery and elevation. Other
means will use other data, for example magnetic anomaly fields and star
pair tables.

At this time, each host gets and identifies this data in its own way:

- `tools/visual-pipeline reference` writes a `photo-reference-catalog` for a
  bounding box. The catalog has no corridor, no vertical datum, no accuracy,
  and no record of missing tiles.
- `navigate_visual_flight::Surface::from_catalog` hashes the catalog bytes.
  The bytes change with the fetch order, so two hosts that fetch the same
  tiles get different identities.
- `navigate_locate::CachedHttpSource` fetches tiles over HTTP. Navigate thus
  fetches data, but its rule is "consumes, never fetches".
- The NVKTILE1 corridor bank and the NVKBANK1 reference bank bind to the
  catalog digest. They do not record the complete recipe that made them.
- `navigate-visual` keys a `MapCell` by the digest of the map manifest
  (`session/evidence.rs`). Two packages cut from the same imagery then
  count as independent evidence, but their registration error is common.

ADR-0007 gave the map release identity to Pilotage. ADR-0010 named one
package identity for the display and the reference renderer, but no crate
computes it. Pilotage ADR-0044 defines one logical store with a
content-addressed `Offline/chunks/<sha256>` layout, a package store that is
the only writer, and an asynchronous read port with `open`, `len`, and
`read_at`. `pilotage-svs-db` holds signed aeronautical database packages
with integer accuracy fields and day numbers.

Each host also composes the filter, the route sequencer, and the guidance
law itself. A simulator that scores Navigate must then copy this
composition, and the score then does not apply to Navigate.

## Decision

### 1. Navigate consumes and never fetches

The host fetches, decides trust, and installs. Navigate owns:

- the data contract: the manifest, its canonical encoding, the digests,
  the integrity checks, and the read port;
- the content specification of each layer and each derived product;
- the resource model that computes what a flight needs;
- the builders that turn supplied layers into derived products.

### 2. One light crate holds the contract

`navigate-data` holds the contract. It has no `burn`, no render types, and
no network client. It is `#![no_std]` with `alloc`, and it compiles for
`wasm32-unknown-unknown` and for the bare-metal targets of the core crates.
The `std` feature adds a directory store. The crate allocates (manifests
and tile indexes are variable in size), so it is not a zero-heap core
crate of ADR-0015 and ADR-0016.

The crate has two ports:

- `FetchPort`: the host implements it. A builder calls it when the host
  prepares a package. A tile key is (source, revision, layer kind, zoom,
  column, row). The key names the content, not a URL.
- `ReadPort`: the host implements it on its store. It has the shape of the
  ADR-0044 read port: asynchronous `open` of a logical path, then `len` and
  `read_at` of a byte range on the open file. A reader reads a range of a
  tile index without reading the whole index.

### 3. What is hashed

The package identity is the SHA-256 of the canonical manifest bytes. The
manifest holds only content and its physical meaning:

| In the identity | Why |
|---|---|
| format and version | A reader must refuse a format it does not know. |
| body, ellipsoid | Heights and positions have no meaning without them. |
| grid scheme | The tile addresses have no meaning without it. |
| per layer: kind, source key, source revision, horizontal datum, vertical datum, encoding, tile pixels, accuracy, correlation zoom | These change what a tile means and how much a navigator can trust it. |
| per layer: the digest, size, and counts of its tile index blob | The index binds each address to a blob digest or a gap. |

| Not in the identity | Where it is |
|---|---|
| release identifier, layer labels | the descriptor |
| attribution text, licence, terms | the descriptor |
| fetch time, host paths, URLs, notes | the descriptor |

The descriptor is a JSON file beside the manifest at
`packages/<identity>.descriptor.json`. The writer writes it, and a reader
reads it for display and licence text. It does not change the identity,
and Navigate never uses it to navigate.

A source key is a name from a small registry in `navigate-data`, for
example `esri_world_imagery` and `mapzen_terrarium`, or a declared source,
`declared:<sha256>`. A host with its own data (a private survey, a
national orthophoto, a simulated world) writes a source declaration: a
canonical document with closed integer fields (layer kind, nominal ground
sample distance in millimetres, the accuracy that the provider states, the
collection day, and the digest of a stable identifier that the provider
chooses). The digest of the declaration is the key. Hosts that declare the
same source with the same values get the same key, so correlation keys are
defined for every source without a change to Navigate. A source revision is a provider release number, a
collection date, or `unversioned`. The blob digests are the content
identity; the revision only tells a reader when the provider changes
content under the same key.

### 4. Canonical encoding

The canonical encoding is compact JSON (no spaces) with these rules:

- Object fields are in the declared order of the type. Unknown fields are
  refused.
- Numbers are integers only. Lengths are in millimetres, angles in
  microradians or microdegrees, rates in millimetres per year, dates in
  days from 2000-01-01 (`DayNumber`). There is no floating-point value in
  a hashed document.
- Strings are only names from closed vocabularies (snake case) or digests
  (64 lower-case hexadecimal digits).
- Collections are sorted in a stated order, with no repeats.
- Each optional value is present as `null` when it is absent. No field is
  left out.

A reader decodes the manifest, encodes it again, and refuses bytes that
differ. The same rules apply to tile indexes (binary), product chunks, and
product lists.

### 5. Tile indexes

Each layer has one tile index blob. It is a canonical binary file:
the magic `NVDIDX01`, then records of 49 bytes, sorted by zoom, row, and
column: zoom (u8), column (u32), row (u32), size (u64, zero for a gap), and
blob digest (32 bytes, zero for a gap). Integers are little endian. The manifest holds the index
digest, its size, and its tile and gap counts. A reader checks the index
digest once, then reads records by range and binary search.

Measured (release build, Apple M-series, in-memory store; the 1 000 km
route is 43 790 zoom 17 imagery tiles in a 2.2 km strip and 1 096 zoom 14
elevation tiles):

| Package | Inline JSON list: manifest, open | Index blobs: manifest, indexes, open, one lookup |
|---|---|---|
| Zaragoza (494 + 15 tiles) | 54 713 B, 0.35 ms | 987 B, 24 957 B, 0.02 ms, 0.7 µs |
| 1 000 km at 300 m above ground | 4 756 513 B, 33 ms | 995 B, 2 199 430 B, 1.2 ms, 0.8 µs |

The open time with index blobs includes the check of each index digest.
The manifest stays under 1 KB for any route, and a vehicle reads index
records by range.

### 6. Package identity, missions, and blobs

- Each blob is stored at `chunks/<sha256>`, the ADR-0044 layout. Packages
  that cover the same ground share blobs.
- A mission package is a manifest at `packages/<sha256>.json`. The writer
  writes the blobs, then the tile indexes, then the manifest last. A store
  without the manifest holds no package.
- A host fills `navigate_visual::MapRevision::manifest_sha256` with the
  package identity.
- `Package::open` checks the manifest digest and its canonical form. Each
  tile read checks the digest and size of its blob. A read in a gap gives
  `Gap`; a read outside the coverage gives `Outside`. A missing tile is
  never skipped silently.
- Retirement: a blob is live while an installed manifest, tile index,
  product chunk, or product list refers to it. The host store computes
  this from its installed documents and deletes the other blobs. Navigate
  never deletes.

### 7. Accuracy, correlation, and datum

- Each layer states its horizontal and vertical one-sigma in millimetres,
  the date of collection as a day number, the growth of the error in
  millimetres per year, and the surface model (bare earth or first
  return). Each value can be absent (`null`), which means unknown.
- The validity of a package in time belongs to the host and its trust
  layer. A navigator computes the present one-sigma of a layer from the
  collection day and the growth, and each reader function that depends on
  time takes the present day as an argument (for example
  `Accuracy::horizontal_sigma_m(today)`). No reader reads a wall clock.
- Correlated error is keyed by (source key, source revision, grid cell at
  the layer correlation zoom), not by the package identity. Two packages
  cut from the same imagery have the same keys, so their fixes in one cell
  share error. The correlation zoom is per layer; zoom 16 is the default.
  `navigate_visual::session::MapCell` holds this key for a fix from a
  package layer. A map without a key (a tile catalog, a frame at a pole
  or on another body, where no grid scheme applies) gets a local cell of
  that one map.
- A Web Mercator cell is not of constant size: a zoom 16 cell is
  approximately 611 m on a side at the equator, 460 m at 41° latitude, and
  306 m at 60°.
- Each layer states its horizontal datum (WGS84 at this time) and, for
  elevation, its vertical datum: `ellipsoidal`, `orthometric` with a named
  geoid model, or `unknown`. Heights are stored as the provider supplies
  them. A reader converts them with a `navigate_geodesy::GeoidUndulation`
  model of that name, which the host supplies. Without one, the reader
  gives `UnknownUndulation` with a bound of 107 m (the EGM96 undulation is
  between approximately −107 m and +86 m), and the error budget carries the
  bound. `navigate-geodesy` has no EGM96 model at this time. The error of
  a coarse grid model is to be measured when such a model is added.

### 8. Grid schemes and bodies

The manifest states the body (Earth) and its ellipsoid (WGS84). The grid
scheme is a named scheme. The one scheme at this time is the Web Mercator
XYZ pyramid, with the latitude limit 85.051 128°. A request past the limit
gets a typed refusal. A polar scheme and the schemes of other bodies are
reserved: ADR-0020 covers polar TRN, and `navigate-visual-render` refuses
`BodyTangentEnu` frames. This record adds no code for them.

### 9. Unknown values and format evolution

- A manifest, tile index, product chunk, or product list of another format
  version is refused with a typed error. There is one current schema. A
  change of schema is a new version, and a host makes its packages again
  from its sources and caches. There is no migration code and no second
  reader.
- In the core manifest, an unknown body, grid scheme, layer kind,
  encoding, datum, or source key is refused with a typed error.
- In a product list, an unknown product kind is carried and enumerated as
  unknown. A navigator never reads it.

### 10. Resource model in two phases

1. **Ground.** The host fetches elevation tiles at zoom 10 (approximately
   120 m pixels) along the route corridor, through the same fetch port.
   The ground heights come from these tiles.
2. **Plan.** A pure function computes, for each leg, the layers, the zoom
   levels, the corridor width, and an estimate of the size. Its inputs are
   the legs with their height above ground (from phase 1), the declared
   cameras (intrinsics, mount depression), the expected navigation
   uncertainty, and a deviation allowance.

The zoom rule is the rule of the map renderer: one texel for one pixel, so
one zoom level for each doubling of the slant range. For tiles of 256
pixels, the texel size at zoom `z` and latitude `φ` is
`156 543.03 m × cos φ / 2^z`. The renderer uses tiles of 512 pixels, so
its level for the same texel is one less. A higher leg and an oblique view
need coarser levels and cost less for each kilometre. Altitude octave
bands key the derived products.

The size of a plan is the count of its planned tiles times the size of
one tile of each layer, because a whole tile is fetched however little of
it a strip covers. The estimate uses a typical tile size, and a budget is
judged on an upper tile size. Measured on the Zaragoza package: Esri
imagery JPEG 15.5 to 17.3 kB a tile (typical 16 kB, upper 20 kB),
Terrarium PNG 76 to 100 kB a tile (typical 90 kB, upper 110 kB). The
NVKTILE1 bank at its default settings is approximately 4.2 MB/km².

### 11. Three onboard representations

The host selects one representation for each leg, against a storage budget
and the requirements that the navigator declares:

- **R**: raw layers at the matched levels.
- **L**: sparse landmarks and a coarse DEM, with no imagery. This
  representation is specified, but no navigator implements it at this
  time: the corridor engine refuses to start without imagery.
- **K**: route keyframes: reference views or features made at planned
  poses, each with a validity envelope (lateral, vertical, and heading
  ranges).

### 12. Derived products in two levels

- A **product chunk** is a canonical document at
  `products/chunks/<sha256>.json` with its blobs in `chunks/`. Its identity
  covers: the product kind, the digest of the complete recipe (algorithm,
  version, weights digest, precision, and each parameter as an integer),
  the grid cell or `null`, the altitude band or `null`, the sorted digests
  of the input blobs, and for route products the route digest and the
  validity envelope. A chunk does not name a package, so any mission whose
  package holds the same input blobs can use it.
- A **product list** is a canonical document per mission at
  `products/lists/<sha256>.json`. It names the package identity and the
  chunks. It is outside the core identity.
- A chunk built for another altitude band, another route digest, or other
  input blobs does not match, and the navigator does not use it. A keyframe
  is used only inside its envelope.
- NVKTILE1 limitation: the bank tiles are in an east, north, up frame at
  the centre of the surface, not on the fixed grid. A bank is thus one
  chunk for its whole area, and it is not reusable cell for cell. The
  builder must tile on the grid before banks can be shared by cell.

### 13. The onboard boundary

A new `std` crate, `navigate-onboard`, holds the composition that a host
links:

- `Navigator`: it receives sensor inputs and gives a
  `navigate_contract::NavigationSolution` and acceptance records.
- `FlightDirector`: it sequences the route and computes the guidance
  setpoint from a `&NavigationSolution`.
- A bundle that owns one of each. A host links each one as a trait object.

Rules of both traits:

- **Time.** Each input carries its acquisition and delivery stamps on the
  navigator clock domain. No navigator reads a wall clock.
- **Alignment.** The initial state is an explicit input with its
  covariances. A navigator gives no solution before it.
- **State.** Each trait has `snapshot` and `restore`: a navigator
  duplicates its complete state into a boxed copy, so a simulator can fork
  a run and restore and replay it. A trait object cannot be cloned, so
  this operation is part of the trait.
- **Sensors.** The host declares the sensor suite once: for each sensor,
  an identifier, a kind, the imager model of `navigate-camera`
  (ADR-0021), the mount and lever arm, the rate, and the error budget.
  Each input carries the sensor identifier, the stamps, and the data. The
  data type is `#[non_exhaustive]` and has a variant that passes a
  `navigate_fusion::Observation` through.
- **Availability.** A sensor is unavailable only when its input does not
  come. The navigator has no mode that disables a sensor, so a host cannot
  hide a failure with one.
- **Data requirements.** Before load, a navigator gives its requirements:
  the layers and product kinds it needs or can use, with the altitude
  bands and representations. After load, it gives a report for each
  product: used, fallback to another representation (with the reason), or
  refused (with the reason). A required layer that is missing is a
  refusal of the load, not a silent degradation.
- **Acceptance records.** The navigator writes each record into a sink
  that the caller supplies. `navigate-onboard` is not a core crate of
  ADR-0015 and ADR-0016, but the sink keeps each step free of allocation
  in the trait.
- The learned corridor stack (`navigate-visual-flight`) has its own
  filter. It is a second `Navigator`, not an aiding means of the first.

The terrain-camera means of the reference navigator follows these rules:

- **Frames.** The host declares the camera once: intrinsics, spectral
  band, mount, and a calibration error. Each input is one frame as the
  sensor gave it: luminance pixels and the time of the middle of the
  exposure. The host makes no observation. A camera that is not a visible
  camera is refused when the navigator is made.
- **Map.** The host opens a store and gives it to the navigator with the
  package identity, the present day, and, if it has them, the digest of
  the product list, the geoid model, and the accuracy to assume for a
  layer that states none. The navigator refuses, with a typed reason, a
  package that is missing or not the expected one, a layer that is absent
  or that the renderer cannot read, and an accuracy that no one states.
  The answer of a load states the identity, the layers with their zoom
  levels, pixel size, and present accuracy, and the use of each product
  chunk. A frame without a loaded map is an error, not a record.
- **Reference.** Inside the envelope of a stored keyframe, the keyframe is
  the reference view. Outside, the navigator renders the reference from
  the package at its believed camera pose.
- **Belief at the exposure.** The filter gives a solution only at or
  after its state time. The search starts at the solution on delivery,
  moved back along its velocity, and its bounds grow with the delay. A
  frame older than a set delay is not matched. The fix comes from the
  image only: its lever arm comes off with the solved attitude, and its
  stamp is the exposure.
- **Covariance.** The covariance of a fix is the image-geometry
  covariance of the match plus the covariances of independent terms (a
  root sum square, not a sum of one-sigma values): the horizontal
  accuracy of the imagery on the present day; the vertical accuracy of
  the elevation, with the bound of an unknown undulation, on the vertical
  axis and, times the tangent of the off-nadir angle, along the ground
  direction of the view; and the calibration angle times the range.
- **Correlation.** The correlation key of the matched ground labels each
  fix. The first fix that the filter accepts in a cell is an independent
  fix. A later fix in that cell shares its map error, so the navigator
  does not offer it to the filter, and the record states that. A fix
  that the filter refuses does not close its cell.
- **Records.** Each frame gives one record: accepted, refused by the
  filter with its reason, or not used with a reason, and the path of the
  reference, `keyframe` or `render`.
- **State.** A snapshot holds the frame order, the cells with a fix, and
  the resident reference data. The map is immutable and shared.

### 14. Host obligations

| Host | Fetch port | Read port | Store writer |
|---|---|---|---|
| Desktop and tools | HTTP with a cache, or a catalog on disk | directory store | the tool or the package store |
| Browser | `fetch` in a worker | OPFS (ADR-0044) | the package store, in a worker |
| Vehicle | none in flight | the installed store, read only | none in flight: install before flight |

### 15. Decisions not taken

| Decision | Seam |
|---|---|
| Signing and trust | The host signs the package identity (a digest), for example with the `pilotage-svs-db` trust root. |
| Transport | The fetch port and the host store. |
| Compression | A future encoding value, which is a new format version. |
| Licence enforcement | The descriptor carries the terms; the host enforces them. |
| Polar and other-body grids | The grid scheme value (§8). |
| Representation L in a navigator | The data requirements of a navigator (§13). |
| Fusion of a fix that shares the map error of an earlier fix | The one function of the terrain means that turns the label of a fix into what the filter is told. A filter state for time-correlated position error lets it pass the fix with a declared correlated error. |
| The belief of the navigator at a past exposure | The one function of the terrain means that makes the belief. A query of the filter history replaces the move back along the velocity. |

### Relation to Pilotage

- **ADR-0044.** The read port has the shape of the ADR-0044 read port. The
  store layout is the ADR-0044 layout. The package store of Pilotage
  writes; Navigate reads.
- **`pilotage-svs-db`.** That crate signs and activates aeronautical
  databases. `navigate-data` does not sign and does not activate. It uses
  the same integer conventions (millimetres, day numbers).
- **`pilotage-mission`.** It uses `PlanExecution` and
  `navigate_guidance::guide_velocity`. Pilotage holds the control
  authority and the command leases. `FlightDirector` holds no authority,
  no leases, and no command output. Its setpoint includes the velocity
  guidance, so Pilotage can put its sequencing and guidance behind it and
  keep its authority logic. This record does not require that change.

## Consequences

- One identity names the map of a fix, and the library computes it. Two
  hosts that install the same tiles from the same source get the same
  identity, whatever labels they give.
- Overlapping missions share blobs and product chunks.
- Fixes from two packages of the same source in one cell count as
  correlated.
- A changed blob, a changed manifest, a non-canonical document, and a
  document of another version are found and refused.
- The reader must apply the geoid model, or carry the unknown undulation in
  the error budget.
- A simulator links the onboard bundle and scores the onboard composition.
- `photo-reference-catalog` stays a tool input. A builder converts it to a
  package.

## Alternatives considered

- A route field in `photo-reference-catalog`. Rejected: the catalog has no
  datum, accuracy, or gaps, and its bytes change with the fetch order.
- Fetch in Navigate. Rejected: Navigate consumes and never fetches.
- Labels, attribution, or fetch time in the identity. Rejected: the
  identity would change with the host and the time.
- Floating-point values in hashed documents. Rejected: two encoders can
  write one value in two ways.
- An inline tile list in the manifest. Rejected: see the measurements in
  §5.
- Products bound to the package identity. Rejected: no product could
  serve a second mission.
- Correlation cells keyed by the package identity. Rejected: two packages
  of one source would count as independent.
- A migration reader for old versions. Rejected: one current schema.
- One trait for navigation and guidance. Rejected: a host could not
  replace one and keep the other.
