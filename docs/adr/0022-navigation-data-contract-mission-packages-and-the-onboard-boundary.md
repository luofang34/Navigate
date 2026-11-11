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
- A simulator that tested a visual prototype selected tiles, fetched them,
  and built atlases itself. A copy of this logic in each host can differ
  from the onboard logic.

ADR-0007 gave the map release identity to Pilotage. ADR-0010 named one
package identity for the display and the reference renderer, but no crate
computes it. Pilotage ADR-0044 defines one logical store with a
content-addressed `Offline/chunks/<sha256>` layout, a package store that is
the only writer, and a read port that starts from the `navigate-data` read
port. `pilotage-svs-db` holds signed aeronautical database packages.

`navigate-data`, `navigate-data-fs`, and `navigate-imagery` had a read port,
storage classes, a coverage planner, and a chunked package. No crate used
them, and they were removed.

Each host also composes the filter, the route sequencer, and the guidance
law itself. A simulator that scores Navigate must then copy this
composition, and the score then does not apply to Navigate.

## Decision

### 1. Navigate consumes and never fetches

The host fetches, decides trust, and installs. The planner and the package
store of the host do this work. Navigate owns:

- the data contract: the manifest, its canonical encoding, the digests, the
  integrity checks, and an asynchronous read port that needs no file
  system;
- the content specification of each navigation layer and each derived
  product;
- the resource model that computes what a flight needs;
- the builders that turn supplied raw layers into derived products.

### 2. One light crate holds the contract

`navigate-data` holds the contract. It has no `burn`, no render types, and
no network client. It is `#![no_std]` with `alloc`, and it compiles for
`wasm32-unknown-unknown` and for the bare-metal targets of the core crates.
The `std` feature adds a directory store.

The crate has two ports:

- `FetchPort`: the host implements it. A builder calls it when the host
  prepares a package. A tile key is (source identifier, source revision,
  layer, zoom, column, row). The key names the content, not a URL. Map
  caches key tiles by URL, and the URL of one tile is different between
  hosts.
- `ReadPort`: the host implements it on its store. `Package` reads tiles by
  layer through it.

`navigate_locate::TileSource` stays where it is until a later change binds
the locate pipeline to packages.

### 3. Packages are content addressed

- Each blob is stored at `chunks/<sha256>`, the layout of ADR-0044.
  Packages that cover the same ground share blobs.
- A mission package is a canonical manifest at `packages/<sha256>.json`.
  The manifest refers to blobs by digest. The writer writes the manifest
  last, so a store without the manifest holds no package.
- The SHA-256 of the canonical manifest is the package identity. It covers
  the layers, their sources, the vertical datums, the grid, the declared
  accuracy, the tiles, and the gaps. A host fills
  `navigate_visual::MapRevision::manifest_sha256` with this identity.
- The canonical encoding is compact JSON with a fixed field order and
  sorted content. A reader encodes the decoded manifest again and refuses
  bytes that differ. Fetch times, host paths, and URLs are not in the
  manifest.
- `Package::open` checks the manifest digest. Each tile read checks the
  digest and the size of its blob.
- Each layer declares its coverage as tiles and gaps. A read in a gap gives
  `Gap`, and a read outside the coverage gives `Outside`. A missing tile is
  never skipped silently.

### 4. Accuracy, correlation, and datum are data

- Each layer states its horizontal and vertical sigma, the date of
  collection, the growth of the error per year, and the surface model (bare
  earth or first return). Each value can be `unknown`.
- A fixed global grid defines the correlation cells: the XYZ tiles at zoom
  16 (approximately 460 m at 41° latitude). The grid has no anchor, so the
  cells and the `MapCell` identifiers are the same in each mission. Derived
  products tile on the same pyramid.
- Each elevation layer states its vertical datum: `ellipsoidal`,
  `orthometric` with a geoid model name, or `unknown`. Heights are stored as
  the provider supplies them. The reader converts them with a
  `navigate_geodesy::GeoidUndulation` model that the host supplies. If the
  host has no model with that name, the reader gives `UnknownUndulation`
  with a bound of 107 m, and the error budget must carry it. The reader
  never takes an orthometric height as an ellipsoidal height.
- `navigate-geodesy` has only `ZeroUndulation` at this time. An EGM96 grid
  at 5° has 37 × 72 values: approximately 5 KB as 16-bit decimetres, with an
  error of approximately 2 m RMS and 23 m at the worst point. It goes in
  `navigate-geodesy`. A finer grid is a derived product of the package.

### 5. A resource model before a fetch

A pure function computes, before a fetch, what each leg of a flight plan
needs. Its inputs are:

- the legs, with position and altitude above ground (from the DEM);
- the declared cameras: intrinsics and mount pitch;
- the expected navigation uncertainty: the search radius;
- a deviation allowance.

For each leg it gives the layers, the zoom levels, the corridor width, and
an estimate of the size. The zoom rule is the rule of the map renderer: one
zoom level for each doubling of the slant range, with one texel for one
pixel. For tiles of 256 pixels, the texel size at zoom `z` and latitude `φ`
is `156 543.03 m × cos φ / 2^z`. The renderer uses tiles of 512 pixels, so
its level for the same texel is one less. A higher leg and an oblique view
need coarser levels and cost less for each kilometre. The far field of an
oblique view uses coarser levels than the near field. Altitude octave bands
key the derived products.

Measured sizes: imagery is approximately 18 KB/km² at zoom 15 and 70 KB/km²
at zoom 16. Terrarium elevation is approximately 2 KB/km² at zoom 12 and
7.5 KB/km² at zoom 13. The NVKTILE1 bank at its default settings is
approximately 4.2 MB/km².

### 6. Three onboard representations

The host selects one representation for each leg, against a storage budget
and the requirements that the navigator declares:

- **R**: raw layers at the matched levels.
- **L**: sparse landmarks and a coarse DEM, with no imagery. The landmarks
  carry their usable-geometry gate (range ratio 0.22 to 5.5, incidence, and
  orientation).
- **K**: route keyframes: reference features or views made at planned poses,
  each with a validity envelope (lateral, vertical, and heading ranges).

A derived product is outside the core identity. Each product records:

- the core package identity that it binds to;
- the complete recipe: algorithm, weights digest, numeric precision, and
  each parameter;
- the camera calibration, where it applies;
- the altitude band and the validity envelope.

A navigator declares the products that it needs. At load, it reports for
each product: use, fallback, or refusal. Product kinds are reserved for
terrain slopes, horizon profiles, scalar fields, star pair tables, station
lists, and ephemeris segments. This record defines no code for them.

### 7. The onboard boundary

A new `std` crate, `navigate-onboard`, holds the composition that a host
links:

- `Navigator`: it receives sensor inputs and gives a
  `navigate_contract::NavigationSolution` and acceptance records.
- `FlightDirector`: it sequences the route and computes the guidance
  setpoint from a `&NavigationSolution`.
- A bundle that owns one of each. A host links each one as a trait object.

The host declares the sensor suite once, when it builds the navigator: for
each sensor, an identifier, a kind, the imager model of `navigate-camera`
(ADR-0021), the mount and lever arm, the spectral band, the rates, and the
error budget. Each input carries the sensor identifier, the data, and the
time stamp. The input type is `#[non_exhaustive]` and has a variant that
passes a `navigate_fusion::Observation` through.

A sensor is unavailable only when its input does not come. The navigator
has no mode that disables a sensor, so a host cannot hide a failure with
one. The navigator writes acceptance records into a sink that the caller
supplies. `navigate-onboard` is not a core crate of ADR-0015 and ADR-0016,
but the sink keeps each step free of allocation in the trait.

The learned corridor stack (`navigate-visual-flight`) has its own filter. It
is a second `Navigator`, not an aiding means of the first one.

### Relation to Pilotage

- **ADR-0044.** The `ReadPort` of this crate is the read port that ADR-0044
  starts from. The store layout is the ADR-0044 layout. The package store of
  Pilotage writes; Navigate reads.
- **`pilotage-svs-db`.** That crate signs and activates aeronautical
  databases. `navigate-data` does not sign and does not activate. A host
  that requires a signature signs the package identity with its trust root.
  The package store then installs the blobs and the manifest, and activates
  the package with its pointer file.
- **`pilotage-mission`.** It uses `PlanExecution` (from `navigate-fpl` at
  the revision that Pilotage pins; `navigate-route` in this workspace) and
  `navigate_guidance::guide_velocity`. Pilotage holds the control authority
  and the command leases. `FlightDirector` holds no authority, no leases,
  and no command output. Its setpoint includes the velocity guidance, so
  Pilotage can put its sequencing and guidance behind it and keep its
  authority logic. This record does not require that change.

## Consequences

- One identity names the map of a fix, and the library computes it. Two
  hosts that install the same tiles get the same identity.
- Overlapping missions share blobs, and derived products tile on a fixed
  grid, so products made for one mission serve the next one.
- A changed blob, a changed manifest, and a non-canonical manifest are
  found. A gap is reported as a gap.
- The reader must apply the geoid model, or carry the unknown undulation in
  the error budget.
- A simulator links the onboard bundle and scores the onboard composition.
- `photo-reference-catalog` stays a tool input. A builder converts it to a
  package. The NVKTILE1 bank must record its recipe and bind to the package
  identity.

## Alternatives considered

- A route field in `photo-reference-catalog`. Rejected: the catalog has no
  datum, accuracy, or gaps, and its bytes change with the fetch order.
- Fetch in Navigate. Rejected: Navigate consumes and never fetches, and
  trust and installation belong to the host.
- A manifest that holds the tile bytes or the fetch times. Rejected: the
  identity would then change with each fetch, and missions could not share
  blobs.
- Grid cells anchored at the mission origin. Rejected: products and cell
  identifiers of one mission could not serve another mission.
- One trait for navigation and guidance. Rejected: a host could not replace
  one and keep the other.
