# Chapter 21: Navigation Data Packages

A navigation data package holds the map layers that a flight needs.
`navigate-data` defines the package: its manifest, its identity, its tile
indexes, its checks, and its ports. The host fetches and installs the
data. Navigate only reads it.
[ADR-0022](../../adr/0022-navigation-data-contract-mission-packages-and-the-onboard-boundary.md)
records the decision.

## Technical names

- **Blob** is one stored file, for example one imagery tile or one tile
  index. Its SHA-256 is its name.
- **Manifest** is the canonical JSON document of a package.
- **Package identity** is the SHA-256 of the manifest bytes. It is the
  `manifest_sha256` of `navigate_visual::MapRevision`.
- **Tile index** is the canonical binary blob of one layer. It binds each
  address of the declared coverage to a blob digest or a gap.
- **Descriptor** is the JSON file beside the manifest with labels,
  attribution, licence text, and host notes. It is outside the identity.
- **Correlation key** is (source, revision, grid cell at the correlation
  zoom of the layer).

## 1. Store layout

```text
chunks/<sha256>                        one blob: a tile or a tile index
packages/<sha256>.json                 one manifest, written after its blobs
packages/<sha256>.descriptor.json      its descriptor, not hashed
```

The paths are logical. A host maps them to a directory, to OPFS, or to its
own store (Pilotage ADR-0044). A path has no empty, `.`, or `..` part and
no `\` separator. Packages that cover the same ground share blobs. A store
that has blobs but no manifest holds no package. The host deletes a blob
when no installed manifest, tile index, product chunk, or product list
refers to it; Navigate never deletes.

## 2. Manifest

| Field | Type | Content |
|---|---|---|
| `format` | text | `navigate.data.package` |
| `version` | integer | `1` |
| `body` | name | `earth` |
| `ellipsoid` | name | `wgs84` |
| `grid` | name | `web_mercator_xyz` |
| `layers[]` | objects | in `key` order |

Each layer:

| Field | Type | Content |
|---|---|---|
| `key.kind` | name | `imagery` or `elevation` |
| `key.source` | name | a registry source (`esri_world_imagery`, `mapzen_terrarium`, `synthetic`) or `declared:<sha256>` of a source declaration |
| `revision` | name or object | `"unversioned"`, `{"release": n}`, or `{"collected": day}` |
| `horizontal_datum` | name | `wgs84` |
| `vertical_datum` | object or `null` | elevation: `{"kind": "ellipsoidal"}`, `{"kind": "orthometric", "geoid": "egm96"}`, or `{"kind": "unknown"}`; imagery: `null` |
| `encoding` | name | `png`, `jpeg`, `webp`, or `terrarium_png` |
| `tile_pixels` | integer | pixels on one side of a tile |
| `accuracy` | object | `horizontal_mm`, `vertical_mm`, `acquired` (day), `age_growth_mm_per_year`, each an integer or `null`; `surface`: `bare_earth`, `first_return`, or `unknown` |
| `correlation_zoom` | integer | zoom of the correlation cells, 16 by default |
| `index` | object | `blob` (digest), `bytes`, `tiles`, `gaps` |

A day is a whole number of days from 2000-01-01.

### Source declaration

A source outside the registry has the key `declared:<sha256>`, the digest
of the canonical bytes of its declaration:

| Field | Type | Content |
|---|---|---|
| `version` | integer | `1` |
| `kind` | name | `imagery` or `elevation` |
| `ground_sample_mm` | integer | nominal ground sample distance, above zero |
| `horizontal_mm`, `vertical_mm` | integer or `null` | one-sigma that the provider states |
| `collected` | integer or `null` | collection day |
| `provider_id` | digest | digest of a stable identifier that the provider chooses |

### Canonical form

1. The encoding is compact JSON, with the fields in the order of the
   tables. Unknown fields are refused.
2. Numbers are integers only. There is no floating-point value.
3. Strings are only names from the closed vocabularies above or 64-digit
   lower-case hexadecimal digests.
4. Layers are in key order (kind, then source), with no repeats.
5. Each optional value is present, as `null` when it is absent.

A reader decodes the manifest, encodes it again, and refuses the bytes if
they differ.

### Unknown values

- A manifest of another version is refused with `DataError::Version`.
  There is one current schema and no reader for other versions.
- An unknown body, ellipsoid, grid, layer kind, source, encoding,
  horizontal datum, vertical datum, or geoid model is refused with
  `DataError::Unsupported`.

## 3. Tile index

```text
"NVDIDX01" | record × N
record = zoom u8 | column u32 | row u32 | size u64 | digest [u8; 32]   (49 bytes, little endian)
```

The records are sorted by zoom, row, and column, with no repeats. A gap
has size zero and a zero digest. A tile has a size above zero and the blob
digest. The manifest holds the index digest, its size
(`8 + 49 × records`), and its tile and gap counts.

## 4. Checks

`Package::open(store, identity)`:

1. reads `packages/<identity>.json`, computes its SHA-256, and compares it
   with `identity`;
2. decodes the manifest and checks its canonical form, its version, its
   vocabulary, and its order;
3. reads each tile index by ranges and checks its digest, its size, its
   order, each record, and its counts.

`Package::tile(store, layer, address)` finds the record by binary search
over ranged reads and gives one of:

- `Present(bytes)`, after it compares the size and the SHA-256 of the blob
  with the record;
- `Gap`, for an address that the layer declares as a gap;
- `Outside`, for an address outside the declared coverage.

A blob that is missing or changed is an error. It is never a gap.

## 5. Writing a package

`PackageWriter` builds one package:

1. `add_layer` declares a layer with its `LayerSpec`.
2. `add_tile` writes the blob to `chunks/<sha256>` at once and declares the
   tile. `add_gap` declares an address that the source did not have.
3. `seal` checks that the store holds each tile blob, writes the tile
   indexes, the manifest, and the descriptor. It returns the package
   identity.

The order of the `add_tile` calls and the descriptor have no effect on
the identity.

## 6. Vertical datum

Heights are stored as the provider supplies them. Terrarium tiles encode
heights above mean sea level:

```text
height_m = R × 256 + G + B / 256 − 32768
```

`VerticalDatum::to_ellipsoidal` converts a stored height with a
`navigate_geodesy::GeoidUndulation` model that the host supplies, together
with the `GeoidModel` that it implements:

```text
h_ellipsoid = H_orthometric + N(lat, lon)
```

Without a model of the stated geoid, the result is `UnknownUndulation`
with a bound of 107 m. The error budget must carry that bound.

## 7. Grid and correlation

The grid scheme is the Web Mercator XYZ pyramid up to 85.051 128° of
latitude. A request past the limit is refused. A cell is not of constant
size: a zoom 16 cell is approximately 611 m on a side at the equator,
460 m at 41°, and 306 m at 60°.

Map error is taken as shared inside one correlation cell and independent
between cells. The key of a cell is (source, revision, cell at the
`correlation_zoom` of the layer): two packages cut from the same imagery
give the same keys.

## 8. Ports

- `ReadPort::open(path)` gives an open file or `None`. The file has
  `len()` and `read_at(offset, length)`. This is the shape of the
  Pilotage ADR-0044 read port.
- `WritePort::write(path, bytes)` writes a file once.
  `WritePort::contains(path)` tells if the store holds a file.
- `FetchPort::fetch(key)` is implemented by the host. It gives
  `Tile(bytes)` or `Absent` for a `TileKey` (source, revision, layer kind,
  zoom, column, row).

`MemoryStore` holds files in memory. `DirectoryStore` (feature `std`)
holds them in a directory and writes each file through a temporary file
and a rename.

## 9. Resource model

`plan_resources` computes what each leg of a flight needs before any
fetch. Its inputs are the legs (position and altitude), the cameras
(focal length, image size, depression of the optical axis), the search
radius, the allowed deviation, the farthest slant range, and the limits
and measured sizes of the imagery and elevation sources. A
`GroundHeight` port supplies the ground heights, for example from coarse
Terrarium tiles.

The zoom rule is the rule of the map renderer: one texel for one pixel.

```text
texel(z, φ) = 40 075 016.69 m × cos φ / (T × 2^z)     T = tile pixels
pixel(d)    = d / f                                  d = slant range, f = focal length in pixels
z(d)        = log2(f × 40 075 016.69 m × cos φ / (T × d))
```

For tiles of 256 pixels, a texel at zoom 16 and 41.7° latitude is
1.78 m. The renderer uses tiles of 512 pixels, so its level for the same
texel is one less.

For each camera, the model finds the slant range of the steepest and the
shallowest ray and gives one band for each zoom between them. Each band
has a ground range ahead of the vehicle and a half width: the slant range
at the far edge times the tangent of the horizontal half field of view,
plus the search radius and the deviation. The elevation band of an
imagery band is `elevation_offset` levels coarser. The size estimate uses
the measured size of each source at a reference zoom and a growth factor
for each level.

A higher leg needs coarser levels, and costs less for each kilometre. The
far field of an oblique camera uses coarser levels than its near field.
The altitude band of a leg is `floor(log2(agl / 100 m))`.

## 10. Preparing a package

`plan_tiles` turns the resource plan into tiles: for each band, the
corridor of the band half width along the leg and `far_m` past its end.
`prepare` fetches each tile once through the host `FetchPort` with the
source and revision of the `LayerSpec` of its layer, writes a gap for each
`Absent` answer, and seals the package with the descriptor that the host
supplies. It refuses a plan
over `max_tiles` before the first fetch.

`navigate-visual-pipeline package` is a host for this step:

```sh
navigate-visual-pipeline package \
  --plan plan.json --store store --cache tile-cache \
  --imagery-url 'https://imagery.example/{z}/{y}/{x}' \
  --imagery-source esri_world_imagery --imagery-revision unversioned \
  --dem-url 'https://elevation.example/terrarium/{z}/{x}/{y}.png' \
  --dem-source mapzen_terrarium
```

The plan file gives `release_id`, `waypoints` (`lat_deg`, `lon_deg`,
`altitude_m` above mean sea level), `cameras` (`width_px`, `height_px`,
`horizontal_fov_deg`, `depression_deg`), `search_radius_m`,
`deviation_m`, and `max_range_m`. The tool fetches zoom 10 Terrarium tiles
for the ground heights, plans, fetches the planned tiles into the cache on
parallel threads, and seals the package. The cache key is the tile key
(`<source>/<revision>/<z>/<x>/<y>`), not the URL. A source is a registry
name; a revision is `unversioned`, `release:N`, or `collected:YYYY-MM-DD`. `--plan-only` prints the
resource plan and fetches nothing else. A `file://` template reads local
files. An HTTP status of 404 or 204 is a gap.

## 11. Derived products

Derived products have two levels.

```text
products/chunks/<sha256>.json     one product chunk, reusable across missions
products/lists/<sha256>.json      one product list, per mission
chunks/<sha256>                   recipes and entry blobs
```

A **product chunk**:

| Field | Content |
|---|---|
| `format`, `version` | `navigate.data.product_chunk`, `1` |
| `kind` | `keyframes`, `corridor_landmarks`, a reserved kind (`terrain_slopes`, `horizon_profiles`, `scalar_field`, `star_pairs`, `station_list`, `ephemeris_segments`), or an unknown name |
| `recipe` | digest of the recipe document |
| `binding.cell` | grid cell (zoom, x, y) or `null` |
| `binding.altitude_band` | integer or `null` |
| `binding.inputs` | sorted digests of the input blobs |
| `binding.route` | route digest or `null` |
| `binding.envelope` | `lateral_mm`, `vertical_mm`, `heading_urad`, or `null` |
| `binding.calibration` | `width`, `height`, `fx_upx`, `fy_upx`, `cx_upx`, `cy_upx` (micro-pixels), or `null` |
| `entries[]` | `key` (increasing integer), `pose` (`lat_ndeg`, `lon_ndeg`, `height_mm`, `heading_urad`) or `null`, `blob`, `bytes` |

A **recipe** is `navigate.data.recipe` version 1 with `algorithm` and
`code_version` (constants of the builder code), `weights` (digest or
`null`), `precision` (`f16`, `f32`, `f64`), and `parameters`: integers by
name, with the unit in the name (`spacing_mm`, `far_mm`).

A **product list** is `navigate.data.product_list` version 1 with
`package` (the package identity) and `chunks[]` (`kind`, `chunk`), sorted.

The documents follow the canonical rule of the manifest. The chunk names
no package, so a chunk serves each mission whose package holds its input
blobs (`ProductChunk::applies_to`). A route chunk serves only its route
digest, inside its envelope, in its altitude band. `open_list` refuses a
list of another package. An unknown kind in a list is carried and
enumerated (`ProductList::unknown_kinds`); `open_chunk` refuses to read a
chunk of an unknown kind.

### Keyframes

`navigate_visual_render::build_keyframes` renders one reference view at
each planned pose along the legs, at a fixed spacing, with the sparse
adaptive ray marcher, and seals one `keyframes` chunk. Its binding holds
the route digest, the envelope, the altitude band, the calibration, and
the digests of each tile blob that a view was rendered from. Each entry
blob is:

```text
"NVKEYV01" | u32 width | u32 height | luma (width × height bytes) | depth (width × height f32, little endian)
```

The depth is along the optical axis, in metres. Zero marks sky or missing
data.

### Corridor landmarks

`navigate_visual_flight::Surface::from_package` loads one imagery zoom and
one elevation zoom of a package and gives the digests of the tile blobs
that it holds. Its `catalog_sha256` is the inputs digest of those blobs
(`inputs_digest`). `build_corridor_bank_on_surface` writes that digest into
the `NVKTILE1` bank, and `write_corridor_product` stores the bank as a
`corridor_landmarks` chunk with its recipe. The bank tiles are in an east,
north, up frame at the centre of the surface, not on the fixed grid, so
one bank is one chunk with `cell` set to `null`.

## 12. The onboard boundary

`navigate-onboard` holds the composition that a host links:

- `Navigator` takes `SensorInput`s and gives a `NavigationSolution`. It
  writes one `AcceptanceRecord` for each measurement into a sink that the
  caller supplies: `Accepted`, `Rejected` with Navigate's reason, or
  `Unused` with a reason.
- `FlightDirector` sequences the route and gives the guidance setpoint
  from a solution. It holds no control authority. `RouteDirector` owns its
  waypoints, keeps the active index and the completion flag, and resumes a
  `navigate_route::PlanExecution` from them at each step, with
  `navigate_guidance::guide` or `guide_velocity`.
- Both traits give `snapshot` and `restore`, so that a host can fork a run
  and restore it. A navigator gives its data requirements before load and
  a report for each product chunk after load.
- `Onboard` owns one of each as trait objects.

The host declares the `SensorSuite` once: for each sensor, the
identifier, the kind, the imager model and the mount of a camera, the
rate, and the error budget. An input carries the sensor identifier, the
acquisition and delivery times, and the data. A sensor is unavailable only
when its inputs do not come.

`ReferenceNavigator` feeds every declared sensor into one 22-state
inertial filter on WGS84: IMU increments, GNSS position and velocity
fixes (with the declared budget when a fix has no covariance), barometric
altitude (with the barometer bias state), star-camera frames (identified
lost in space or with the navigator's own attitude as the prior, then one
line of sight for each star, with consecutive sequence numbers; each
record of a star frame names its identification mode, `tracking` or
`lost_in_space`), and observations that a sensor makes itself.
