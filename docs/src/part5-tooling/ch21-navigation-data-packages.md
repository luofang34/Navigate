# Chapter 21: Navigation Data Packages

A navigation data package holds the map layers that a flight needs.
`navigate-data` defines the package: its manifest, its identity, its
checks, and its ports. The host fetches and installs the data. Navigate
only reads it. [ADR-0022](../../adr/0022-navigation-data-contract-mission-packages-and-the-onboard-boundary.md)
records the decision.

## Technical names

- **Blob** is one encoded file, for example one imagery tile. Its digest
  is its name.
- **Manifest** is the canonical JSON file that lists the layers of a
  package and the blobs of each layer.
- **Package identity** is the SHA-256 of the manifest bytes. It is the
  `manifest_sha256` of `navigate_visual::MapRevision`.
- **Coverage** is the set of tile addresses that a layer declares. Each
  address is a tile or a gap.
- **Correlation cell** is one XYZ tile at zoom 16 of the fixed global grid.

## 1. Store layout

```text
chunks/<sha256>              one blob, written once
packages/<sha256>.json       one manifest, written after its blobs
```

The paths are logical. A host maps them to a directory, to OPFS, or to its
own store (Pilotage ADR-0044). A path has no empty, `.`, or `..` part and
no `\` separator. Packages that cover the same ground share blobs. A store
that has blobs but no manifest holds no package.

## 2. Manifest

| Field | Content |
|---|---|
| `format` | `navigate.data.package` |
| `version` | `1` |
| `release_id` | Release identifier that the host gives |
| `grid` | `{"scheme": "web_mercator_xyz", "correlation_zoom": 16}` |
| `layers[]` | The layers, in identifier order |

Each layer:

| Field | Content |
|---|---|
| `id` | Identifier, unique in the package, for example `imagery` |
| `kind` | `imagery` or `elevation` |
| `source` | `id`, `revision`, and `attribution` of the provider product |
| `encoding` | `png`, `jpeg`, `webp`, or `terrarium_png` |
| `tile_pixels` | Pixels on one side of a tile |
| `vertical_datum` | For elevation: `{"kind": "ellipsoidal"}`, `{"kind": "orthometric", "geoid_model": "egm96"}`, or `{"kind": "unknown"}`. For imagery: `null` |
| `accuracy` | `horizontal`, `vertical`, `acquired`, `age_growth_per_year`, `surface` |
| `zooms[]` | For each zoom, in increasing order: `zoom`, `tiles[]` (`x`, `y`, `blob`, `bytes`), and `gaps[]` (`[x, y]`) |

A sigma is `{"known": 2.5}` (metres) or `"unknown"`. `acquired` is
`{"date": "2024-06"}` or `"unknown"`. `surface` is `bare_earth`,
`first_return`, or `unknown`.

### Canonical form

1. The encoding is compact JSON, with the fields in the order of the
   tables.
2. Layers are in identifier order. Zoom levels increase. Tiles and gaps are
   in row order, then column order. No address is both a tile and a gap.
3. Unknown fields are refused.

A reader decodes the manifest, encodes it again, and refuses the bytes if
they differ. Fetch times, host paths, and URLs are not in the manifest, so
the same tiles from the same source give the same identity.

## 3. Checks

`Package::open(store, identity)`:

1. reads `packages/<identity>.json`;
2. computes its SHA-256 and compares it with `identity`;
3. decodes the manifest and checks its canonical form, its format, its
   order, and its values.

`Package::tile(store, layer, address)` gives one of:

- `Present(bytes)`, after it compares the size and the SHA-256 of the blob
  with the manifest;
- `Gap`, for an address that the layer declares as a gap;
- `Outside`, for an address outside the declared coverage.

A blob that is missing or changed is an error. It is never a gap.
`Package::verify_all` reads and checks each blob once.

## 4. Writing a package

`PackageWriter` builds one package:

1. `add_layer` declares a layer without tiles.
2. `add_tile` writes the blob to `chunks/<sha256>` at once and declares the
   tile. `add_gap` declares an address that the source did not have.
3. `seal` checks that the store holds each blob, then writes the manifest.
   It returns the package identity.

The order of the `add_tile` calls has no effect on the identity.

## 5. Vertical datum

Heights are stored as the provider supplies them. Terrarium tiles encode
heights above mean sea level:

```text
height_m = R × 256 + G + B / 256 − 32768
```

`VerticalDatum::to_ellipsoidal` converts a stored height with a
`navigate_geodesy::GeoidUndulation` model and its name, which the host
supplies:

```text
h_ellipsoid = H_orthometric + N(lat, lon)
```

Without a model of the same name, the result is `UnknownUndulation` with a
bound of 107 m (the EGM96 undulation is between approximately −107 m and
+86 m). The error budget must carry that bound.

## 6. Fixed grid

The grid is the XYZ tile pyramid. It has no anchor. `GridCell` is the zoom
16 tile that holds a point. A correlation cell is approximately 611 m on a
side at the equator and 460 m at 41° latitude. Map error is taken as
shared inside one cell and independent between cells. Derived products
tile on the same pyramid, so two missions over the same ground use the
same cells.

## 7. Ports

- `ReadPort::read(path)` gives the bytes at a logical path, or `None`.
- `WritePort::write(path, bytes)` writes a file once.
  `WritePort::contains(path)` tells if the store holds a file.
- `FetchPort` is implemented by the host. `fetch(key)` gives `Tile(bytes)`
  or `Absent` for a `TileKey` (source, revision, layer, zoom, column, row).

`MemoryStore` holds files in memory. `DirectoryStore` (feature `std`) holds
them in a directory and writes each file through a temporary file and a
rename.

## 8. Resource model

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

## 9. Preparing a package

`plan_tiles` turns the resource plan into tiles: for each band, the
corridor of the band half width along the leg and `far_m` past its end.
`prepare` fetches each tile once through the host `FetchPort`, writes a
gap for each `Absent` answer, and seals the package. It refuses a plan
over `max_tiles` before the first fetch.

`navigate-visual-pipeline package` is a host for this step:

```sh
navigate-visual-pipeline package \
  --plan plan.json --store store --cache tile-cache \
  --imagery-url 'https://imagery.example/{z}/{y}/{x}' \
  --imagery-source example-imagery --imagery-revision 2025-06 \
  --dem-url 'https://elevation.example/terrarium/{z}/{x}/{y}.png' \
  --dem-source mapzen-terrarium
```

The plan file gives `release_id`, `waypoints` (`lat_deg`, `lon_deg`,
`altitude_m` above mean sea level), `cameras` (`width_px`, `height_px`,
`horizontal_fov_deg`, `depression_deg`), `search_radius_m`,
`deviation_m`, and `max_range_m`. The tool fetches zoom 10 Terrarium tiles
for the ground heights, plans, fetches the planned tiles into the cache on
parallel threads, and seals the package. The cache key is the tile key
(`<source>/<revision>/<z>/<x>/<y>`), not the URL. `--plan-only` prints the
resource plan and fetches nothing else. A `file://` template reads local
files. An HTTP status of 404 or 204 is a gap.
