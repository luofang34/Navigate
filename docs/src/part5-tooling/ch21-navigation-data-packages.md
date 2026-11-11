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
