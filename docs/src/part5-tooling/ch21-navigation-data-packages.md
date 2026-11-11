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
| `key.source` | name | a registry source: `esri_world_imagery`, `mapzen_terrarium`, `synthetic` |
| `revision` | name or object | `"unversioned"`, `{"release": n}`, or `{"collected": day}` |
| `horizontal_datum` | name | `wgs84` |
| `vertical_datum` | object or `null` | elevation: `{"kind": "ellipsoidal"}`, `{"kind": "orthometric", "geoid": "egm96"}`, or `{"kind": "unknown"}`; imagery: `null` |
| `encoding` | name | `png`, `jpeg`, `webp`, or `terrarium_png` |
| `tile_pixels` | integer | pixels on one side of a tile |
| `accuracy` | object | `horizontal_mm`, `vertical_mm`, `acquired` (day), `age_growth_mm_per_year`, each an integer or `null`; `surface`: `bare_earth`, `first_return`, or `unknown` |
| `correlation_zoom` | integer | zoom of the correlation cells, 16 by default |
| `index` | object | `blob` (digest), `bytes`, `tiles`, `gaps` |

A day is a whole number of days from 2000-01-01.

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
