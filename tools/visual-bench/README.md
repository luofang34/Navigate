# Visual position demo

This demo uses Rust and WebAssembly (WASM). It has no Python runtime requirement.
It accepts a rough location and an image or video. The browser keeps camera media
local. It shows each geometric hypothesis with the MapLibre Rust fork.

The shared `navigate-visual` library provides camera geometry, planar retrieval
proposals, surface pose checks, candidate decisions, and evidence identities.
Its default matcher uses the CPU. Its `gpu` feature adds wgpu patch tracking.
The browser learned adapter uses ONNX Runtime WebGPU, with WASM fallback for some
operators. The map and reference renderer require WebGPU in this demo.

The shared `navigate-imagery` library provides area and route plans, checksums,
and immutable packages. Its default build supports native Rust and WASM. It has
no filesystem, network, Python, or GPU dependency. Its `native` feature adds a
NAIP provider, GDAL raster reads, and filesystem publication. The HTTP service
uses this feature. These libraries remain separate from the demo UI.

## Build

Use Rust, Node.js, `wasm-bindgen-cli`, and the `wasm32-unknown-unknown` target.
Use the `wasm-bindgen-cli` version in `wasm-preview/Cargo.lock`.
Put the MapLibre fork beside Navigate. The renderer dependency uses this layout.
Use the MapLibre fork at commit
`5c323427e29c1796003c5e7052b2887f6d35c173`, or a later commit that includes
[the required renderer changes](https://github.com/luofang34/maplibre-rs-experimental/pull/32).
Run commands from the Navigate root:

```sh
node tools/visual-bench/build_web.mjs
node tools/visual-bench/prepare_browser_models.mjs
cargo build --release --manifest-path tools/visual-service/Cargo.toml
```

The model preparation command downloads the Apache-2.0 XFeat ONNX release.
It checks the pinned SHA-256 digest. It installs the pinned ONNX browser runtime.
The public export includes their licences and model provenance.
Model loading starts when the user requests matching.

The native provider requires GDAL development files and PROJ data. GDAL 3.6 to
3.12 works with the selected Rust binding. GDAL 3.13 requires a newer binding.
On a machine with several installations, select one consistent set of headers,
libraries, and `pkg-config` metadata. Do not substitute a different ABI version.

The public matcher uses XFeat features and mutual nearest descriptor matches.
Its adapter owns preprocessing, feature limits, descriptor filters, and GPU setup.
The release uses an 800 by 600 coordinate system. The adapter preserves image
aspect ratio and maps model coordinates back to input pixels.
Higher matching detail increases the feature budget and geometry resolution.
It does not increase this model's internal image resolution.
The optional [model probe](../visual-inference/README.md) is separate from the
public demo. Do not put research-only SuperGlue weights in the public site.

## Start the Rust service

```sh
RUST_LOG=info tools/visual-service/target/release/navigate-visual-service \
  --state target/visual-web-data \
  --webapp tools/visual-bench/webapp \
  --origin http://127.0.0.1:8080 \
  --bind 127.0.0.1:8080
```

An optional `--catalog /path/to/regions.json` imports schema-2 reference folders:

```json
[{"id":"test-region","label":"Test region","path":"/absolute/path/to/map"}]
```

Keep a stable origin to reuse browser storage. For deployment, set the public
HTTPS origin and use the deployment's HTTPS reverse proxy. `NAVIGATE_DATA`,
`NAVIGATE_ORIGIN`, and `NAVIGATE_CATALOG` are alternatives to the corresponding
command options. The service has no camera-upload or native-inference endpoint.
It runs one coverage download at a time and accepts at most three pending jobs.
Package files survive restart. Active job records do not. Use one service instance
for each package directory. Provide normal deployment access and request limits.

Build the runtime image from the Navigate root after preparing browser assets:

```sh
docker build -f tools/visual-bench/Dockerfile.web -t navigate-visual .
docker run --rm -p 8080:8080 -v navigate-data:/data \
  -e NAVIGATE_ORIGIN=https://navigate.example navigate-visual
```

The runtime image contains the Rust executable, GDAL, and browser assets. It does not contain
Python or model export tools. Build and test the image in your deployment environment.

## Static WASM deployment

Export verified packages and prepared browser assets to a static site folder:

```sh
node tools/visual-bench/export_web.mjs target/visual-web-data target/visual-site \
  naip-2864a5ec8e4abb24
```

Serve that folder through HTTPS or localhost. The browser reads `catalog.json`,
package manifests, and chunks with ordinary file requests. Image and video
localization runs in the browser. Prepared packages work without a server API.
New provider coverage requires the Rust service deployment.
The export requires an empty output directory and explicit region IDs.
It checks chunk hashes and exports only selected NAIP packages.
It excludes test media, test pages, and research-only model weights.
See [Pages deployment](PAGES.md) for the public build.

## Use the demo

1. Select a region, or download an area or route corridor.
2. Set the precise prior, position radius, height above ground, and sensor FOV.
3. Download the package and wait for its checksum checks.
4. Select an image or video. For video, choose a frame or a sampled sequence.
5. Choose matching detail. Higher detail costs more time and memory.
6. Select **Estimate camera pose**. Results appear after each frame.
7. Select a frame and a geometric hypothesis. Use **Cancel processing** to stop.

Drag the map to pan. Use right-drag or Shift-drag to turn the camera. Use the
wheel or the in-frame **+** and **−** controls to change height. Use WASD to move.
Globe projection is always active. Zoom out to see the globe. The in-frame
**Reset camera** control returns to the selected pose. The display uses a direct wgpu
canvas at the screen pixel ratio. Display size does not change calibration.
Terrain uses tile-relative globe coordinates to retain small features at low altitude.

Camera clearance uses the loaded DEM beneath the camera. Its minimum is 10% of
the entered AGL, limited to 20 through 100 metres. A missing DEM is unknown.
The display then stays at least 10 km above the package datum. These controls
protect the preview camera. They do not provide an aircraft terrain warning.
A display adjustment does not change the estimated pose or its evidence.

Enable **Load imagery and terrain as the map moves** to fetch the viewed area.
This sends that area to Microsoft Planetary Computer and Mapzen/AWS. Downloads
use the same verified packages and OPFS storage as area and route downloads.
The loader reuses stored coverage even when provider downloads are disabled.
It requests zoom 18 below 400 metres and permits one request at a time.
Source resolution still limits image detail. It does not create new source detail. Globe-scale
views use coarse context. A view retains at most four display packages. Select a
region to start another view. Select a downloaded region to use it for localization.
Dynamic display data cannot change an active observation's reference data or prior.

Models and their runtime load when matching starts. The worker keeps model setup,
preprocessing, inference, reference rendering, and geometry work off the UI thread.
Cancellation terminates that worker. A cancelled result cannot enter acceptance.

The FOV input is the diagonal FOV of a 4:3 sensor. The demo assumes the full sensor
long side, no digital zoom, and no lens distortion. These are assumptions, not
calibration measurements. Intrinsics scale with the actual resized image dimensions.
A rejected frame shows the region overview. Unresolved candidates remain separate.

## Match detail and accuracy

Fast mode uses a 640-pixel observation, one crop scale, and eight headings.
Balanced mode uses a 960-pixel observation, five scales, and 72 headings.
Detailed mode uses 1280 pixels, five scales, and 72 headings. Retrieval uses a
640-pixel image in all modes. The denser modes retain all eight fast-mode headings.
Pose refinement uses the selected observation
resolution and a larger feature budget at higher detail. The wider search takes
more processing time. It does not reduce the geometric acceptance requirements.
Invalid source pixels and their immediate borders cannot produce features.

Retrieval remains bounded. It can miss a valid place or orientation. More inliers
or a lower reprojection error do not prove lower absolute geographic error.
Compare residuals at a common image scale. Higher-resolution pixels are smaller.
The demo has no independent geographic ground truth or validated night result.
A unique selection means unique among evaluated candidates, not unique worldwide.

## Offline data and provider scope

The Rust provider queries public NAIP records from Microsoft Planetary Computer.
It reads georeferenced COG ranges through GDAL. It produces 512-pixel imagery tiles
with validity masks and 256-pixel Mapzen Terrarium terrain tiles at zoom 14.
NAIP coverage is limited to parts of the US. Source dates and resolution vary.
The provider keeps source identities, dates, URLs, checksums, and processing identity.
Source elevation datum and absolute registration error remain unverified.

`POST /api/coverage-plan` validates an area or route without provider access.
`POST /api/coverage-download` creates a package job. Use `GET /api/downloads/{id}`
for progress. Areas use `bounds: [west,south,east,north]`. Routes use longitude,
latitude pairs and `buffer_m`. Imagery zoom is 14 to 18. The limit is 400 imagery
tiles. Antimeridian and polar selections are unsupported. Split large routes.

The download area is separate from the precise navigation prior. A larger area
can reduce location precision disclosed by the request. It does not provide
anonymity. Never offset source georeferencing or the navigation prior.

OPFS holds bulk data. IndexedDB holds package and observation records. A manifest
binds each asset to an immutable chunk, byte range, and checksum. Catalogue
publication occurs only after chunk writes succeed. Corrupt or missing chunks
require repair. CacheStorage holds the app shell and runtime resources used by
the browser. Origin quota and storage eviction still apply.

`navigate-data` defines `DataStore` and `RandomAccess`. `navigate-data-fs` supplies
native reads. The WASM host supplies OPFS range reads. The package builder accepts
a storage callback. Core storage and package types do not call browser APIs.

## Replace an algorithm or backend

Rust matchers implement `navigate_visual::ImageMatcher` and return `PixelMatch`.
Native and WASM callers can use `planar_proposal` with `GroundCorrespondence`.
Its output is a retrieval proposal. It is not an accepted location. Another
retriever can supply arbitrary orientations without this planar model.

The browser accepts a matcher in `LocalizationPipeline`. Required methods are
`initialize(progress)`, `matchImages(reference, query, keys)`, and `close()`.
Images contain grayscale bytes, dimensions, and optional validity bytes.
The result contains pixel pairs and a backend identity. Model loading, feature
scores, device selection, and runtime state remain inside the adapter. The
optional `retrievePairs` method returns reference and query indices. Without it,
the pipeline evaluates all pairs. No general plugin framework is required.

`PoseVerifier` checks rendered surface depth against the original navigation prior.
It does not require a homography. `CandidateResults` retains separate alternatives.
A repeated fit replaces the result for that candidate. It does not add independent
evidence. Missing imagery, missing terrain, and unsupported surfaces stay invalid.
Rendered depth describes the available world model. It is not independent scene
verification. The demo does not reconstruct buildings or update the map.

Reports retain observation and map identities. Geometry covariance covers local
image geometry. It excludes unknown map, registration, calibration, and association
errors. Shared evidence and correlation remain unquantified. Reprocessing an image
must not create confidence. No pose fusion or map write-back occurs here.

## Checks

```sh
./ci.sh
cargo test -p navigate-imagery --features native
cargo check -p navigate-imagery --target wasm32-unknown-unknown
cargo test --manifest-path tools/visual-service/Cargo.toml
cargo clippy --manifest-path tools/visual-service/Cargo.toml --all-targets -- -D warnings
node --test tools/visual-bench/tests/test_*.mjs
cargo test --manifest-path tools/visual-bench/wasm-preview/Cargo.toml
```

The native imagery tests need a compatible GDAL installation. GPU tests require
an adapter. Browser tests use ignored local files `webapp/models/test-input.png`
and `test-video.mp4`. Run `tests/browser_server.mjs` with a loopback service URL,
a report path, and an optional port. Open `/qa-quality.html` to compare matching
detail, `/qa-performance.html?run=1&report=1` for GPU and camera checks, or
`/qa-app.html` for upload, selected video frame, saved result, and theme checks.
Use `/qa-flow.html` for frame decoding checks. Use `/qa-map.html` for
terrain clearance and the rendered difference between coarse and fine NAIP data.
The map test needs prepared zoom-16 and zoom-17 NAIP packages. The production service has no
test-report endpoint. Browser tests do not establish geographic accuracy.

For the nine-frame DJI evaluation, put the local `DJI_0029_frame_1.png` through
`DJI_0029_frame_5.png` and `DJI_0030_frame_1.png` through
`DJI_0030_frame_4.png` files in `webapp/models/test-frames/`. Open
`/qa-dataset.html?region=<region-id>`. The default profile is `balanced`.
The default prior uses the package centre. Set `lat`, `lon`, `radius`, and `agl`
in the local test URL to select another prior. Angles use degrees. Distances use
metres. Keep precise test priors in the local test URL.
The report records each retrieval result and geometric decision. A completed
evaluation can contain rejected frames. It is not a successful localization check.
The public export excludes these test pages and private input files.

## Optional asset maintenance

`prepare_globe_context.py` generates the display context from Natural Earth GeoJSON.
It uses Pillow from `requirements-web.txt`. It is not a service or web runtime
dependency. Use `prepare_browser_models.mjs` to install the public XFeat model.

## Browser matcher boundary

Supply a matcher to the `LocalizationPipeline` constructor. Implement
`initialize(progress)`, `matchImages(reference, query, keys)`, and `close()`.
Images contain `gray`, `width`, and `height`. A reference can contain a
`valid` pixel mask. Coordinates use input pixel centres.
Return `{pairs, backend_identity}`. Each pair contains `reference: [x, y]`
and `query: [x, y]`. Keep model tensors and backend scores inside the adapter.
An optional `retrievePairs` method returns reference and query indices.
The pipeline owns priors, rendered references, pose checks, and final decisions.
A retrieval result is not an accepted pose. Alternatives stay separate.

The public example image is made from the reference map. It checks the pipeline.
It does not measure independent geographic accuracy. In the nine-frame DJI
browser check, balanced mode produced geometric hypotheses for seven frames
with the supplied reference data. It produced none with public NAIP data.
These checks used a 500 m search radius and an assumed 110 m camera height
above ground. They did not establish correct geographic associations.
Night operation and absolute geographic accuracy have not been validated.
