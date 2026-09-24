# VNAV on GitHub Pages

The public site uses the `VNAV Pages` workflow. It builds Rust WASM from source.
It uses a fixed MapLibre revision and checks the XFeat, LighterGlue, and LoFTR model checksums.
It reads one fixed public data archive and checks its checksum.
The export checks each data chunk before it includes the chunk in the site.

The initial site address is `https://luofang34.github.io/Navigate/`.
Asset and worker URLs also support a domain root.
Do not add a CNAME until `vnav.navigate.sokoly.app` has the required DNS records.
Set the custom domain in the repository's Pages settings when DNS is ready.

## Site contents

- The Sokoly VNAV app, its licences, and a MapLibre Rust globe preview.
- Apache-2.0 matcher models and the MIT-licensed ONNX browser runtime.
- NAIP imagery and Mapzen terrain for the approved New Jersey demo area.
- A map-derived example image with its package identity and camera assumptions.

Camera uploads, saved observations, precise navigation priors, local source
paths, research weights, and private test images are not site assets.
The export takes explicit region IDs. It rejects a non-empty output folder.
Do not put unrelated files in the source webapp folder.

## Browser requirements

Use HTTPS and a browser with WebGPU and origin private file storage (OPFS).
The model runs in a worker. Some ONNX operators can use WASM on the CPU.
The runtime uses one WASM thread. It does not require cross-origin isolation
headers or SharedArrayBuffer. The map uses wgpu directly on its canvas.

Reference packages use OPFS. IndexedDB holds the package catalog and observations.
The service worker stores the app shell. The browser checks storage quota.
The user can request persistent storage. The browser can still remove site data.

GitHub Pages is a static host. It supports prepared package downloads and browser
matching. It cannot run the Rust imagery service. New provider coverage and
map-driven provider downloads require that service on a suitable host.
The static UI explains this limit and disables provider requests.

## Checks

Run the JavaScript tests before export:

```sh
for test in tools/visual-bench/tests/test_*.mjs; do node "$test"; done
```

Run `qa-xfeat.html` through the local test server to check pixel coordinates.
Run `qa-performance.html?run&report&synthetic` for the public package flow.
The synthetic check shares imagery with its reference. It is a pipeline test,
not independent geographic validation. The test pages are not public assets.
Use the real image and video input in the published UI for the final check.

Retrieval candidates, geometric acceptance, and measured geographic accuracy
are separate results. This demo does not establish absolute accuracy or night
performance. Its map-derived example does not establish real-flight performance.

The detailed New Jersey package has finer imagery than the regional package.
Both use the approved public demo area. Enter the local navigation prior before
matching. Package coverage is not a camera-location measurement.
