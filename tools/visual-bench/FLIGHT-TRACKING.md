# Camera sequence processing

The browser keeps its worker and model sessions between observations. A package,
camera, or matching configuration change creates a new worker. Cancellation
terminates the worker. A failed initialization cannot return a partial session.

An image sequence uses filename order. Its flight times are unknown. A video
uses browser seek times. These times are not independently verified frame PTS.
The file size limit applies to images. The browser streams video from the file.

## Geometry boundary

`PoseVerifier::evaluate` returns the normal acceptance result. It can also return
a bounded refinement pose when spatial support fails. That pose is only a seed.
A new render and a new match must pass all geometric checks before acceptance.
The minimum inlier count and spatial support checks do not change.

`PoseVerifier::track` takes the current frame, the previous frame, and surface
depth rendered at the previous estimated pose. Matches refer to the two camera
images. They do not refer to the rendered image. The result is a
`TrackingProposal`. It is not an `Estimate` or an independent map measurement.

Tracking retains the initial map candidate and observation identities. Results
state that pose, calibration, surface error, and correlation are unknown.
Tracking does not report a geographic covariance. Reprocessing does not add
confidence. The browser keeps geographic alternatives separate. It attempts map
verification at intervals. A tracking failure starts a new area search.

The current surface is rendered terrain. It does not include buildings. Rendered
depth is not independently measured scene geometry. A small image residual can
coexist with a large geographic error.

## Matcher boundary

`ImageMatcher` returns Navigate pixel correspondences. Model loading, device
selection, image transforms, tensor shapes, and model scores stay in adapters.
A host can pass the same correspondences to map verification or relative
tracking. Neither geometric check requires a homography.

The browser uses XFeat for retrieval. The balanced and detailed profiles load
LoFTR-DS when pair matching starts. The fast profile uses XFeat and LighterGlue.
The retrieval shortlist limits duplicate nearby positions and headings. Its
ranking is not a location probability. The current area retrieval targets
near-downward camera views. The camera and verification types retain full 3D
orientation.

The native `LoFtrMatcher` uses Rust `ort`. It accepts an explicit model file and
execution configuration. The browser uses ONNX Runtime WebGPU with WASM fallback
for unsupported operators. Deployment does not require Python. Python scripts
build and check model assets only.

## Display and export

The map follows a selected supported hypothesis. Relative tracking has its own
result label. It does not increase the count of accepted map hypotheses.
GeoJSON retains all supported points and their evidence identities. Its line
uses the first supported hypothesis per frame as an illustration. The line is
not a fused navigation track. Rejections and map identity changes split lines.
Altitude remains a property because its vertical datum is unverified.

Independent DJI positions and orientations are unavailable. Real-frame checks
measure retrieval, geometric support, and execution time. They do not establish
absolute camera accuracy. Night operation and embedded-board performance need
separate tests.
