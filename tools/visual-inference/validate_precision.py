"""Compare accelerator tensors and decoded keypoints with the FP32 model."""

import argparse, json, sys, time
from pathlib import Path
import numpy as np, onnxruntime as ort, torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model-directory", type=Path, required=True)
parser.add_argument("--models", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parent
sys.path.insert(0, str(root))
from export_models import DensePoint, Glue

sys.path.insert(0, str(args.model_directory.resolve()))
from models.matching import Matching
from models.superpoint import simple_nms, sample_descriptors

models = args.models.resolve()
torch.set_num_threads(4)
torch.set_grad_enabled(False)
inputs = {
    r["name"]: np.fromfile(models / r["path"], dtype="<f4").reshape(r["shape"])
    for r in json.loads((models / "superpoint-inputs.json").read_text())
}
opt = ort.SessionOptions()
opt.intra_op_num_threads = 4
cpu = ort.InferenceSession(
    str(models / "superpoint-dense.onnx"),
    sess_options=opt,
    providers=["CPUExecutionProvider"],
)
ane = ort.InferenceSession(
    str(models / "superpoint-mixed.onnx"),
    sess_options=opt,
    providers=[
        (
            "CoreMLExecutionProvider",
            {"ModelFormat": "MLProgram", "MLComputeUnits": "CPUAndNeuralEngine"},
        ),
        "CPUExecutionProvider",
    ],
)
a = [v.astype(np.float32) for v in cpu.run(None, inputs)]
b = [v.astype(np.float32) for v in ane.run(None, inputs)]


def features(outputs):
    scores = torch.tensor(outputs[0])
    scores = torch.softmax(scores, 1)[:, :-1]
    n, _, h, w = scores.shape
    scores = (
        scores.permute(0, 2, 3, 1)
        .reshape(n, h, w, 8, 8)
        .permute(0, 1, 3, 2, 4)
        .reshape(n, h * 8, w * 8)
    )
    scores = simple_nms(scores, 4)[0]
    points = torch.nonzero(scores > 0.005)
    valid = (
        (points[:, 0] >= 4)
        & (points[:, 1] >= 4)
        & (points[:, 0] < h * 8 - 4)
        & (points[:, 1] < w * 8 - 4)
    )
    points = points[valid]
    values = scores[points[:, 0], points[:, 1]]
    order = torch.argsort(values, descending=True)[:512]
    points = points[order].flip(1).float()
    descriptors = sample_descriptors(points[None], torch.tensor(outputs[1]), 8)[0]
    return points.numpy(), descriptors.numpy()


x, dx = features(a)
y, dy = features(b)
dist = np.linalg.norm(x[:, None, :] - y[None, :, :], axis=2)
nearest = dist.argmin(1)
same = dist.min(1) <= 1
report = dict(
    scope="One real rendered reference, 640x360. Tensor/keypoint consistency, not localization accuracy.",
    fp32_keypoints=len(x),
    fp16_keypoints=len(y),
    keypoints_within_1px=int(same.sum()),
    dense_descriptor_cosine_mean=float(np.sum(a[1] * b[1], axis=1).mean()),
    matched_descriptor_cosine_mean=float(
        np.sum(dx[:, same] * dy[:, nearest[same]], axis=0).mean()
    ),
    score_logit_max_abs_error=float(np.max(np.abs(a[0] - b[0]))),
)
net = (
    Matching(
        dict(
            superpoint=dict(max_keypoints=512, nms_radius=4, keypoint_threshold=0.005),
            superglue=dict(
                weights="outdoor", sinkhorn_iterations=20, match_threshold=0.2
            ),
        )
    )
    .eval()
    .to("mps")
)
dense = DensePoint(net.superpoint)
data = torch.tensor(inputs["image"]).to("mps")
glue_inputs = [
    torch.from_numpy(
        np.fromfile(models / r["path"], dtype="<f4").reshape(r["shape"])
    ).to("mps")
    for r in json.loads((models / "superglue-inputs.json").read_text())
]
# Glue constructs a CPU-only shape sentinel; SuperGlue uses its dimensions only.
glue = Glue(net.superglue, data.shape)
for name, model, values in [
    ("superpoint", dense, [data]),
    ("superglue", glue, glue_inputs),
]:
    times = []
    for i in range(11):
        torch.mps.synchronize()
        start = time.perf_counter()
        out = model(*values)
        torch.mps.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        if i:
            times.append(elapsed)
    report[name + "_pytorch_mps_ms"] = times
report["precision_passed"] = bool(
    report["dense_descriptor_cosine_mean"] > 0.999
    and report["matched_descriptor_cosine_mean"] > 0.999
    and report["keypoints_within_1px"] >= 0.95 * len(x)
)
args.output.write_text(json.dumps(report, indent=2))
if not report["precision_passed"]:
    raise RuntimeError("Mixed-precision feature check failed")
print(json.dumps(report))
