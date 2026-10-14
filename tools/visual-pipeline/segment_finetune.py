"""Fine-tune a SegFormer sky model to sky, ground, airframe, and propeller classes.

Labels are pseudo-labels: video sky and propeller masks from video_masks.py,
photo sky from the photo test set, and airframe masks from airframe_labels.py.
Scores measure agreement with these labels, not ground truth.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time

import cv2
import numpy as np

from airframe_labels import decode_video

CLASSES = ['sky', 'ground', 'airframe', 'propeller']
SKY, GROUND, AIRFRAME, PROPELLER = range(4)
ADE_SKY_LABEL = 2
CONFIDENCE_MIN = .35
TRAIN_POLICY = {'size': 512, 'batch': 8, 'crop_scale': (.6, 1.), 'crop_aspect': (.75, 1.8),
                'hue_shift': 10, 'saturation': (.5, 1.5), 'value': (.6, 1.4), 'random_gray': .3,
                'photo_weight': 2., 'encoder_lr': 6e-5, 'decoder_lr': 6e-4, 'minimum_label_fraction': .001}
GRAY_WEIGHTS = np.array([.299, .587, .114], np.float32)


def drop_specks(mask, minimum_fraction=TRAIN_POLICY['minimum_label_fraction']):
    """Remove small prompt leaks; real airframe parts are much larger."""
    count, components, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = np.zeros(count, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= mask.size * minimum_fraction
    return keep[components]


def compose_labels(sky, airframe, propeller):
    """Later classes win: an airframe edge seen against the sky is airframe."""
    labels = np.full(sky.shape, GROUND, np.uint8)
    labels[sky] = SKY
    labels[airframe] = AIRFRAME
    labels[propeller] = PROPELLER
    return labels


def normalize(rgb, mean, std, color):
    """Return model input in NHWC float32; `gray` replicates luma so the pretrained encoder keeps three channels."""
    x = rgb.astype(np.float32) / 255
    if color == 'gray':
        x = np.repeat((x @ GRAY_WEIGHTS)[..., None], 3, axis=-1)
    return (x - mean) / std


class Score:
    """Per-class IoU plus exclusion rates. Excluded means not predicted as ground."""

    def __init__(self):
        self.intersection = np.zeros(len(CLASSES))
        self.union = np.zeros(len(CLASSES))
        self.count = dict.fromkeys(['airframe', 'airframe_excluded', 'ground', 'ground_excluded'], 0)

    def add(self, excluded, labels, prediction=None):
        if prediction is not None:
            for c in range(len(CLASSES)):
                p, t = prediction == c, labels == c
                self.intersection[c] += (p & t).sum()
                self.union[c] += (p | t).sum()
        airframe, ground = labels >= AIRFRAME, labels == GROUND
        self.count['airframe'] += airframe.sum()
        self.count['airframe_excluded'] += (airframe & excluded).sum()
        self.count['ground'] += ground.sum()
        self.count['ground_excluded'] += (ground & excluded).sum()

    def result(self, with_iou=True):
        n = self.count
        result = {'airframe_excluded': round(float(n['airframe_excluded'] / max(n['airframe'], 1)), 3),
                  'ground_wrongly_excluded': round(float(n['ground_excluded'] / max(n['ground'], 1)), 3)}
        if with_iou:
            result['iou'] = {name: round(float(self.intersection[c] / self.union[c]), 3) if self.union[c] else None
                             for c, name in enumerate(CLASSES)}
        return result


def load_video(part):
    frames = decode_video(Path(part['video']), part.get('width', 960))
    airframe = np.load(part['airframe'])['masks']
    if len(airframe) != len(frames):
        raise ValueError(f'{part["airframe"]}: {len(airframe)} masks for {len(frames)} frames')
    masks = Path(part['masks'])
    labels = [compose_labels(cv2.imread(str(masks / f'{i:06d}-sky.png'), 0) > 0, airframe[i],
                             cv2.imread(str(masks / f'{i:06d}-propeller.png'), 0) > 0)
              for i in range(len(frames))]
    return list(frames), labels, [None] * len(labels)


def load_photos(part, held_out):
    """Photo propeller blur is not labelled; it stays in the class its pixels already have."""
    root = Path(part['testset'])
    manifest = json.loads((root / 'manifest.json').read_text())
    flight = {pid: f['id'] for f in manifest['flights'] for pid in f['photos']}
    test_flights = set(part['held_out_flights'])
    images, labels, valid = [], [], []
    for photo in manifest['photos']:
        if (flight[photo['id']] in test_flights) != held_out:
            continue
        semantic = cv2.imread(str(root / photo['semantic_labels']), 0)
        sky = (semantic == ADE_SKY_LABEL) & (np.load(root / photo['confidence_map']) >= CONFIDENCE_MIN)
        airframe = drop_specks(np.load(Path(part['airframe']) / f'{photo["id"]}.npz')['mask'])
        images.append(cv2.imread(str(root / photo['image'])))
        labels.append(compose_labels(sky, airframe, np.zeros_like(sky)))
        valid.append(cv2.imread(str(root / photo['valid_mask']), 0) > 0)
    return images, labels, valid


def split_name(name):
    """A photo part takes a `-train` or `-test` suffix for the flight split."""
    return name.rpartition('-')[::2] if name.endswith(('-train', '-test')) else (name, '')


def load(spec, name):
    base, split = split_name(name)
    part = spec['parts'][base]
    if part['kind'] == 'video':
        return load_video(part)
    if not split:
        return tuple(a + b for a, b in zip(load_photos(part, False), load_photos(part, True)))
    return load_photos(part, held_out=split == 'test')


def augment(image, labels, rng, color):
    p = TRAIN_POLICY
    h, w = labels.shape
    scale = rng.uniform(*p['crop_scale'])
    ch = int(h * scale)
    cw = int(min(w, h * scale * rng.uniform(*p['crop_aspect'])))
    y0, x0 = rng.integers(h - ch + 1), rng.integers(w - cw + 1)
    size = (p['size'], p['size'])
    image = cv2.resize(image[y0:y0 + ch, x0:x0 + cw], size, interpolation=cv2.INTER_AREA)
    labels = cv2.resize(labels[y0:y0 + ch, x0:x0 + cw], size, interpolation=cv2.INTER_NEAREST)
    if rng.random() < .5:
        image, labels = image[:, ::-1], labels[:, ::-1]
    hsv = cv2.cvtColor(np.ascontiguousarray(image), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-p['hue_shift'], p['hue_shift'])) % 180
    hsv[..., 1] *= rng.uniform(*p['saturation'])
    hsv[..., 2] *= rng.uniform(*p['value'])
    image = cv2.cvtColor(hsv.clip(0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    # Random gray teaches shape cues and keeps colour cues such as a red cowl.
    if color == 'randgray' and rng.random() < p['random_gray']:
        image = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), labels


class Model:
    def __init__(self, path, device, num_labels=None):
        import torch
        from transformers import SegformerForSemanticSegmentation
        self.torch, self.device = torch, device
        processor = json.loads((path / 'preprocessor_config.json').read_text())
        self.mean = np.array(processor['image_mean'], np.float32)
        self.std = np.array(processor['image_std'], np.float32)
        extra = {} if num_labels is None else {
            'num_labels': num_labels, 'ignore_mismatched_sizes': True,
            'id2label': dict(enumerate(CLASSES)), 'label2id': {c: i for i, c in enumerate(CLASSES)}}
        self.net = SegformerForSemanticSegmentation.from_pretrained(
            path, local_files_only=True, use_safetensors=True, **extra).to(device)
        # MPS BatchNorm backward rejects the non-contiguous fused decoder features.
        self.net.decode_head.batch_norm.register_forward_pre_hook(lambda _m, a: (a[0].contiguous(),))

    def tensor(self, rgb, color):
        x = normalize(rgb, self.mean, self.std, color)
        return self.torch.from_numpy(x.transpose(0, 3, 1, 2).copy()).to(self.device)

    def logits(self, rgb, color, size):
        logits = self.net(pixel_values=self.tensor(rgb, color)).logits
        return self.torch.nn.functional.interpolate(logits, size=size, mode='bilinear', align_corners=False)

    def predict(self, image, color):
        edge = TRAIN_POLICY['size']
        rgb = cv2.cvtColor(cv2.resize(image, (edge, edge), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        with self.torch.inference_mode():
            return self.logits(rgb[None], color, image.shape[:2]).argmax(1)[0].cpu().numpy().astype(np.uint8)


def train(model, groups, weights, color, steps, seed):
    torch, p = model.torch, TRAIN_POLICY
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    counts = sum(np.bincount(np.concatenate([l.ravel() for l in labels]), minlength=len(CLASSES)) * w
                 for (_, labels), w in zip(groups, weights))
    frequency = counts / counts.sum()
    # Inverse square-root frequency keeps thin struts and blades from vanishing without overweighting them.
    class_weight = np.clip(1 / np.sqrt(np.maximum(frequency, 1e-4)), 0, 20)
    class_weight = torch.tensor(class_weight / class_weight.mean(), dtype=torch.float32, device=model.device)
    optimizer = torch.optim.AdamW([
        {'params': model.net.segformer.parameters(), 'lr': p['encoder_lr']},
        {'params': model.net.decode_head.parameters(), 'lr': p['decoder_lr']}], weight_decay=.01)
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[p['encoder_lr'], p['decoder_lr']], total_steps=steps, pct_start=.1)
    model.net.train()
    for step in range(steps):
        batch = []
        for _ in range(p['batch']):
            images, labels = groups[rng.choice(len(groups), p=weights)]
            i = rng.integers(len(images))
            batch.append(augment(images[i], labels[i], rng, color))
        x = np.stack([b[0] for b in batch])
        y = torch.from_numpy(np.stack([b[1] for b in batch]).astype(np.int64)).to(model.device)
        loss = torch.nn.functional.cross_entropy(model.logits(x, color, y.shape[1:]), y, weight=class_weight)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        schedule.step()
        if step % 500 == 0 or step == steps - 1:
            print(json.dumps({'step': step, 'loss': round(loss.item(), 4)}), flush=True)
    model.net.eval()
    return frequency


def evaluate(model, images, labels, valid, color):
    score, baseline = Score(), Score()
    for image, label, mask in zip(images, labels, valid):
        prediction = model.predict(image, color)
        score.add(prediction != GROUND, label, prediction)
        if mask is not None:
            baseline.add(~mask, label)
    result = {**score.result(), 'images': len(images)}
    if valid and valid[0] is not None:
        result['current_policy'] = baseline.result(with_iou=False)
    return result


def provenance(spec_path, spec, base_model):
    """Tie a model to the training code commit and to every input file by digest."""
    from photo_testset import digest
    here = Path(__file__).resolve().parent

    def git(*args):
        result = subprocess.run(['git', '-C', str(here), *args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    status = git('status', '--porcelain', '--', '.')
    labels = {}
    for name, part in sorted(spec['parts'].items()):
        if part['kind'] == 'video':
            labels[name] = {'video': digest(Path(part['video'])), 'airframe': digest(Path(part['airframe']))}
        else:
            labels[name] = {path.stem: digest(path) for path in sorted(Path(part['airframe']).glob('*.npz'))}
    # A workspace merge commit is not durable; the last commit of this script is.
    return {'code_commit': git('rev-list', '-1', 'HEAD', '--', Path(__file__).name),
            'code_dirty': None if status is None else bool(status),
            'script_sha256': digest(Path(__file__).resolve()), 'spec_sha256': digest(spec_path),
            'base_model_sha256': digest(base_model / 'model.safetensors'), 'label_sha256': labels}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('spec', type=Path, help='JSON with dataset parts; see the README')
    parser.add_argument('output', type=Path)
    parser.add_argument('--base-model', type=Path, required=True)
    parser.add_argument('--train', required=True, help='comma list of parts, for example v5730,photos-train')
    parser.add_argument('--test', default='', help='comma list of parts to score after training')
    parser.add_argument('--color', choices=['color', 'randgray', 'gray'], default='randgray')
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cpu')
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    inputs = provenance(args.spec, spec, args.base_model)
    train_parts = args.train.split(',')
    groups = [load(spec, name)[:2] for name in train_parts]
    weights = np.array([TRAIN_POLICY['photo_weight'] if spec['parts'][split_name(n)[0]]['kind'] == 'photos'
                        else 1. for n in train_parts])
    model = Model(args.base_model, args.device, num_labels=len(CLASSES))
    start = time.perf_counter()
    frequency = train(model, groups, weights / weights.sum(), args.color, args.steps, args.seed)
    del groups
    result = {'train': args.train, 'color': args.color, 'steps': args.steps, 'seed': args.seed,
              'train_s': round(time.perf_counter() - start), 'policy': TRAIN_POLICY, 'provenance': inputs,
              'train_class_fraction': [round(float(f), 3) for f in frequency]}
    for name in filter(None, args.test.split(',')):
        result[name] = evaluate(model, *load(spec, name), args.color)
    model.net.save_pretrained(args.output, safe_serialization=True)
    shutil.copy(args.base_model / 'preprocessor_config.json', args.output)
    (args.output / 'training.json').write_text(json.dumps(result, indent=1) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
