"""Masked visual retrieval from an explicitly supplied, georeferenced photo bank."""
import hashlib
import json
from pathlib import Path
import tempfile

import cv2
import numpy as np

from photo_testset import digest, write_json
from photo_tracking import prepared


def read_profile(path, blind=False):
    profile = json.loads(path.read_text())
    allowed = {'id', 'image', 'size', 'valid_mask', 'mask_status', 'image_sha256',
               'mask_sha256', 'semantic_labels', 'semantic_sha256'}
    if blind and (profile['profile'] != 'no-prior'
                  or any(set(case) - allowed for case in profile['cases'])):
        raise ValueError('acquisition requires a metadata-free no-prior query profile')
    cases = {case['id']: case for case in profile['cases']}
    if len(cases) != len(profile['cases']):
        raise ValueError('profile contains duplicate photo IDs')
    for identifier in cases:
        if Path(identifier).name != identifier or identifier in {'.', '..'}:
            raise ValueError('photo IDs must be directory names')
    return cases, (path.parent / profile['root']).resolve()


def flight_partition(path):
    if path is None:
        return None
    flights = {}
    for flight in json.loads(path.read_text())['flights']:
        for identifier in flight['photos']:
            if identifier in flights:
                raise ValueError('a photo belongs to multiple excluded-flight groups')
            flights[identifier] = flight['id']
    return flights


def rank_references(case, query, references, features, flights=None, anchors=None):
    """Flight labels only exclude references; neither position nor time ranks them."""
    chains = {q: [q, *[item['id'] for item in (anchors or {}).get(q, {}).get('lineage', [])]]
              for q in features}
    if flights is not None and any(q not in flights for q in [case['id'],
                                  *[item for chain in chains.values() for item in chain]]):
        raise ValueError('flight exclusion requires a group for every query and reference')
    ranked = []
    for identifier, descriptors in features.items():
        chain = chains[identifier]
        if any(q == case['id'] or references.get(q, {}).get('image_sha256') == case['image_sha256']
               for q in chain):
            continue
        if flights is not None and any(flights[q] == flights[case['id']] for q in chain):
            continue
        if not len(query) or not len(descriptors):
            continue
        ranked.append({'anchor': identifier, 'similarity': float((query @ descriptors.T).max())})
    return sorted(ranked, key=lambda item: (-item['similarity'], item['anchor']))


def pooled_descriptors(patches, valid):
    height, width = valid.shape
    regions = [(0, 0, width, height)] + [
        (x, y, x + width // 2, y + height // 2)
        for y in [0, height // 2] for x in [0, width // 2]]
    descriptors = []
    for left, top, right, bottom in regions:
        values = patches[top:bottom, left:right][valid[top:bottom, left:right]]
        if len(values) >= 10:
            mean = values.mean(0)
            descriptors.append(mean / max(np.linalg.norm(mean), 1e-12))
    return np.asarray(descriptors, np.float32).reshape(-1, patches.shape[-1])


class DinoFeatures:
    def __init__(self, model, device):
        import torch
        from romatch.models.transformer import vit_large
        from photo_roma import device_lock
        self.torch = torch
        self.device = device
        torch.set_num_threads(4)
        self.lease = (device_lock(Path(tempfile.gettempdir()) / 'sokoly-photo-mps.lock')
                      if device == 'mps' else None)
        try:
            self.model = vit_large(img_size=518, patch_size=14, init_values=1.,
                                   ffn_layer='mlp', block_chunks=0).eval()
            self.model.load_state_dict(torch.load(model, map_location='cpu', weights_only=True))
            self.model = self.model.to(device)
        except BaseException:
            if self.lease is not None:
                self.lease.close()
            raise

    def extract(self, image, mask):
        torch = self.torch
        image = cv2.resize(image, (560, 420))
        mask = cv2.resize(mask, (560, 420), interpolation=cv2.INTER_NEAREST)
        image[mask == 0] = 0
        tensor = torch.from_numpy(image[:, :, ::-1].copy()).permute(2, 0, 1).float() / 255
        tensor = ((tensor - torch.tensor([.485, .456, .406])[:, None, None])
                  / torch.tensor([.229, .224, .225])[:, None, None])
        with torch.inference_mode():
            output = self.model.forward_features(tensor[None].to(self.device))
        patches = output['x_norm_patchtokens'][0].float().cpu().numpy().reshape(30, 40, 1024)
        valid = cv2.resize(mask, (40, 30), interpolation=cv2.INTER_AREA) > .8 * 255
        return pooled_descriptors(patches, valid)

    def close(self):
        try:
            del self.model
            if self.device == 'mps':
                self.torch.mps.empty_cache()
        finally:
            if self.lease is not None:
                self.lease.close()


def cached_features(case, root, extractor, cache, configuration):
    if case.get('mask_status') == 'unreviewed_all_pixels':
        raise ValueError('photo-bank acquisition requires preprocessed obstruction masks')
    image, mask, _ = prepared(case, root)
    key = {'image': case['image_sha256'], 'mask': case['mask_sha256'], **configuration}
    name = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()
    path, metadata = cache / (name + '.npz'), cache / (name + '.json')
    if metadata.exists():
        saved = json.loads(metadata.read_text())
        if saved['key'] != key or not path.exists() or digest(path) != saved['sha256']:
            raise ValueError(f'photo feature cache changed: {path}')
        with np.load(path) as data:
            return data['descriptors'].copy()
    descriptors = extractor.extract(image, mask)
    np.savez_compressed(path, descriptors=descriptors)
    write_json(metadata, {'key': key, 'sha256': digest(path)})
    return descriptors
