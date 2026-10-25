"""Optional local dense photo matching with explicit RoMa and encoder weights."""
import importlib.metadata
from pathlib import Path
import tempfile

import cv2
import numpy as np

from photo_testset import digest


def configuration(model,encoder,device,grayscale=False):
    return {'backend':'roma-outdoor','model_sha256':digest(model),'encoder_sha256':digest(encoder),
            'device':device,'romatch':importlib.metadata.version('romatch'),
            'torch':importlib.metadata.version('torch'),'coarse_resolution':560,
            'fine_resolution':864,'sample_mode':'balanced','sample_count':5000,
            'sample_seed':0,'minimum_confidence':.1,'grayscale':grayscale}


def match_input(path,grayscale):
    if not grayscale:
        return str(path)
    from PIL import Image
    pixels=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
    if pixels is None:
        raise ValueError(f'cannot decode grayscale matching input: {path}')
    return Image.fromarray(pixels).convert('RGB')


def pixel_pairs(normalized,reference_size,query_size):
    # Normalized tensor coordinates refer to pixel centres; OpenCV centres start at zero.
    scale=np.array([*reference_size,*query_size],np.float64)
    return (((np.asarray(normalized)+1)*scale-1)/2).reshape(-1,2,2)


def device_lock(path):
    # Metal allocations are shared system memory; concurrent models can force
    # the image and terrain working sets into swap. The kernel releases this
    # lease even when a matching process is interrupted.
    import fcntl
    lease=path.open('a+b')
    try:
        fcntl.flock(lease,fcntl.LOCK_EX)
    except BaseException:
        lease.close();raise
    return lease


class RomaMatcher:
    variable_sizes=True
    color=True
    patch_refinement=False

    def __init__(self,model,encoder,device,grayscale=False):
        import torch
        from romatch import roma_outdoor
        self.torch=torch;self.device=device;self.config=configuration(model,encoder,device,grayscale)
        if device=='mps' and not torch.backends.mps.is_available():
            raise RuntimeError('Metal is unavailable; select the CPU device or enable local GPU access')
        torch.set_num_threads(4)
        self.lease=device_lock(Path(tempfile.gettempdir())/'sokoly-photo-mps.lock') if device=='mps' else None
        try:
            self.model=roma_outdoor(device=device,
                weights=torch.load(model,map_location='cpu',weights_only=True),
                dinov2_weights=torch.load(encoder,map_location='cpu',weights_only=True),
                coarse_res=560,upsample_res=864,use_custom_corr=False).eval()
        except BaseException:
            if self.lease is not None:
                self.lease.close()
            raise
        self.model.sample_mode='balanced'

    def match(self,reference,query,refinement=False):
        if refinement:
            raise ValueError('RoMa does not implement the Navigate patch-refinement profile')
        with self.torch.inference_mode():
            warp,certainty=self.model.match(match_input(reference,self.config['grayscale']),
                match_input(query,self.config['grayscale']),device=self.device)
            self.torch.manual_seed(self.config['sample_seed'])
            matches,confidence=self.model.sample(warp,certainty,num=self.config['sample_count'])
        selected=confidence.cpu().numpy()>=self.config['minimum_confidence']
        a=cv2.imread(str(reference),0);b=cv2.imread(str(query),0)
        pairs=pixel_pairs(matches.cpu().numpy()[selected],a.shape[::-1],b.shape[::-1])
        identity='/'.join(['roma-outdoor',self.config['model_sha256'],self.config['encoder_sha256'],
                           self.device,'560-864-balanced-5000-seed0-confidence0.1'])
        if self.config['grayscale']:
            identity+='/opencv-grayscale'
        return {'backend':identity,'pairs':pairs.tolist()}

    def close(self):
        try:
            del self.model
            if self.device=='mps':
                self.torch.mps.empty_cache()
        finally:
            if self.lease is not None:
                self.lease.close()
