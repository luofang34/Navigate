"""Local semantic surface masks; labels remain available for non-terrain consumers."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from photo_testset import cases, digest, write_json

# Keep terrain and mapped structures. The excluded labels remain in the semantic map;
# a sky or celestial pipeline can choose another valid-pixel policy independently.
SURFACE_LABELS = {1,4,6,9,11,13,16,17,21,25,26,29,34,46,48,52,54,60,61,68,72,79,84,91,94,128}
OBSTRUCTION_LABELS = {0,2,5,8,12,18,90}


def support_mask(labels, confidence):
    # ADE20K can call a nearby aircraft strut a pole, fence or railing. Require
    # a narrow, elongated edge component; broad runway/fence regions stay usable.
    short=min(labels.shape)
    parts=(np.isin(labels,[32,38,93])&(confidence>=.18)).astype(np.uint8)
    parts=cv2.morphologyEx(parts,cv2.MORPH_CLOSE,np.ones((9,9),np.uint8))
    count,components,stats,_=cv2.connectedComponentsWithStats(parts)
    obstruction=np.zeros_like(labels,np.uint8)
    for index in range(1,count):
        x,y,width,height,area=stats[index]
        edge=min(x,y,labels.shape[1]-x-width,labels.shape[0]-y-height)
        if area<short*short*.002 or edge>short*.02:
            continue
        yy,xx=np.where(components==index)
        center,size,angle=cv2.minAreaRect(np.stack([xx,yy],axis=1).astype(np.float32))
        narrow,long=sorted(size)
        if narrow<=0 or narrow>short*.16 or long/narrow<3 or long<short*.25:
            continue
        # Coarse semantic cells often cover only the centre of a smooth strut.
        expanded=(size[0]+short*.06,size[1]+short*.06)
        box=cv2.boxPoints((center,expanded,angle)).round().astype(np.int32)
        cv2.fillConvexPoly(obstruction,box,255)
    return obstruction


def surface_mask(labels, confidence, margin=6, policy='occlusions', image=None):
    if policy=='surfaces':
        valid=np.isin(labels,list(SURFACE_LABELS))&(confidence>=.35)
    elif policy in {'attached-occlusions','airframe-occlusions','airframe-lines'}:
        apparatus=np.isin(labels,[0,5,8,18,90])&(confidence>=.35)
        apparatus=cv2.morphologyEx(apparatus.astype(np.uint8),cv2.MORPH_CLOSE,np.ones((9,9),np.uint8))
        _,components=cv2.connectedComponents(apparatus)
        border=max(9,round(min(labels.shape)*.02))
        attached=np.unique(np.concatenate([components[:border].ravel(),components[-border:].ravel(),
                                          components[:,:border].ravel(),components[:,-border:].ravel()]))
        attached=attached[attached!=0]
        obstruction=np.isin(components,attached)|(np.isin(labels,[2,12])&(confidence>=.35))
        if policy in {'airframe-occlusions','airframe-lines'}:
            obstruction|=support_mask(labels,confidence)>0
        if policy=='airframe-lines':
            from photo_airframe import border_support_mask
            obstruction|=border_support_mask(image,labels,confidence)>0
        valid=~obstruction
    else:
        # Low semantic confidence on oblique roads does not imply an obstruction.
        valid=~(np.isin(labels,list(OBSTRUCTION_LABELS))&(confidence>=.35))
    valid=valid.astype(np.uint8)*255
    # Small confident islands inside a wing or support strut must not become match seeds.
    valid = 255 - cv2.morphologyEx(255 - valid, cv2.MORPH_CLOSE, np.ones((9,9), np.uint8))
    return cv2.erode(valid, np.ones((2*margin+1, 2*margin+1), np.uint8))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument('--policy',choices=['occlusions','surfaces','attached-occlusions','airframe-occlusions','airframe-lines'],default='occlusions')
    parser.add_argument('--airframe-model',type=Path,help='segment_finetune.py model; only its ground class stays valid')
    args = parser.parse_args()
    import torch
    import transformers
    from PIL import Image
    from transformers import SegformerForSemanticSegmentation
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    cv2.setNumThreads(1)
    processor = json.loads((args.model / "preprocessor_config.json").read_text())
    if not processor.get("do_resize") or not processor.get("do_normalize"):
        raise ValueError("the segmentation input requires resize and normalization")
    model = SegformerForSemanticSegmentation.from_pretrained(
        args.model, local_files_only=True, use_safetensors=True).eval()
    config = json.loads((args.model / "config.json").read_text())
    if config["id2label"].get("2") != "sky" or config["id2label"].get("90") != "airplane":
        raise ValueError("the mask policy requires ADE20K semantic labels")
    root = args.dataset.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    for name in ["semantic", "mask-previews"]:
        (root / name).mkdir(exist_ok=True)
    policy = {"version": 3, "selection":args.policy,"model_sha256": digest(args.model / "model.safetensors"),
              "config_sha256": digest(args.model / "config.json"),
              "processor_sha256": digest(args.model / "preprocessor_config.json"),
              "surface_labels": sorted(SURFACE_LABELS), "confidence_min": .35,
              "obstruction_labels":sorted(OBSTRUCTION_LABELS),
              "margin_pixels": 6, "close_kernel_pixels": 9,
              "torch": torch.__version__, "transformers": transformers.__version__,
              "device": "cpu", "labels": config["id2label"]}
    if args.policy in {'attached-occlusions','airframe-occlusions','airframe-lines'}:
        policy.update({'version':4,'apparatus_border_fraction':.02,'apparatus_border_min_pixels':9})
    if args.policy in {'airframe-occlusions','airframe-lines'}:
        policy.update(version=5,support_labels=[32,38,93],support_confidence_min=.18,
            support_min_area_fraction=.002,support_max_width_fraction=.16,
            support_min_length_fraction=.25,support_min_aspect=3,support_padding_fraction=.06)
    if args.policy=='airframe-lines':
        from photo_airframe import LINE_POLICY
        policy.update(version=6,border_support=LINE_POLICY)
    airframe=None
    if args.airframe_model:
        airframe=SegformerForSemanticSegmentation.from_pretrained(
            args.airframe_model,local_files_only=True,use_safetensors=True).eval()
        if json.loads((args.airframe_model/'config.json').read_text())['id2label'].get('1')!='ground':
            raise ValueError('the airframe model requires segment_finetune.py classes')
        policy.update(airframe_model_sha256=digest(args.airframe_model/'model.safetensors'),
                      airframe_ground_label=1)
    policy_id=hashlib.sha256(json.dumps(policy,sort_keys=True).encode()).hexdigest()[:16]
    for folder in ['semantic','masks','mask-previews']:
        (root/folder/policy_id).mkdir(exist_ok=True)
    for entry in manifest["photos"][:args.limit]:
        status_file = root / "semantic" / policy_id / (entry["id"] + ".json")
        if status_file.exists():
            saved = json.loads(status_file.read_text())
            if saved["policy"] == policy and saved["image_sha256"] == entry["image_sha256"]:
                if digest(root / saved['entry_fields']["valid_mask"]) != saved["mask_sha256"]:
                    raise ValueError("saved mask changed without an explicit override")
                entry.update(saved["entry_fields"])
                entry["semantic_sha256"] = digest(root / entry["semantic_labels"])
                write_json(root / "metadata" / (entry["id"] + ".json"), entry)
                continue
        with Image.open(root / entry["image"]) as image:
            edge = processor["size"]
            target = (edge, edge) if isinstance(edge, int) else (edge["width"], edge["height"])
            pixels = np.asarray(image.convert("RGB").resize(target, processor["resample"]), np.float32) / 255
            pixels = (pixels - np.array(processor["image_mean"], np.float32)) / np.array(processor["image_std"], np.float32)
            inputs = {"pixel_values": torch.from_numpy(pixels.transpose(2,0,1).copy()[None])}
        with torch.inference_mode():
            logits = model(**inputs).logits
            # Resize only the winning class and confidence, avoiding a full-resolution
            # 150-class tensor for every source photograph.
            probabilities = logits.softmax(dim=1)[0]
            confidence, labels = probabilities.max(dim=0)
        size = tuple(entry["size"])
        label_map = cv2.resize(labels.numpy().astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)
        confidence_map = cv2.resize(confidence.numpy(), size, interpolation=cv2.INTER_LINEAR)
        source_image = cv2.imread(str(root / entry['image'])) if args.policy=='airframe-lines' else None
        mask = surface_mask(label_map, confidence_map,policy=args.policy,image=source_image)
        if airframe is not None:
            with torch.inference_mode():
                ground=airframe(**inputs).logits.argmax(dim=1)[0].numpy()==policy['airframe_ground_label']
            ground=cv2.resize(ground.astype(np.uint8)*255,size,interpolation=cv2.INTER_NEAREST)
            # The same margin as the semantic policy keeps match seeds off airframe edges.
            mask=np.minimum(mask,cv2.erode(ground,np.ones((13,13),np.uint8)))
        semantic = f"semantic/{policy_id}/{entry['id']}.png"
        entry['valid_mask']=f"masks/{policy_id}/{entry['id']}.png"
        confidence_path=f"semantic/{policy_id}/{entry['id']}-confidence.npy"
        np.save(root/confidence_path,confidence_map)
        for name, image in [(semantic, label_map), (entry["valid_mask"], mask)]:
            if not cv2.imwrite(str(root / name), image):
                raise OSError(f"cannot write {name}")
        preview = cv2.imread(str(root / entry["image"]))
        excluded = mask == 0
        preview[excluded] = (preview[excluded] * .35 + np.array([40,0,160]) * .65).astype(np.uint8)
        preview = cv2.resize(preview, (480, round(size[1] * 480 / size[0])))
        if not cv2.imwrite(str(root / "mask-previews" / policy_id / (entry["id"] + ".jpg")), preview):
            raise OSError("cannot save mask review image")
        fields = {"mask_status": "automatic_semantic_unreviewed", "semantic_labels": semantic,
                  "semantic_sha256": digest(root / semantic),
                  "valid_mask":entry['valid_mask'],"mask_policy":policy_id,
                  "confidence_map":confidence_path,"confidence_sha256":digest(root/confidence_path),
                  "valid_fraction": float(np.mean(mask > 0)), "mask_sha256": digest(root / entry["valid_mask"])}
        entry.update(fields)
        write_json(status_file, {"policy": policy, "image_sha256": entry["image_sha256"],
                                "mask_sha256": fields["mask_sha256"], "entry_fields": fields})
        write_json(root / "metadata" / (entry["id"] + ".json"), entry)
        print(json.dumps({"id": entry["id"], "valid_fraction": fields["valid_fraction"]}), flush=True)
    write_json(root / "mask-policy.json", policy)
    write_json(root / 'semantic' / policy_id / 'policy.json',policy)
    write_json(root / "manifest.json", manifest)
    for name, prior in [("with-prior", True), ("no-prior", False)]:
        path = root / "profiles" / (name + ".json")
        profile = json.loads(path.read_text())
        profile["cases"] = cases(manifest["photos"], prior)
        write_json(path, profile)


if __name__ == "__main__":
    main()
