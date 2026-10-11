"""Run prior-guided terrain matching with a resident Navigate LoFTR worker."""
import argparse
import json
import math
from pathlib import Path
import subprocess
from queue import Queue, Empty
from threading import Thread
import time

import cv2
import numpy as np

from photo_pose import fit, ground_pairs,prior_radius
from photo_surface import Surface, horizon_attitude
from photo_testset import digest, write_json
from photo_runs import resolved_inputs


class Matcher:
    def __init__(self, worker, model, runtime):
        self.process=subprocess.Popen([str(worker.resolve()),"match-worker",str(model.resolve()),str(runtime.resolve())],
                                      stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
        self.responses=Queue()
        self.reader=Thread(target=self.read_responses,name="photo-match-responses",daemon=True)
        self.reader.start()

    def read_responses(self):
        try:
            for line in self.process.stdout:
                self.responses.put(line)
        finally:
            self.responses.put(None)

    def match(self, reference, query, refinement=False):
        self.process.stdin.write(json.dumps({"reference":str(reference.resolve()),
                                            "query":str(query.resolve()),"dense":True,
                                            **({'refinement':True} if refinement else {})})+"\n")
        self.process.stdin.flush()
        try:
            line=self.responses.get(timeout=120)
        except Empty as error:
            self.process.kill(); self.process.wait()
            raise TimeoutError("Navigate match exceeded 120 seconds") from error
        if not line:
            raise RuntimeError(f"Navigate matcher exited with {self.process.poll()}")
        return json.loads(line)

    def close(self):
        if self.process.poll() is not None:
            return
        self.process.stdin.close()
        try:
            result=self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill(); self.process.wait()
            raise RuntimeError("Navigate matcher shutdown timed out")
        if result:
            raise RuntimeError(f"Navigate matcher exited with {result}")
        self.reader.join(timeout=1)
        self.process.stdout.close()


def match_pixels(image,mask=None,contrast=False,color=False):
    gray=image.copy() if color else cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    if contrast:
        gray=cv2.createCLAHE(clipLimit=2,tileGridSize=(8,8)).apply(gray)
    if mask is not None:
        gray[mask==0]=0
    return gray


def scaled_camera(K,size,target):
    scale=target/max(size)
    resized=(round(size[0]*scale),round(size[1]*scale))
    camera=K.copy();camera[:2,:2]*=scale
    camera[:2,2]=(camera[:2,2]+.5)*scale-.5
    return camera,resized,scale


def run_case(case, root, output, surface, matcher, attempts, steps, contrast=False,heading_step=0,coarse_edge=0,height_factors=(1.,),refinement=False,pitch_offsets=(0.,),focal_factors=(1.,),roll_offsets=(0.,),position_step_m=0.,position_radius_m=0.):
    from photo_grid import offsets,location
    directory=output/case["id"]; directory.mkdir(parents=True,exist_ok=True)
    if digest(root/case["image"]) != case["image_sha256"]:
        raise ValueError("query hash changed")
    if case.get("mask_sha256") and digest(root/case["valid_mask"]) != case["mask_sha256"]:
        raise ValueError("query mask hash changed")
    prior=case.get("prior")
    if not prior or not prior.get("position"):
        return {"id":case["id"],"status":"unsupported_prior","attempts":[]}
    p=prior["position"]; lat,lon=p["latitude"],p["longitude"]
    terrain,covered=surface.sample("dem",lat,lon)
    if not covered:
        return {"id":case["id"],"status":"outside_reference","attempts":[]}
    image=cv2.imread(str(root/case["image"]))
    h,w=image.shape[:2]; scale=min(640/w,640/h); size=(round(w*scale),round(h*scale))
    image=cv2.resize(image,size,interpolation=cv2.INTER_AREA)
    mask=cv2.resize(cv2.imread(str(root/case["valid_mask"]),0),size,interpolation=cv2.INTER_NEAREST)
    if case["mask_status"]=="unreviewed_all_pixels":
        raise ValueError("semantic mask preprocessing has not run")
    color_input=getattr(matcher,'color',False)
    query=directory/"query.png"; cv2.imwrite(str(query),match_pixels(image,mask,contrast,color_input))
    focal=(prior.get("focal_35mm_mm") or 26)*math.hypot(*size)/math.hypot(36,24)
    K=np.array([[focal,0,(size[0]-1)/2],[0,focal,(size[1]-1)/2],[0,0,1.]])
    label_path=root/case["semantic_labels"]
    if digest(label_path)!=case["semantic_sha256"]:
        raise ValueError("query semantic labels changed")
    labels=cv2.resize(cv2.imread(str(label_path),0),size,interpolation=cv2.INTER_NEAREST)
    horizon=horizon_attitude(labels,K)
    down=horizon["down_deg"] if horizon else 40
    roll=horizon["roll_deg"] if horizon else 0
    altitude=p.get("altitude_m")
    if altitude is None:
        altitude=float(terrain)+1000
    heading=prior.get("direction_deg") if prior.get("direction_reference")=="T" else None
    if heading is None and heading_step:
        heading=0
    if heading is None:
        return {"id":case["id"],"status":"missing_direction","attempts":[]}
    initial_eye=surface.local(lat,lon,altitude)
    seeds=[(0,0,0),(7.5,0,0),(-7.5,0,0),(0,-7,0),(0,7,0),(-15,0,0),(15,0,0),(0,0,-15),(0,0,15),
           (0,20,0),(0,-20,0),(-30,0,0),(30,0,0)]
    if heading_step:
        angles=np.arange(0,360,heading_step)
        angles=sorted(angles,key=lambda a:min(a,360-a))
        seeds=[(float(angle),0,0) for angle in angles]
    records=[]; best=None; started=time.perf_counter()
    initial_K=K.copy()
    seeds=[(dh,dd+offset,dr+turn,factor,ff) for ff in focal_factors for offset in pitch_offsets
           for turn in roll_offsets for factor in height_factors for dh,dd,dr in seeds]
    response={'backend':None}
    locations=offsets(p,position_step_m,position_radius_m)
    hypotheses=[(east,north,seed) for east,north in locations for seed in seeds[:attempts]]
    for number,(east,north,(dh,dd,dr,height_factor,focal_factor)) in enumerate(hypotheses):
        seed_lat,seed_lon=location(lat,lon,east,north)
        seed_ground,seed_covered=surface.sample('dem',seed_lat,seed_lon)
        if not seed_covered:
            continue
        K=initial_K.copy()
        K[0,0]=K[1,1]=focal*focal_factor
        seed_horizon=horizon_attitude(labels,K)
        down=seed_horizon['down_deg'] if seed_horizon else 40
        roll=seed_horizon['roll_deg'] if seed_horizon else 0
        if not -89<=down+dd<=90:
            continue
        eye=surface.local(seed_lat,seed_lon,float(seed_ground)+(altitude-float(seed_ground))*height_factor)
        rot=surface.camera_rotation(seed_lat,seed_lon,heading+dh,down+dd,roll+dr)
        for step in range(steps):
            t=time.perf_counter()
            reference=directory/"reference.png"
            work_K,work_size,work_scale=scaled_camera(K,size,coarse_edge if coarse_edge and step==0 else max(size))
            work_mask=cv2.resize(mask,work_size,interpolation=cv2.INTER_NEAREST)
            work_query=query
            if work_size!=size:
                work_query=directory/'coarse-query.png'
                cv2.imwrite(str(work_query),match_pixels(cv2.resize(image,work_size,interpolation=cv2.INTER_AREA),work_mask,contrast,color_input))
            color,depth,world=surface.render(eye,rot,work_K,work_size)
            cv2.imwrite(str(reference),match_pixels(color,(depth>0).astype(np.uint8),contrast,color_input))
            response=matcher.match(reference,work_query,refinement=refinement and step>=2)
            points,pixels=ground_pairs(response["pairs"],depth,world,work_mask)
            seed_focal=focal*focal_factor
            result=fit(points,pixels,work_K,work_size,(seed_focal*.75*work_scale,seed_focal*1.25*work_scale) if step>=2 else None)
            record={"seed":number,"step":step,"raw_pairs":len(response["pairs"]),
                    "position_offset_m":[east,north],
                    "ground_pairs":len(points),"resolution":work_size,"height_factor":height_factor,
                    "focal_factor":focal_factor,"pitch_deg":down+dd,"roll_deg":roll+dr,
                    "heading_deg":heading+dh,"seconds":time.perf_counter()-t}
            if result is None:
                records.append({**record,"status":"insufficient_geometry"})
                print(json.dumps({"case":case["id"],**records[-1]}),flush=True)
                break
            proposal,orientation,score,inliers=result
            np.savez_compressed(directory/f"evidence-{number:02}-{step:02}.npz",
                points=points,pixels=pixels,input_K=work_K,focal_px=score["focal_px"],
                eye=proposal,rotation=orientation,inliers=inliers)
            delta=float(np.linalg.norm(proposal-initial_eye))
            plat,plon,palt=surface.geodetic(proposal)
            ground,available=surface.sample("dem",plat,plon)
            radius=prior_radius(p)
            plausible=bool(available and 50<float(palt-ground)<6000 and delta<radius)
            record.update({**score,"prior_delta_m":delta,"prior_gate":plausible})
            records.append(record)
            print(json.dumps({"case":case["id"],**record}),flush=True)
            if not plausible or score["inliers"]<25 or score["image_support_fraction"]<.03:
                break
            eye,rot=proposal,orientation
            K[0,0]=K[1,1]=score["focal_px"]/work_scale
            merit=score["inliers"]*score["image_support_fraction"]/(1+score["holdout_median_px"])
            # A plausible unverified proposal can guide the next render, but only a
            # held-out geometric pass may become a reviewable candidate.
            if work_size==size and score["geometry_accepted"] and (best is None or merit>best[0]):
                state={"eye":eye,"rotation":rot,"K":K,"size":np.array(size),
                       "points":points,"pixels":pixels,"inliers":inliers}
                np.savez_compressed(directory/"candidate.npz",**state)
                best=(merit,{"latitude":float(plat),"longitude":float(plon),"altitude_m":float(palt),
                             "score":score,"seed":number,"step":step})
            if step==steps-1:
                break
        if best is not None:
            break
    if best:
        state=np.load(directory/"candidate.npz")
        color,_,_=surface.render(state["eye"],state["rotation"],state["K"],size)
        cv2.imwrite(str(directory/"candidate-reference.png"),color)
        cv2.imwrite(str(directory/"comparison.jpg"),np.concatenate([image,color,cv2.addWeighted(image,.5,color,.5,0)],axis=1))
    return {"id":case["id"],"status":"candidate_requires_review" if best else "no_candidate",
            "candidate":best[1] if best else None,"horizon":horizon,"attempts":records,
            "seconds":time.perf_counter()-started,"matcher":response["backend"]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile",type=Path); parser.add_argument("output",type=Path)
    parser.add_argument("--catalog",type=Path,required=True)
    parser.add_argument("--worker",type=Path,default=Path(__file__).resolve().parents[2] / "target/release/navigate-recording-worker")
    parser.add_argument("--model",type=Path); parser.add_argument("--runtime",type=Path)
    parser.add_argument('--backend',choices=['loftr','roma'],default='loftr')
    parser.add_argument('--roma-model',type=Path);parser.add_argument('--roma-encoder',type=Path)
    parser.add_argument('--device',choices=['cpu','mps'],default='cpu')
    parser.add_argument('--grayscale',action='store_true',help='remove seasonal colour differences before RoMa matching')
    parser.add_argument("--case",action="append"); parser.add_argument("--attempts",type=int,default=5)
    parser.add_argument('--skip-run',type=Path,action='append',default=[],help='skip geometrically accepted cases from a completed pass')
    parser.add_argument("--steps",type=int,default=3)
    parser.add_argument("--shard",type=int,nargs=2,metavar=("INDEX","COUNT"),default=[0,1])
    parser.add_argument("--contrast",action="store_true",help="apply local contrast normalization before matching")
    parser.add_argument("--refinement",action="store_true",help="use Navigate overlapping patches from the third iteration")
    parser.add_argument("--heading-step",type=float,default=0,help="search a full circle at this degree interval")
    parser.add_argument("--coarse-edge",type=int,default=0,help="first-step resolution; final acceptance still uses 640 pixels")
    parser.add_argument("--height-factors",type=float,nargs='+',default=[1.],help="AGL hypotheses for uncertain EXIF altitude")
    parser.add_argument("--pitch-offsets",type=float,nargs='+',default=[0.],help="pitch hypotheses relative to the semantic horizon seed")
    parser.add_argument("--roll-offsets",type=float,nargs='+',default=[0.],help="roll hypotheses when the horizon is missing or uncertain")
    parser.add_argument("--focal-factors",type=float,nargs='+',default=[1.],help="focal hypotheses relative to the claimed or default focal length")
    parser.add_argument('--position-step-m',type=float,default=0.,help='grid spacing for uncertain camera positions; zero disables the grid')
    parser.add_argument('--position-radius-m',type=float,default=0.,help='grid radius; zero uses the reported position uncertainty')
    args=parser.parse_args()
    if args.attempts<1 or args.steps<1:
        parser.error("attempts and steps must be positive")
    if not np.isfinite([args.position_step_m,args.position_radius_m]).all() or min(args.position_step_m,args.position_radius_m)<0 or (args.position_radius_m and not args.position_step_m):
        parser.error('position grid requires finite nonnegative distances and a positive step when a radius is supplied')
    if not 0<=args.shard[0]<args.shard[1]:
        parser.error("shard requires 0 <= INDEX < COUNT")
    if (args.heading_step and not 5<=args.heading_step<=180) or (args.coarse_edge and not 128<=args.coarse_edge<=640):
        parser.error('invalid heading step or coarse resolution')
    if any(not .25<=factor<=8 for factor in args.height_factors):
        parser.error('height factors must be within .25 to 8')
    if any(not -60<=offset<=60 for offset in args.pitch_offsets):
        parser.error('pitch offsets must be within -60 to 60 degrees')
    if any(not -90<=offset<=90 for offset in args.roll_offsets):
        parser.error('roll offsets must be within -90 to 90 degrees')
    if any(not .125<=factor<=8 for factor in args.focal_factors):
        parser.error('focal factors must be within .125 to 8')
    if args.backend=='roma':
        if args.roma_model is None or args.roma_encoder is None:
            parser.error('RoMa requires explicit local model and encoder weights')
        if args.contrast or args.refinement:
            parser.error('contrast normalization and overlapping patches are only supported by LoFTR')
        from photo_roma import configuration as roma_configuration,RomaMatcher
        backend=roma_configuration(args.roma_model,args.roma_encoder,args.device,args.grayscale)
    else:
        if args.grayscale:
            parser.error('grayscale selection is for RoMa; LoFTR already uses grayscale')
        if args.model is None or args.runtime is None:
            parser.error('LoFTR requires its local model and ONNX runtime')
        backend={'backend':'loftr','model_sha256':digest(args.model),
                 'worker_sha256':digest(args.worker),'runtime_sha256':digest(args.runtime)}
    cv2.setNumThreads(4)
    profile=json.loads(args.profile.read_text())
    if profile["profile"]!="with-prior":
        parser.error("this runner requires explicit priors; blind retrieval must use a separate acquisition stage")
    root=(args.profile.parent/profile["root"]).resolve(); output=args.output.resolve(); output.mkdir(parents=True,exist_ok=True)
    resolved,skip_inputs=resolved_inputs(args.skip_run)
    configuration={"profile_sha256":digest(args.profile),"catalog_sha256":digest(args.catalog),
                   **backend,"opencv":cv2.__version__,"numpy":np.__version__,
                   "attempts":args.attempts,"steps":args.steps,"shard":args.shard,"contrast":args.contrast,
                   "heading_step":args.heading_step,"coarse_edge":args.coarse_edge,
                   "height_factors":args.height_factors,"refinement":args.refinement,"pitch_offsets":args.pitch_offsets,
                   "focal_factors":args.focal_factors,"roll_offsets":args.roll_offsets,
                   "position_step_m":args.position_step_m,"position_radius_m":args.position_radius_m,
                   "coordinate_frame":{"kind":"spherical_enu","radius_m":6371008.8},
                   "code":{p.name:digest(p) for p in Path(__file__).parent.glob("photo_*.py")}}
    if skip_inputs:
        configuration['skip_inputs']=skip_inputs
    if (output/"run.json").exists() and json.loads((output/"run.json").read_text())!=configuration:
        raise ValueError("run inputs or solver changed; use a new output directory")
    write_json(output/"run.json",configuration)
    surface=Surface(args.catalog)
    matcher=(RomaMatcher(args.roma_model,args.roma_encoder,args.device,args.grayscale) if args.backend=='roma'
             else Matcher(args.worker,args.model,args.runtime))
    try:
        for index,case in enumerate(profile["cases"]):
            if index%args.shard[1]!=args.shard[0]:
                continue
            if args.case and case["id"] not in args.case:
                continue
            if case['id'] in resolved:
                continue
            if (output/(case["id"]+".json")).exists():
                continue
            print(json.dumps({"case":case["id"],"status":"started"}),flush=True)
            result=run_case(case,root,output,surface,matcher,args.attempts,args.steps,args.contrast,args.heading_step,
                            args.coarse_edge,args.height_factors,args.refinement,args.pitch_offsets,args.focal_factors,args.roll_offsets,
                            args.position_step_m,args.position_radius_m)
            write_json(output/(case["id"]+".json"),result)
            print(json.dumps({"case":case["id"],"status":result["status"]}),flush=True)
    finally:
        matcher.close()


if __name__ == "__main__":
    main()
