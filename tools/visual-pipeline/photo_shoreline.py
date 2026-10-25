"""Propose camera seeds from water outlines; independent terrain verification is required."""
import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

from photo_pose import fit,prior_radius
from photo_runs import resolved_inputs
from photo_surface import Surface,horizon_attitude
from photo_testset import digest,write_json
from photo_tracking import prepared


def largest_water(mask):
    count,labels,stats,_=cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    eligible=[i for i in range(1,count) if stats[i,4]>=500
              and max(stats[i,2:4])/max(1,min(stats[i,2:4]))<15]
    return np.isin(labels,[max(eligible,key=lambda i:stats[i,4])]) if eligible else np.zeros_like(mask,bool)


def photo_water(image,labels,valid):
    seed=largest_water(np.isin(labels,[21,26,60,128])&(valid>0))
    y,x=np.where(seed)
    if not len(x):
        return seed
    h,w=seed.shape;dx=round(w*.27);dy=round(h/12)
    region=np.zeros(seed.shape,np.uint8)
    region[max(0,y.min()-dy):min(h,y.max()+dy+1),max(0,x.min()-dx):min(w,x.max()+dx+1)]=cv2.GC_PR_BGD
    region[seed]=cv2.GC_PR_FGD
    core=cv2.erode(seed.astype(np.uint8),np.ones((7,7),np.uint8))>0
    region[core]=cv2.GC_FGD
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    # Specular water can lose its water label while remaining next to the water seed.
    region[(hsv[:,:,2]>210)&(hsv[:,:,1]<50)&(region==cv2.GC_PR_BGD)]=cv2.GC_FGD
    region[valid==0]=cv2.GC_BGD
    if not np.any(region==cv2.GC_FGD) or not np.any(region==cv2.GC_BGD):
        return seed
    cv2.setRNGSeed(0)
    cv2.grabCut(image,region,None,np.zeros((1,65)),np.zeros((1,65)),5,cv2.GC_INIT_WITH_MASK)
    return largest_water(np.isin(region,[cv2.GC_FGD,cv2.GC_PR_FGD]))


def reference_water(image,depth):
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    seed=((hsv[:,:,0]>65)&(hsv[:,:,2]<65)&(depth>0)).astype(np.uint8)
    return largest_water(cv2.morphologyEx(seed,cv2.MORPH_OPEN,np.ones((5,5),np.uint8)))


def outline_pairs(query,reference,world,depth):
    if min(query.sum(),reference.sum())<500:
        return None
    smooth=lambda mask:cv2.GaussianBlur(mask.astype(np.float32),(11,11),3)
    warp=np.eye(2,3,dtype=np.float32)
    try:
        correlation,warp=cv2.findTransformECC(smooth(query),smooth(reference),warp,cv2.MOTION_AFFINE,
            (cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,150,1e-6),None,5)
    except cv2.error:
        return None
    singular=np.linalg.svd(warp[:,:2],compute_uv=False)
    if singular.min()<.5 or singular.max()>2 or np.linalg.det(warp[:,:2])<=0:
        return None
    aligned=cv2.warpAffine(reference.astype(np.uint8),warp,query.shape[::-1],
        flags=cv2.INTER_NEAREST|cv2.WARP_INVERSE_MAP)>0
    overlap=float(np.sum(aligned&query)/max(1,np.sum(aligned|query)))
    if overlap<.55:
        return None
    contours,_=cv2.findContours(reference.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    xy=max(contours,key=cv2.contourArea).reshape(-1,2)[::2]
    h,w=depth.shape;inside=(xy[:,0]>=5)&(xy[:,0]<w-5)&(xy[:,1]>=5)&(xy[:,1]<h-5)
    xy=xy[inside];xy=xy[depth[xy[:,1],xy[:,0]]>0]
    inverse=cv2.invertAffineTransform(warp)
    pixels=xy@inverse[:,:2].T+inverse[:,2]
    inside=(pixels[:,0]>=0)&(pixels[:,0]<w)&(pixels[:,1]>=0)&(pixels[:,1]<h)
    return world[xy[inside,1],xy[inside,0]],pixels[inside],{'correlation':float(correlation),'water_iou':overlap,'warp':warp.tolist()}


def propose(case,root,output,surface):
    image,valid,size=prepared(case,root);prior=case.get('prior',{});position=prior.get('position')
    result={'id':case['id'],'status':'no_candidate','candidate':None,'attempts':[]}
    if not position or prior.get('direction_reference')!='T' or prior.get('direction_deg') is None:
        return result
    label_path=root/case['semantic_labels']
    if digest(label_path)!=case['semantic_sha256']:
        raise ValueError('shoreline semantic inputs changed')
    labels=cv2.resize(cv2.imread(str(label_path),0),size,interpolation=cv2.INTER_NEAREST)
    focal=(prior.get('focal_35mm_mm') or 26)*math.hypot(*size)/math.hypot(36,24)
    K=np.array([[focal,0,(size[0]-1)/2],[0,focal,(size[1]-1)/2],[0,0,1.]])
    horizon=horizon_attitude(labels,K)
    if horizon is None:
        return result
    lat,lon=position['latitude'],position['longitude'];ground,covered=surface.sample('dem',lat,lon)
    if not covered:
        return result
    altitude=position.get('altitude_m') or float(ground)+1000
    eye=surface.local(lat,lon,altitude)
    rotation=surface.camera_rotation(lat,lon,prior['direction_deg'],horizon['down_deg'],horizon['roll_deg'])
    reference,depth,world=surface.render(eye,rotation,K,size)
    query_mask=photo_water(image,labels,valid);reference_mask=reference_water(reference,depth)
    pairs=outline_pairs(query_mask,reference_mask,world,depth)
    if pairs is None:
        return result
    points,pixels,outline=pairs;solved=fit(points,pixels,K,size)
    if solved is None:
        return result
    candidate,rotation,score,inliers=solved
    latitude,longitude,height=surface.geodetic(candidate);ground,covered=surface.sample('dem',latitude,longitude)
    gate=bool(covered and 50<float(height-ground)<6000 and np.linalg.norm(candidate-eye)<prior_radius(position))
    # Outline registration supplies synthetic correspondences, not held-out texture
    # evidence. Its numerical fit can only seed a separate terrain-matching pass.
    result['attempts']=[{**score,'geometry_accepted':False,'prior_gate':gate,'seed':0,'step':0,
                        'resolution':list(size),'evidence_kind':'water_outline_proposal',**outline}]
    directory=output/case['id'];directory.mkdir(exist_ok=True)
    np.savez_compressed(directory/'evidence-00-00.npz',eye=candidate,rotation=rotation,K=K,size=size)
    cv2.imwrite(str(directory/'water-query.png'),query_mask.astype(np.uint8)*255)
    cv2.imwrite(str(directory/'water-reference.png'),reference_mask.astype(np.uint8)*255)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('profile',type=Path);parser.add_argument('output',type=Path)
    parser.add_argument('--catalog',type=Path,required=True);parser.add_argument('--case',action='append')
    parser.add_argument('--skip-run',type=Path,action='append',default=[])
    args=parser.parse_args();cv2.setNumThreads(4)
    profile=json.loads(args.profile.read_text())
    if profile['profile']!='with-prior':
        parser.error('shoreline proposals require explicit position, focal and direction priors')
    root=(args.profile.parent/profile['root']).resolve();args.output.mkdir(parents=True,exist_ok=True)
    resolved,skipped=resolved_inputs(args.skip_run)
    config={'profile_sha256':digest(args.profile),'catalog_sha256':digest(args.catalog),'skip_inputs':skipped,
        'opencv':cv2.__version__,'numpy':np.__version__,'code':{p.name:digest(p) for p in Path(__file__).parent.glob('photo_*.py')}}
    path=args.output/'run.json'
    if path.exists() and json.loads(path.read_text())!=config:
        raise ValueError('shoreline inputs changed; use a new output directory')
    write_json(path,config);surface=Surface(args.catalog)
    for case in profile['cases']:
        if case['id'] in resolved or (args.case and case['id'] not in args.case):
            continue
        destination=args.output/(case['id']+'.json')
        if destination.exists():
            continue
        result=propose(case,root,args.output,surface);write_json(destination,result)
        print(json.dumps({'case':case['id'],'proposals':len(result['attempts'])}),flush=True)


if __name__=='__main__':
    main()
