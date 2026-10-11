"""Locate same-flight photos against terrain-backed, geometrically checked photo anchors."""
import argparse
from datetime import datetime
import json
import math
from pathlib import Path

import cv2
import numpy as np

from photo_match import Matcher,match_pixels
from photo_pose import fit,ground_pairs,candidate_priority,prior_radius
from photo_surface import Surface,frame,reframe_camera
from photo_testset import digest,write_json
from photo_runs import resolved_inputs


def prepared(case,root):
    if digest(root/case['image'])!=case['image_sha256'] or digest(root/case['valid_mask'])!=case['mask_sha256']:
        raise ValueError('photo anchor inputs changed')
    image=cv2.imread(str(root/case['image']));h,w=image.shape[:2]
    scale=640/max(w,h);size=(round(w*scale),round(h*scale))
    image=cv2.resize(image,size,interpolation=cv2.INTER_AREA)
    mask=cv2.resize(cv2.imread(str(root/case['valid_mask']),0),size,interpolation=cv2.INTER_NEAREST)
    return image,mask,size


def matching_pixels(image,mask,matcher,contrast=False):
    return match_pixels(image,mask,contrast,getattr(matcher,'color',False))


def load_anchors(runs,catalog_hash,max_depth=0,catalog_paths=()):
    anchors={}
    frames={digest(path):frame(json.loads(path.read_text())) for path in catalog_paths}
    for run in runs:
        source_hash=json.loads((run/'run.json').read_text())['catalog_sha256']
        if source_hash!=catalog_hash and source_hash not in frames:
            raise ValueError('anchor coordinate frame requires its explicit source catalog')
        for path in sorted(run.glob('q-*.json')):
            data=json.loads(path.read_text())
            if data['status']!='candidate_requires_review':
                continue
            candidate=data['candidate']
            depth=candidate.get('anchor_depth',int(candidate.get('evidence_kind')=='photo_to_terrain_anchor'))
            if depth>max_depth:
                continue
            score=data['candidate']['score']
            if not score['geometry_accepted']:
                continue
            merit=candidate_priority(candidate)
            q=data['id'];previous=anchors.get(q)
            if previous is None or merit>previous['merit']:
                lineage=candidate.get('anchor_lineage',[])
                if not lineage and candidate.get('anchor'):
                    lineage=[{'id':candidate['anchor'],'pose_sha256':candidate['anchor_state_sha256'],
                              'report_sha256':candidate['anchor_report_sha256']}]
                anchors[q]={'merit':merit,'path':run/q/'candidate.npz','report':path,
                            'depth':depth,'lineage':lineage,'source_frame':frames.get(source_hash),
                            'catalog_sha256':source_hash}
    return anchors


def match_pair(matcher,reference,query,directory):
    a=cv2.imread(str(reference),0);b=cv2.imread(str(query),0)
    if a.shape==b.shape or getattr(matcher,'variable_sizes',False):
        return matcher.match(reference,query)['pairs']
    h=max(a.shape[0],b.shape[0]);w=max(a.shape[1],b.shape[1]);offsets=[];paths=[]
    for index,image in enumerate([a,b]):
        y=(h-image.shape[0])//2;x=(w-image.shape[1])//2
        padded=np.zeros((h,w),np.uint8);padded[y:y+image.shape[0],x:x+image.shape[1]]=image
        path=directory/f'padded-{index}.png';cv2.imwrite(str(path),padded)
        offsets.append([x,y]);paths.append(path)
    pairs=np.asarray(matcher.match(*paths)['pairs'],np.float64).reshape(-1,2,2)
    return pairs-np.array(offsets)


def reference_windows(image,scales):
    height,width=image.shape[:2]
    for scale in scales:
        if scale==1:
            yield image,np.array([0.,0.]),np.array([1.,1.])
            continue
        cw,ch=round(width/scale),round(height/scale)
        count=math.ceil(2*(scale-1))+1
        for y in np.linspace(0,height-ch,count).round().astype(int):
            for x in np.linspace(0,width-cw,count).round().astype(int):
                patch=image[y:y+ch,x:x+cw]
                if np.mean(np.any(patch>0,axis=2))<.2:
                    continue
                yield cv2.resize(patch,(width,height)),np.array([x,y]),np.array([cw/width,ch/height])


def match_windows(matcher,reference,query,directory,scales):
    if scales==[1.]:
        return match_pair(matcher,reference,query,directory)
    image=cv2.imread(str(reference));collected=[]
    for patch,offset,ratio in reference_windows(image,scales):
        path=directory/'anchor-window.png';cv2.imwrite(str(path),patch)
        pairs=np.asarray(match_pair(matcher,path,query,directory),np.float64).reshape(-1,2,2)
        pairs[:,0]=(pairs[:,0]+.5)*ratio-.5+offset
        collected.append(pairs)
    if not collected:
        return np.empty((0,2,2))
    pairs=np.concatenate(collected)
    _,unique=np.unique(np.round(pairs.reshape(-1,4)*2).astype(np.int64),axis=0,return_index=True)
    return pairs[np.sort(unique)]


def uses_query_capture(identifier,anchor_id,anchor,captures):
    chain=[anchor_id,*[node['id'] for node in anchor['lineage']]]
    group=captures.get(identifier)
    return any(q==identifier or (group is not None and captures.get(q)==group) for q in chain)


def select_anchors(case,anchors,cases,surface,flights,captures,association,radius):
    identifier=case['id'];position=case['prior']['position']
    clock=lambda c:datetime.fromisoformat(c['prior']['capture_time']).timestamp()
    origin=surface.local(position['latitude'],position['longitude'],0)
    candidates=[]
    for q in anchors:
        if uses_query_capture(identifier,q,anchors[q],captures):
            continue
        if association=='flight':
            if flights.get(q)==flights.get(identifier):
                candidates.append((abs(clock(cases[q])-clock(case)),q))
        else:
            point=cases[q]['prior']['position']
            distance=float(np.linalg.norm(surface.local(point['latitude'],point['longitude'],0)-origin))
            if distance<=radius:
                candidates.append((distance,q))
    return [q for _,q in sorted(candidates)]


def match_case(case,anchors,cases,root,output,surface,matcher,flights,captures,limit,association,radius,crop_scales=(1.,)):
    identifier=case['id'];directory=output/identifier;directory.mkdir(exist_ok=True)
    image,mask,size=prepared(case,root)
    query=directory/'query.png';cv2.imwrite(str(query),matching_pixels(image,mask,matcher))
    prior=case['prior'];position=prior.get('position')
    if position is None:
        return {'id':identifier,'status':'unsupported_prior','candidate':None,'attempts':[]}
    focal=(prior.get('focal_35mm_mm') or 26)*math.hypot(*size)/math.hypot(36,24)
    K=np.array([[focal,0,(size[0]-1)/2],[0,focal,(size[1]-1)/2],[0,0,1.]])
    candidates=select_anchors(case,anchors,cases,surface,flights,captures,association,radius)
    records=[];best=None
    for index,q in enumerate(candidates[:limit]):
        anchor=anchors[q];cache=output/'anchors'/q
        cache.mkdir(parents=True,exist_ok=True)
        if not (cache/'surface.npz').exists():
            reference,valid,ref_size=prepared(cases[q],root)
            with np.load(anchor['path']) as state:
                eye,rotation=state['eye'],state['rotation']
                if anchor['source_frame'] is not None:
                    eye,rotation=reframe_camera(eye,rotation,anchor['source_frame'],(surface.basis,surface.origin))
                _,depth,world=surface.render(eye,rotation,state['K'],ref_size)
            depth[valid==0]=0
            cv2.imwrite(str(cache/'reference.png'),matching_pixels(reference,(depth>0).astype(np.uint8),matcher))
            np.savez_compressed(cache/'surface.npz',depth=depth,world=world)
        with np.load(cache/'surface.npz') as reference:
            pairs=match_windows(matcher,cache/'reference.png',query,directory,list(crop_scales))
            points,pixels=ground_pairs(pairs,reference['depth'],reference['world'],mask)
        solved=fit(points,pixels,K,size,(focal*.75,focal*1.25))
        record={'anchor':q,'pairs':len(points),'seed':index,'step':0,'resolution':list(size)}
        if solved is None:
            records.append({**record,'status':'insufficient_geometry'});continue
        eye,rotation,score,inliers=solved
        intrinsics=K.copy();intrinsics[0,0]=intrinsics[1,1]=score['focal_px']
        np.savez_compressed(directory/f'evidence-{index:02}-00.npz',
                            eye=eye,rotation=rotation,K=intrinsics,size=size,
                            points=points,pixels=pixels,inliers=inliers)
        lat,lon,alt=surface.geodetic(eye);terrain,covered=surface.sample('dem',lat,lon)
        horizontal=float(np.linalg.norm(surface.local(lat,lon,0)-surface.local(position['latitude'],position['longitude'],0)))
        gate=bool(covered and 30<float(alt-terrain)<6000 and horizontal<prior_radius(position))
        records.append({**record,**score,'prior_gate':gate,'horizontal_prior_delta_m':horizontal})
        if not gate or not score['geometry_accepted']:
            continue
        merit=score['inliers']/(1+score['holdout_median_px'])
        if best is None or merit>best[0]:
            np.savez_compressed(directory/'candidate.npz',eye=eye,rotation=rotation,K=intrinsics,size=size,
                                points=points,pixels=pixels,inliers=inliers)
            candidate={'latitude':float(lat),'longitude':float(lon),'altitude_m':float(alt),'score':score,
                'evidence_kind':'photo_to_terrain_anchor','anchor':q,'anchor_state_sha256':digest(anchor['path']),
                'anchor_report_sha256':digest(anchor['report']),'anchor_depth':anchor['depth']+1,
                'anchor_lineage':[{'id':q,'pose_sha256':digest(anchor['path']),
                                   'report_sha256':digest(anchor['report'])},*anchor['lineage']]}
            best=(merit,candidate)
    if best:
        with np.load(directory/'candidate.npz') as state:
            color,_,_=surface.render(state['eye'],state['rotation'],state['K'],size)
        cv2.imwrite(str(directory/'comparison.jpg'),np.concatenate([image,color,cv2.addWeighted(image,.5,color,.5,0)],axis=1))
    return {'id':identifier,'status':'candidate_requires_review' if best else 'no_candidate',
            'candidate':best[1] if best else None,'attempts':records}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('profile',type=Path);parser.add_argument('output',type=Path)
    parser.add_argument('--catalog',type=Path,required=True);parser.add_argument('--anchor-run',type=Path,action='append',required=True)
    parser.add_argument('--anchor-catalog',type=Path,action='append',default=[],
                        help='source catalog for an anchor using another coordinate frame')
    parser.add_argument('--worker',type=Path,default=Path(__file__).resolve().parents[2] / 'target/release/navigate-recording-worker')
    parser.add_argument('--model',type=Path);parser.add_argument('--runtime',type=Path)
    parser.add_argument('--backend',choices=['loftr','roma'],default='loftr')
    parser.add_argument('--roma-model',type=Path);parser.add_argument('--roma-encoder',type=Path)
    parser.add_argument('--device',choices=['cpu','mps'],default='cpu')
    parser.add_argument('--grayscale',action='store_true')
    parser.add_argument('--case',action='append');parser.add_argument('--anchors-per-case',type=int,default=4)
    parser.add_argument('--skip-run',type=Path,action='append',default=[],help='skip geometrically accepted cases from a completed pass')
    parser.add_argument('--association',choices=['flight','spatial'],default='flight')
    parser.add_argument('--radius-m',type=float,default=5000)
    parser.add_argument('--max-anchor-depth',type=int,default=0,choices=range(5))
    parser.add_argument('--crop-scales',type=float,nargs='+',default=[1.],help='overlapping anchor crops for changes in optical or digital zoom')
    args=parser.parse_args();cv2.setNumThreads(4)
    if args.radius_m<=0 or args.anchors_per_case<1:
        parser.error('anchor radius and count must be positive')
    if any(not 1<=scale<=8 for scale in args.crop_scales):
        parser.error('anchor crop scales must be within 1 to 8')
    if args.backend=='roma':
        if args.roma_model is None or args.roma_encoder is None:
            parser.error('RoMa requires explicit local model and encoder weights')
        from photo_roma import configuration,RomaMatcher
        backend=configuration(args.roma_model,args.roma_encoder,args.device,args.grayscale)
    else:
        if args.grayscale:
            parser.error('grayscale selection is for RoMa; LoFTR already uses grayscale')
        if args.model is None or args.runtime is None:
            parser.error('LoFTR requires its local model and ONNX runtime')
        backend={'backend':'loftr','model_sha256':digest(args.model),
                 'worker_sha256':digest(args.worker),'runtime_sha256':digest(args.runtime)}
    profile=json.loads(args.profile.read_text())
    if profile['profile']!='with-prior':
        parser.error('same-flight association uses the explicitly declared prior profile')
    root=(args.profile.parent/profile['root']).resolve();output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    cases={c['id']:c for c in profile['cases']}
    manifest_path=args.profile.parent/profile['manifest'] if profile.get('manifest') else root/'manifest.json'
    manifest=json.loads(manifest_path.read_text())
    flights={q:f['id'] for f in manifest['flights'] for q in f['photos']}
    captures={q:i for i,capture in enumerate(manifest['capture_groups']) for q in capture['photos']}
    resolved,skip_inputs=resolved_inputs(args.skip_run)
    config={'profile_sha256':digest(args.profile),'catalog_sha256':digest(args.catalog),
            'manifest_sha256':digest(manifest_path),**backend,
            'opencv':cv2.__version__,'numpy':np.__version__,
            'anchors_per_case':args.anchors_per_case,'association':args.association,'radius_m':args.radius_m,
            'max_anchor_depth':args.max_anchor_depth,'crop_scales':args.crop_scales,
            'code':{p.name:digest(p) for p in Path(__file__).parent.glob('photo_*.py')}}
    if skip_inputs:
        config['skip_inputs']=skip_inputs
    anchors=load_anchors(args.anchor_run,config['catalog_sha256'],args.max_anchor_depth,args.anchor_catalog)
    config['anchors']={q:{'pose':digest(a['path']),'report':digest(a['report']),
                          'catalog_sha256':a['catalog_sha256']} for q,a in anchors.items()}
    if (output/'run.json').exists() and json.loads((output/'run.json').read_text())!=config:
        raise ValueError('tracking inputs changed; choose a new output directory')
    write_json(output/'run.json',config)
    surface=Surface(args.catalog)
    matcher=(RomaMatcher(args.roma_model,args.roma_encoder,args.device,args.grayscale) if args.backend=='roma'
             else Matcher(args.worker,args.model,args.runtime))
    try:
        for case in profile['cases']:
            if case['id'] in anchors or case['id'] in resolved or (args.case and case['id'] not in args.case):
                continue
            destination=output/(case['id']+'.json')
            if destination.exists():
                continue
            result=match_case(case,anchors,cases,root,output,surface,matcher,flights,captures,
                              args.anchors_per_case,args.association,args.radius_m,args.crop_scales)
            write_json(destination,result)
            print(json.dumps({'id':case['id'],'status':result['status'],'candidate':result['candidate']}),flush=True)
    finally:
        matcher.close()


if __name__=='__main__':
    main()
