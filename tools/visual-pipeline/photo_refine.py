"""Verify proposed camera poses against terrain with a local image matcher."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from photo_match import Matcher
from photo_pose import fit,ground_pairs,prior_radius
from photo_render_check import verify_pose,merit
from photo_surface import Surface,ContextSurface,frame,reframe_camera
from photo_preprocessing import select_preprocessing
from photo_testset import digest,write_json
from photo_tracking import prepared,matching_pixels


def proposals(run,result,limit):
    directory=run/result['id']
    if result.get('candidate'):
        return [directory/'candidate.npz']
    ranked=[a for a in result.get('attempts',[]) if a.get('prior_gate') and a.get('inliers',0)>=30
            and max(a.get('resolution',[640]))==640]
    ranked.sort(key=lambda a:a['inliers']*a['image_support_fraction']/(1+a['holdout_median_px']),reverse=True)
    selected=[];poses=[]
    for item in ranked:
        path=directory/f"evidence-{item['seed']:02}-{item['step']:02}.npz"
        with np.load(path) as state:
            eye=state['eye'];rotation=state['rotation']
            if any(np.linalg.norm(eye-e)<25 and np.trace(rotation.T@rot)>2.99 for e,rot in poses):
                continue
            poses.append((eye.copy(),rotation.copy()));selected.append(path)
        if len(selected)==limit:
            break
    return selected


def refine(case,root,output,surface,matcher,seeds,steps,contrast,source_frame,preprocessing_search=False):
    directory=output/case['id'];directory.mkdir(exist_ok=True)
    image,mask,size=prepared(case,root)
    query=directory/'query.png';cv2.imwrite(str(query),matching_pixels(image,mask,matcher,contrast))
    records=[];best=None;preprocessing=[]
    for index,path in enumerate(seeds):
        with np.load(path) as state:
            eye=state['eye'].copy();rotation=state['rotation'].copy()
            K=state['K'].copy() if 'K' in state else state['input_K'].copy()
            if 'focal_px' in state:
                K[0,0]=K[1,1]=state['focal_px']
        eye,rotation=reframe_camera(eye,rotation,source_frame,(surface.basis,surface.origin))
        selected_matcher=matcher
        if preprocessing_search:
            color,depth,_=surface.render(eye,rotation,K,size)
            reference=directory/f'preprocessing-{index:02}.png'
            cv2.imwrite(str(reference),matching_pixels(color,depth>0,matcher,contrast))
            selected_matcher,selection=select_preprocessing(matcher,reference,query,mask,
                depth>0,directory/f'preprocessing-{index:02}')
            preprocessing.append({'seed':index,**selection})
        focal=K[0,0]
        for step in range(steps):
            color,depth,world=surface.render(eye,rotation,K,size)
            reference=directory/'reference.png'
            cv2.imwrite(str(reference),matching_pixels(color,(depth>0).astype(np.uint8),matcher,contrast))
            response=selected_matcher.match(reference,query,refinement=getattr(selected_matcher,'patch_refinement',True))
            points,pixels=ground_pairs(response['pairs'],depth,world,mask)
            np.savez_compressed(directory/f'matches-{index:02}-{step:02}.npz',
                                points=points,pixels=pixels,input_K=K,size=size)
            result=fit(points,pixels,K,size,(focal*.85,focal*1.15))
            record={'seed':index,'step':step,'raw_pairs':len(response['pairs']),'backend':response['backend']}
            if result is None:
                records.append({**record,'status':'insufficient_geometry'});break
            eye,rotation,score,inliers=result
            K[0,0]=K[1,1]=score['focal_px']
            lat,lon,alt=surface.geodetic(eye);ground,covered=surface.sample('dem',lat,lon)
            p=case['prior']['position'];origin=surface.local(p['latitude'],p['longitude'],p.get('altitude_m') or alt)
            gate=bool(covered and 50<float(alt-ground)<6000 and np.linalg.norm(eye-origin)<prior_radius(p))
            records.append({**record,**score,'prior_gate':gate})
            print(json.dumps({'case':case['id'],**records[-1]}),flush=True)
            if not gate or score['inliers']<25:
                break
            if not score['geometry_accepted']:
                continue
            verification=verify_pose(surface,eye,rotation,K,size,selected_matcher,query,mask,
                                     directory/f'verification-{index:02}-{step:02}',contrast)
            records[-1]['fixed_render']=verification
            print(json.dumps({'case':case['id'],'seed':index,'step':step,
                              'fixed_render':verification}),flush=True)
            rank=merit(verification)
            if rank is not None and (best is None or rank>best[0]):
                np.savez_compressed(directory/'candidate.npz',eye=eye,rotation=rotation,K=K,size=size,
                                    points=points,pixels=pixels,inliers=inliers)
                best=(rank,{'latitude':float(lat),'longitude':float(lon),'altitude_m':float(alt),
                    'score':score,'evidence_kind':'terrain_refinement','seed':index,'step':step,
                    'fixed_render':verification})
        if best:
            break
    if best:
        with np.load(directory/'candidate.npz') as state:
            color,_,_=surface.render(state['eye'],state['rotation'],state['K'],size)
        cv2.imwrite(str(directory/'comparison.jpg'),np.concatenate([image,color,cv2.addWeighted(image,.5,color,.5,0)],axis=1))
    return {'id':case['id'],'status':'candidate_requires_review' if best else 'no_candidate',
            'candidate':best[1] if best else None,'attempts':records,
            'preprocessing':preprocessing,
            'initial_states':{str(p.resolve()):digest(p) for p in seeds}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('profile',type=Path);parser.add_argument('output',type=Path)
    parser.add_argument('--catalog',type=Path,required=True);parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--initial-catalog',type=Path,help='coordinate frame of initial poses when refining against another catalog')
    parser.add_argument('--context-catalog',type=Path,help='frozen wider terrain and imagery for uncovered local samples')
    parser.add_argument('--preprocessing-search',action='store_true',help='select full/cropped and color/grayscale RoMa inputs at each seed')
    parser.add_argument('--worker',type=Path,default=Path(__file__).resolve().parents[2] / 'target/release/navigate-recording-worker')
    parser.add_argument('--model',type=Path);parser.add_argument('--runtime',type=Path)
    parser.add_argument('--backend',choices=['loftr','roma'],default='loftr')
    parser.add_argument('--roma-model',type=Path);parser.add_argument('--roma-encoder',type=Path)
    parser.add_argument('--device',choices=['cpu','mps'],default='cpu')
    parser.add_argument('--grayscale',action='store_true')
    parser.add_argument('--case',action='append');parser.add_argument('--include-proposals',action='store_true')
    parser.add_argument('--attempts',type=int,default=3);parser.add_argument('--steps',type=int,default=5)
    parser.add_argument('--contrast',action='store_true')
    args=parser.parse_args();cv2.setNumThreads(4)
    if args.attempts<1 or args.steps<1:
        parser.error('attempts and steps must be positive')
    if args.preprocessing_search and (args.backend!='roma' or args.grayscale or args.contrast):
        parser.error('preprocessing search requires color RoMa without contrast normalization')
    if args.backend=='roma':
        if args.roma_model is None or args.roma_encoder is None:
            parser.error('RoMa requires explicit local model and encoder weights')
        if args.contrast:
            parser.error('contrast normalization is only supported by LoFTR')
        from photo_roma import configuration,RomaMatcher
        backend=configuration(args.roma_model,args.roma_encoder,args.device,args.grayscale)
    else:
        if args.grayscale:
            parser.error('grayscale selection is for RoMa; LoFTR already uses grayscale')
        if args.model is None or args.runtime is None:
            parser.error('LoFTR requires its local model and ONNX runtime')
        backend={'backend':'loftr','model_sha256':digest(args.model),
                 'worker_sha256':digest(args.worker),'runtime_sha256':digest(args.runtime)}
    profile=json.loads(args.profile.read_text());parent=json.loads((args.run/'run.json').read_text())
    initial_catalog=args.initial_catalog or args.catalog
    if profile['profile']!='with-prior' or digest(initial_catalog)!=parent['catalog_sha256']:
        parser.error('refinement requires explicit priors and the initial pose coordinate frame')
    root=(args.profile.parent/profile['root']).resolve();output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    config={'profile_sha256':digest(args.profile),'catalog_sha256':digest(args.catalog),
        'initial_run_sha256':digest(args.run/'run.json'),'initial_catalog_sha256':digest(initial_catalog),
        **backend,'opencv':cv2.__version__,'numpy':np.__version__,
        'attempts':args.attempts,'steps':args.steps,'contrast':args.contrast,'include_proposals':args.include_proposals,
        'preprocessing_search':args.preprocessing_search,
        'context_catalog_sha256':digest(args.context_catalog) if args.context_catalog else None,
        'code':{p.name:digest(p) for p in Path(__file__).parent.glob('photo_*.py')}}
    if (output/'run.json').exists() and json.loads((output/'run.json').read_text())!=config:
        raise ValueError('refinement inputs changed; choose a new output directory')
    write_json(output/'run.json',config)
    surface=Surface(args.catalog)
    if args.context_catalog:
        surface=ContextSurface(surface,Surface(args.context_catalog))
    matcher=(RomaMatcher(args.roma_model,args.roma_encoder,args.device,args.grayscale) if args.backend=='roma'
             else Matcher(args.worker,args.model,args.runtime))
    source_frame=frame(json.loads(initial_catalog.read_text()))
    try:
        for case in profile['cases']:
            q=case['id'];report=args.run/(q+'.json')
            if (args.case and q not in args.case) or not report.exists():
                continue
            previous=json.loads(report.read_text())
            if not previous.get('candidate') and not args.include_proposals:
                continue
            seeds=proposals(args.run,previous,args.attempts)
            if not seeds:
                continue
            destination=output/(q+'.json')
            if destination.exists():
                saved=json.loads(destination.read_text())
                if saved['initial_states']!={str(p.resolve()):digest(p) for p in seeds}:
                    raise ValueError('initial poses changed; choose a new output directory')
                continue
            result=refine(case,root,output,surface,matcher,seeds,args.steps,args.contrast,source_frame,args.preprocessing_search)
            result['initial_report_sha256']=digest(report)
            write_json(destination,result)
            print(json.dumps({'case':q,'status':result['status']}),flush=True)
    finally:
        matcher.close()


if __name__=='__main__':
    main()
