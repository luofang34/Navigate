"""Acquire terrain candidates inside a frozen region without photograph priors."""
import argparse
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np
import onnxruntime as ort

from photo_pose import fit
from photo_rectification import planar_views
from photo_surface import Surface
from photo_testset import digest, write_json


class Features:
    def __init__(self, model):
        options=ort.SessionOptions()
        options.intra_op_num_threads=4
        options.inter_op_num_threads=1
        self.session=ort.InferenceSession(str(model),sess_options=options,providers=['CPUExecutionProvider'])

    def extract(self,image,limit=512):
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        h,w=gray.shape; scale=min(800/w,600/h)
        rw,rh=round(w*scale),round(h*scale); x,y=(800-rw)//2,(600-rh)//2
        padded=np.zeros((600,800),np.float32)
        padded[y:y+rh,x:x+rw]=cv2.resize(gray,(rw,rh))/255
        try:
            points,descriptors,scores=self.session.run(None,{'images':np.broadcast_to(padded,(1,3,600,800)).copy()})
        except ort.capi.onnxruntime_pybind11_state.Fail as error:
            if "Name:'/Where'" in str(error):
                return np.empty((0,2),np.float32),np.empty((0,64),np.float32)
            raise
        points=(points-np.array([x+(scale-1)/2,y+(scale-1)/2]))/scale
        good=(scores>0)&(points[:,0]>=4)&(points[:,1]>=4)&(points[:,0]<w-4)&(points[:,1]<h-4)
        selected=np.flatnonzero(good)
        selected=selected[np.argsort(scores[selected],kind='stable')[::-1]][:limit]
        desc=descriptors[selected]
        desc=desc/np.maximum(np.linalg.norm(desc,axis=1,keepdims=True),1e-12)
        return points[selected].astype(np.float32),desc.astype(np.float32)


def origins(size,span):
    return sorted(set(list(range(0,max(1,size-span+1),span))+[max(0,size-span)]))


def views(image):
    h,w=image.shape[:2]
    crops=[(0,0,w,h)]+[(x,y,x+w//2,y+h//2) for y in [0,h//2] for x in [0,w//2]]
    for x,y,right,bottom in crops:
        patch=image[y:bottom,x:right]; ph,pw=patch.shape[:2]
        for angle in range(0,360,90):
            matrix=cv2.getRotationMatrix2D(((pw-1)/2,(ph-1)/2),angle,1)
            c,s=abs(matrix[0,0]),abs(matrix[0,1]); nw,nh=round(ph*s+pw*c),round(ph*c+pw*s)
            matrix[:,2]+=[(nw-pw)/2,(nh-ph)/2]
            rotated=cv2.warpAffine(patch,matrix,(nw,nh))
            inverse=cv2.invertAffineTransform(matrix); inverse[:,2]+=[x,y]
            yield rotated,inverse,{'crop':[x,y,right,bottom],'rotation_deg':angle}


def reference_bank(surface,extractor,cache,model_hash):
    image=surface.grids['imagery'][0]; h,w=image.shape[:2]
    bank=[]; parts=[]; owners=[]
    for span in [512,1024,2048]:
        for y in origins(h,span):
            for x in origins(w,span):
                path=cache/f'{model_hash[:16]}-{x}-{y}-{span}.npz'
                if path.exists():
                    with np.load(path) as saved:
                        p,d=saved['points'],saved['descriptors']
                else:
                    p,d=extractor.extract(image[y:y+span,x:x+span])
                    np.savez_compressed(path,points=p,descriptors=d)
                if len(p)<8:
                    continue
                owners.extend([len(bank)]*len(d))
                bank.append({'x':x,'y':y,'span':span,'points':p})
                parts.append(d)
        print(json.dumps({'stage':'index','span':span,'blocks':len(bank)}),flush=True)
    if not parts:
        raise ValueError('reference has no usable visual features')
    matcher=cv2.FlannBasedMatcher(dict(algorithm=1,trees=4),dict(checks=128))
    matcher.add([np.concatenate(parts)]); matcher.train()
    return bank,parts,np.array(owners),matcher


def ground_points(surface,pixels):
    _,_,z,x0,y0=surface.grids['imagery']
    x=(pixels[:,0]+.5)/256+x0; y=(pixels[:,1]+.5)/256+y0
    lon=x/2**z*360-180; lat=np.degrees(np.arctan(np.sinh(np.pi*(1-2*y/2**z))))
    alt,covered=surface.sample('dem',lat,lon)
    la,lo=np.radians(lat),np.radians(lon)
    xyz=np.column_stack([np.cos(la)*np.cos(lo),np.cos(la)*np.sin(lo),np.sin(la)])
    world=(xyz*(6371008.8+alt[:,None])-surface.origin)@surface.basis
    return world,covered,lat,lon


def query_views(image,mask,labels):
    yield from views(image)
    for rectified,inverse,description in planar_views(image,mask,labels):
        for rotated,turn,view in list(views(rectified))[:4]:
            transform=inverse@np.vstack([turn,[0,0,1]])
            yield rotated,transform,{**description,**view}


def retrieve(case,root,directory,surface,extractor,bank,parts,owners,matcher):
    directory.mkdir(parents=True,exist_ok=True)
    if digest(root/case['image'])!=case['image_sha256']:
        raise ValueError('query pixels changed')
    if digest(root/case['valid_mask'])!=case['mask_sha256']:
        raise ValueError('query mask changed')
    image=cv2.imread(str(root/case['image'])); h,w=image.shape[:2]
    scale=640/max(w,h); size=(round(w*scale),round(h*scale))
    image=cv2.resize(image,size,interpolation=cv2.INTER_AREA)
    mask=cv2.resize(cv2.imread(str(root/case['valid_mask']),0),size,interpolation=cv2.INTER_NEAREST)
    if digest(root/case['semantic_labels'])!=case['semantic_sha256']:
        raise ValueError('semantic labels changed')
    labels=cv2.resize(cv2.imread(str(root/case['semantic_labels']),0),size,interpolation=cv2.INTER_NEAREST)
    queries=[];votes=[]
    for k,(rotated,inverse,view) in enumerate(query_views(image,mask,labels)):
        p,d=extractor.extract(rotated,1024)
        if inverse.shape==(2,3):
            p=p@inverse[:,:2].T+inverse[:,2]
        else:
            projected=np.column_stack([p,np.ones(len(p))])@inverse.T
            p=projected[:,:2]/projected[:,2,None]
        xy=np.rint(p).astype(int)
        valid=(xy[:,0]>=0)&(xy[:,0]<size[0])&(xy[:,1]>=0)&(xy[:,1]<size[1])
        indices=np.flatnonzero(valid)
        indices=indices[mask[xy[indices,1],xy[indices,0]]>0]
        p=p[indices].astype(np.float32);d=d[indices]
        queries.append((p,d,view)); counts=np.zeros(len(bank))
        if len(d):
            for row in matcher.knnMatch(d,k=4):
                used=set()
                for match in row:
                    block=owners[match.trainIdx]
                    if match.distance<.8 and block not in used:
                        counts[block]+=1;used.add(block)
        votes.extend((float(counts[i]/max(1,math.sqrt(len(d)))),int(i),k,int(counts[i]))
                     for i in np.argsort(counts,kind='stable')[-12:])
    candidates=[];poses=[]
    for _,index,k,vote in sorted(votes,reverse=True)[:64]:
        if vote<6:
            continue
        item=bank[index];rp=item['points'];rd=parts[index];qp,qd,view=queries[k]
        similarity=rd@qd.T;qi=np.argmax(similarity,axis=1);ri=np.argmax(similarity,axis=0)
        good=(ri[qi]==np.arange(len(rp)))&(similarity[np.arange(len(rp)),qi]>.65)
        a=rp[good]+[item['x'],item['y']];b=qp[qi[good]]
        if len(a)<8:
            continue
        cv2.setRNGSeed(0)
        _,inliers=cv2.findHomography(a,b,cv2.USAC_MAGSAC,4,maxIters=10000,confidence=.999)
        valid=np.zeros(len(a),bool) if inliers is None else inliers.ravel().astype(bool)
        world,covered,lat,lon=ground_points(surface,a)
        candidate={'block':index,'view':view,'votes':vote,'matches':len(a),'homography_inliers':int(valid.sum()),
                   'latitude':float(np.median(lat)),'longitude':float(np.median(lon))}
        candidates.append(candidate)
        np.savez_compressed(directory/f'matches-{index}-{k}.npz',reference=a,query=b,world=world,covered=covered)
        # Homography is only a retrieval score; a nonplanar terrain pose needs an independent PnP check.
        for equivalent in [14,20,26,35,52,85,132]:
            focal=equivalent*math.hypot(*size)/math.hypot(36,24)
            K=np.array([[focal,0,(size[0]-1)/2],[0,focal,(size[1]-1)/2],[0,0,1.]])
            result=fit(world[covered],b[covered],K,size)
            if result is None or not result[2]['geometry_accepted']:
                continue
            eye,rotation,score,inliers=result;plat,plon,palt=surface.geodetic(eye)
            terrain,inside=surface.sample('dem',plat,plon)
            if not inside or not 50<float(palt-terrain)<6000:
                continue
            filename=f'pose-{index}-{k}-{equivalent}.npz'
            np.savez_compressed(directory/filename,eye=eye,rotation=rotation,K=K,size=size)
            poses.append({'file':filename,'score':score,'latitude':float(plat),'longitude':float(plon),'altitude_m':float(palt)})
    candidates.sort(key=lambda v:v['homography_inliers'],reverse=True)
    return {'id':case['id'],'status':'candidate_requires_refinement' if poses else 'retrieval_only',
            'scope':'regional; no photograph metadata or prior','candidates':candidates,'poses':poses}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('profile',type=Path); parser.add_argument('output',type=Path)
    parser.add_argument('--catalog',type=Path,required=True);parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--case',action='append')
    args=parser.parse_args();cv2.setNumThreads(4);cv2.setRNGSeed(0)
    profile=json.loads(args.profile.read_text())
    if profile['profile']!='no-prior':
        parser.error('acquisition accepts only the metadata-free profile')
    allowed={'id','image','size','valid_mask','mask_status','image_sha256','mask_sha256',
             'semantic_labels','semantic_sha256'}
    if any(set(case)-allowed for case in profile['cases']):
        raise ValueError('blind profile contains unexpected fields')
    root=(args.profile.parent/profile['root']).resolve();output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=True)
    config={'profile_sha256':digest(args.profile),'catalog_sha256':digest(args.catalog),
            'model_sha256':digest(args.model),'onnxruntime':ort.__version__,'opencv':cv2.__version__,
            'code':{p.name:digest(p) for p in Path(__file__).parent.glob('photo_*.py')}}
    run=output/'run.json'
    if run.exists() and json.loads(run.read_text())!=config:
        raise ValueError('run configuration changed; use a new output directory')
    write_json(run,config)
    cache=output/'features';cache.mkdir(exist_ok=True)
    surface=Surface(args.catalog);extractor=Features(args.model)
    bank,parts,owners,matcher=reference_bank(surface,extractor,cache,config['model_sha256'])
    for case in profile['cases']:
        if args.case and case['id'] not in args.case:
            continue
        destination=output/(case['id']+'.json')
        if destination.exists():
            continue
        start=time.perf_counter()
        result=retrieve(case,root,output/case['id'],surface,extractor,bank,parts,owners,matcher)
        result['seconds']=time.perf_counter()-start;write_json(destination,result)
        print(json.dumps({'id':case['id'],'status':result['status'],'poses':len(result['poses']),
                          'best':result['candidates'][:1],'seconds':result['seconds']}),flush=True)


if __name__=='__main__':
    main()
