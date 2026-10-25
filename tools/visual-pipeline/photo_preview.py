"""Package terrain candidates and unlocated photos into flight-grouped local media."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import cv2
import numpy as np

from photo_surface import Surface, RADIUS
from photo_pose import candidate_priority
from photo_testset import digest, write_json

# The globe renderer uses x east at 90 E, y at the north pole, and z at 0 E.
TO_GLOBE=np.array([[0,1,0],[0,0,1],[1,0,0.]])


def publish_asset(path,write):
    # An open viewer can request an asset while a preview is being regenerated.
    # Keep the complete previous file readable until its replacement is ready.
    with tempfile.TemporaryDirectory(prefix='.publish-',dir=path.parent) as temporary:
        staged=Path(temporary)/path.name
        write(staged)
        staged.replace(path)


def write_image(path,image):
    def encode(staged):
        if not cv2.imwrite(str(staged),image):
            raise OSError(f'cannot encode preview image: {path}')
    publish_asset(path,encode)


def quaternion(matrix):
    vector=cv2.Rodrigues(matrix)[0].ravel();angle=np.linalg.norm(vector)
    xyz=vector*np.sin(angle/2)/angle if angle>1e-12 else vector/2
    return [float(np.cos(angle/2)),*xyz.tolist()]


def view(eye,rotation,surface,distance=10000.):
    world_eye=TO_GLOBE@(surface.basis@eye+surface.origin)
    orientation=TO_GLOBE@surface.basis@rotation
    target=world_eye-orientation[:,2]*distance
    lat,lon,_=surface.geodetic(eye)
    return {'centre':{'lat':float(lat),'lon':float(lon)},'zoom':10,'bearing':0,'pitch':0,
            'auto_tilt':False,'three_dimensional':True,'north_locked':False,
            'globe_pose':{'target':(target/np.linalg.norm(target)).tolist(),
                          'target_elevation_meters':float(np.linalg.norm(target)-RADIUS),
                          'orientation':quaternion(orientation.T),'distance_meters':distance}}


def overview(surface,locations):
    lat=float(np.mean([p[0] for p in locations]));lon=float(np.mean([p[1] for p in locations]))
    span=max(np.linalg.norm(surface.local(p[0],p[1],0)-surface.local(lat,lon,0)) for p in locations)
    ground,covered=surface.sample('dem',lat,lon)
    elevation=float(ground) if covered else 0
    eye=surface.local(lat-.035,lon,elevation+max(8000,span*2))
    return view(eye,surface.camera_rotation(lat-.035,lon,0,65),surface,max(10000,span*2))


def map_corners(surface,x0,y0,x1,y1):
    _,_,z,tx,ty=surface.grids['imagery']
    def geo(x,y):
        lon=(tx+x/256)/2**z*360-180
        lat=np.degrees(np.arctan(np.sinh(np.pi*(1-2*(ty+y/256)/2**z))))
        return [float(lon),float(lat)]
    return [geo(x0,y0),geo(x1,y0),geo(x1,y1),geo(x0,y1)]


def ground_overlay(surface,state,image,mask,directory):
    size=tuple(int(v) for v in state['size']);K=state['K'];eye=state['eye'];rotation=state['rotation']
    reference,depth,world=surface.render(eye,rotation,K,size)
    write_image(directory/'reference.png',reference)
    valid=(depth>0)&(mask>0)
    if valid.sum()<100:
        raise ValueError('candidate has no visible masked ground')
    lat,lon,_=surface.geodetic(world[valid]);grid,_,z,tx,ty=surface.grids['imagery']
    x=((lon+180)/360*2**z-tx)*256
    y=((1-np.arcsinh(np.tan(np.radians(lat)))/np.pi)/2*2**z-ty)*256
    x0=max(0,int(np.floor(x.min())));x1=min(grid.shape[1],int(np.ceil(x.max()))+1)
    y0=max(0,int(np.floor(y.min())));y1=min(grid.shape[0],int(np.ceil(y.max()))+1)
    scale=min(1,1536/max(x1-x0,y1-y0));width=max(2,round((x1-x0)*scale));height=max(2,round((y1-y0)*scale))
    u,v=np.meshgrid(x0+(np.arange(width)+.5)*(x1-x0)/width,y0+(np.arange(height)+.5)*(y1-y0)/height)
    lon=(tx+u/256)/2**z*360-180
    lat=np.degrees(np.arctan(np.sinh(np.pi*(1-2*(ty+v/256)/2**z))))
    altitude,covered=surface.sample('dem',lat,lon)
    la,lo=np.radians(lat),np.radians(lon)
    unit=np.stack([np.cos(la)*np.cos(lo),np.cos(la)*np.sin(lo),np.sin(la)],-1)
    points=(unit*(RADIUS+altitude[...,None])-surface.origin)@surface.basis
    camera=(points-eye)@rotation;optical=-camera[...,2]
    safe=np.maximum(optical,.001)
    px=(K[0,0]*camera[...,0]/safe+K[0,2]).astype(np.float32)
    py=(-K[1,1]*camera[...,1]/safe+K[1,2]).astype(np.float32)
    colors=cv2.remap(image,px,py,cv2.INTER_LINEAR)
    alpha=cv2.remap(mask,px,py,cv2.INTER_NEAREST)
    visible=cv2.remap(depth.astype(np.float32),px,py,cv2.INTER_LINEAR)
    alpha[(~covered)|(optical<=0)|(visible<=0)|(np.abs(visible-optical)>np.maximum(20,optical*.02))]=0
    write_image(directory/'projection.png',np.dstack([colors,alpha]))
    contours,_=cv2.findContours(alpha,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    footprint=[]
    if contours:
        boundary=cv2.approxPolyDP(max(contours,key=cv2.contourArea),3,True).reshape(-1,2)[:512]
        for bx,by in boundary:
            footprint.append([float(lon[by,bx]),float(lat[by,bx])])
    corners=map_corners(surface,x0,y0,x1,y1)
    return {'image':'projection.png','corners':corners},footprint


def reconstruction(surface,state,entry,root,directory,evidence):
    if digest(root/entry['valid_mask'])!=entry['mask_sha256']:
        raise ValueError('projection mask changed')
    image=cv2.imread(str(root/entry['image']));height,width=image.shape[:2]
    size=tuple(int(v) for v in state['size'])
    small=cv2.resize(image,size,interpolation=cv2.INTER_AREA)
    mask=cv2.resize(cv2.imread(str(root/entry['valid_mask']),0),size,interpolation=cv2.INTER_NEAREST)
    projection,footprint=ground_overlay(surface,state,small,mask,directory)
    eye,rotation,K=state['eye'],state['rotation'],state['K']
    lat,lon,alt=surface.geodetic(eye);terrain,covered=surface.sample('dem',lat,lon)
    if not covered:
        raise ValueError('candidate camera has no terrain sample')
    full_focal=float(K[0,0]*width/size[0])
    frustum=[]
    for x,y in [(0,0),(size[0]-1,0),(size[0]-1,size[1]-1),(0,size[1]-1)]:
        ray=rotation@np.array([(x-K[0,2])/K[0,0],(K[1,2]-y)/K[1,1],-1])
        la,lo,al=surface.geodetic(eye+ray*1500)
        frustum.append([float(lo),float(la),float(al)])
    return {'camera':{'size':[width,height],'focal':[full_focal,full_focal],
                      'principal':[(width-1)/2,(height-1)/2]},
            'review':{'camera_agl_m':float(alt-terrain),'evidence_sha256':evidence},
            'view':view(eye,rotation,surface),'overview':overview(surface,[(lat,lon,alt)]),
            'reference_image':'reference.png','projection':projection,'footprint':footprint,'frustum':frustum}


def cached_reconstruction(directory,key,build):
    path=directory/'projection-cache.json'
    if path.exists():
        try:
            saved=json.loads(path.read_text())
            if (saved['key']==key and set(saved['files'])=={'reference.png','projection.png'}
                    and all((directory/name).is_file() and digest(directory/name)==value
                    for name,value in saved['files'].items())):
                return saved['reconstruction']
        except (ValueError,KeyError,TypeError):
            pass
    value=build()
    files={name:digest(directory/name) for name in ['reference.png','projection.png']}
    write_json(path,{'key':key,'files':files,'reconstruction':value})
    return value


def research_inputs(runs):
    """Read explicit paired-view proposals without granting geometric acceptance."""
    proposals={}
    for run in runs:
        configuration=json.loads((run/'run.json').read_text())
        for path in sorted(run.glob('q-*.json')):
            result=json.loads(path.read_text())
            if result.get('status')!='research_proposal' or result.get('geometry_accepted') is not False:
                continue
            state=(run/result['camera']).resolve()
            if state.parent!=run.resolve() or digest(state)!=result['camera_sha256']:
                raise ValueError(f'research camera changed or escaped its run: {path}')
            if digest(run/'report.json')!=result['report_sha256']:
                raise ValueError(f'research report changed: {path}')
            report=json.loads((run/'report.json').read_text())
            if (report.get('status')!='research_proposal' or report.get('geometry_accepted') is not False
                    or result['id'] not in report.get('cases',[])
                    or report.get('catalog_sha256')!=configuration['catalog_sha256']):
                raise ValueError(f'research report does not support this photo: {path}')
            proposals.setdefault(result['id'],(run,result,path,configuration['catalog_sha256'],state))
    return proposals


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset',type=Path);parser.add_argument('output',type=Path)
    parser.add_argument('--manifest',type=Path,help='frozen manifest supplying flight groups and mask versions')
    parser.add_argument('--catalog',type=Path,action='append',required=True)
    parser.add_argument('--run',action='append',type=Path,required=True)
    parser.add_argument('--include-research',action='store_true',
                        help='show labelled, unverified paired-view proposals in a local review')
    args=parser.parse_args();cv2.setNumThreads(4)
    root=args.dataset.resolve();output=args.output.resolve()
    if output==root:
        parser.error('preview output must differ from the dataset root')
    output.mkdir(parents=True,exist_ok=True)
    projection_code={name:digest(Path(__file__).with_name(name)) for name in ['photo_preview.py','photo_surface.py']}
    projection_code.update(opencv=cv2.__version__,numpy=np.__version__)
    manifest=json.loads((args.manifest or root/'manifest.json').read_text())
    catalog_paths={digest(path):path for path in args.catalog}
    surfaces={key:Surface(path) for key,path in catalog_paths.items()}
    surface=surfaces[next(iter(surfaces))]
    candidates={}
    proposals=research_inputs(args.run) if args.include_research else {}
    for run in args.run:
        catalog_hash=json.loads((run/'run.json').read_text())['catalog_sha256']
        if catalog_hash not in surfaces:
            raise ValueError(f'candidate frame belongs to another reference catalog: {run}')
        for path in sorted(run.glob('q-*.json')):
            result=json.loads(path.read_text())
            if result['status']!='candidate_requires_review' or not result['candidate']['score']['geometry_accepted']:
                continue
            merit=candidate_priority(result['candidate'])
            previous=candidates.get(result['id'])
            if previous is None or merit>previous[0]:
                candidates[result['id']]=(merit,run,result,path,catalog_hash)
    basemaps={}
    for key,model in surfaces.items():
        grid=model.grids['imagery'][0];h,w=grid.shape[:2];scale=min(1,4096/max(w,h))
        filename='basemap-'+key[:16]+'.png'
        write_image(output/filename,cv2.resize(grid,(round(w*scale),round(h*scale))))
        basemaps[key]=(filename,{'image':'basemap.png','corners':map_corners(model,0,0,w,h)})
    entries={e['id']:e for e in manifest['photos']};units=[];count=0;research_count=0
    for flight in manifest['flights']:
        media=[];documents=[];locations=[]
        ordered=sorted(flight['photos'],key=lambda key:(entries[key]['claims']['capture_time'] or '',key))
        for index,identifier in enumerate(ordered):
            entry=entries[identifier];directory=output/identifier;directory.mkdir(exist_ok=True)
            if digest(root/entry['image'])!=entry['image_sha256']:
                raise ValueError('testset image changed')
            publish_asset(directory/'photo.png',lambda staged:shutil.copyfile(root/entry['image'],staged))
            doc={'version':3,'name':Path(entry['source']).stem,'duration_s':9,'fps':1,'agl_m':None,
                 'origin':None,'mosaic':None,'registration':'Photo has no accepted reconstruction.',
                 'frames':[{'time_s':0,'image':'photo.png','corners':None,'position':None,'inliers':0,'rms_px':None,'navigation':None}]}
            if identifier in candidates:
                _,run,result,path,catalog_hash=candidates[identifier];state_path=run/identifier/'candidate.npz'
                model=surfaces[catalog_hash]
                evidence=hashlib.sha256(''.join([digest(path),digest(state_path),entry['image_sha256'],
                    entry['mask_sha256'],catalog_hash]).encode()).hexdigest()
                if digest(root/entry['valid_mask'])!=entry['mask_sha256']:
                    raise ValueError('projection mask changed')
                with np.load(state_path) as state:
                    doc['reconstruction']=cached_reconstruction(directory,
                        {'evidence':evidence,'code':projection_code},
                        lambda:reconstruction(model,state,entry,root,directory,evidence))
                filename,basemap=basemaps[catalog_hash]
                doc['reconstruction']['basemap']=basemap
                link=directory/'basemap.png'
                if not link.exists() or not link.samefile(output/filename):
                    publish_asset(link,lambda staged:staged.hardlink_to(output/filename))
                point=result['candidate'];locations.append([point['latitude'],point['longitude'],point['altitude_m']])
                doc['registration']='Geometric candidate for visual review. Capture claim: '+str(entry['claims']['capture_time'])+'. Imagery: Esri. Terrain: Mapzen / AWS.'
                doc['frames'][0].update({'inliers':point['score']['inliers'],'rms_px':point['score']['train_rms_px']})
                write_json(directory/'evidence.json',{'result':result,'run_sha256':digest(run/'run.json'),'candidate_sha256':digest(state_path),
                    'catalog_sha256':catalog_hash,'image_sha256':entry['image_sha256'],'projection_mask_sha256':entry['mask_sha256']})
                count+=1
            elif identifier in proposals:
                run,result,path,catalog_hash,state_path=proposals[identifier]
                model=surfaces[catalog_hash]
                if digest(root/entry['valid_mask'])!=entry['mask_sha256']:
                    raise ValueError('projection mask changed')
                evidence={'result':result,'run_sha256':digest(run/'run.json'),
                    'camera_sha256':digest(state_path),'catalog_sha256':catalog_hash,
                    'image_sha256':entry['image_sha256'],'projection_mask_sha256':entry['mask_sha256']}
                evidence_key=hashlib.sha256(json.dumps(evidence,sort_keys=True).encode()).hexdigest()
                with np.load(state_path) as state:
                    doc['reconstruction']=cached_reconstruction(directory,
                        {'evidence':evidence_key,'code':projection_code},
                        lambda:reconstruction(model,state,entry,root,directory,evidence_key))
                    locations.append(list(map(float,model.geodetic(state['eye']))))
                filename,basemap=basemaps[catalog_hash]
                doc['reconstruction']['basemap']=basemap
                publish_asset(directory/'basemap.png',lambda staged:staged.hardlink_to(output/filename))
                doc['registration']='Research proposal: absolute pose requires map verification. Imagery: Esri. Terrain: Mapzen / AWS.'
                doc['name']+=' · research'
                write_json(directory/'evidence.json',evidence)
                research_count+=1
            documents.append((directory/'replay.json',doc))
            media.append({'id':identifier,'label':doc['name'],'replay':identifier+'/replay.json','start_s':index*9})
        if locations:
            wide=overview(surface,locations)
            for _,doc in documents:
                if 'reconstruction' in doc:
                    doc['reconstruction']['route_view']=wide
        for path,doc in documents:
            write_json(path,doc)
        timestamp=entries[ordered[0]]['claims']['capture_time']
        date=datetime.fromisoformat(timestamp).date().isoformat() if timestamp else 'unknown-date'
        units.append({'id':flight['id']+' '+date,'sidc':'10040100001201000003','media':media})
    name='Photo terrain testset'+(' · research preview' if research_count else '')
    write_json(output/'debrief.json',{'format':'sokoly-debrief','version':1,'name':name,'units':units})
    print(json.dumps({'photos':len(entries),'flights':len(units),'geometric_candidates':count,
                      'research_proposals':research_count,'output':str(output)}))


if __name__=='__main__':
    main()
