"""Build immutable imagery packages from public NAIP cloud-optimized rasters."""
import hashlib, json, math, urllib.request
from contextlib import ExitStack
from pathlib import Path
import numpy as np
from PIL import Image
from .coverage import tile_bounds
from .packages import encoded, digest, build_region

API='https://planetarycomputer.microsoft.com/api/stac/v1/search'
ATTRIBUTION='Imagery: USDA NAIP (public domain), hosted by Microsoft Planetary Computer. Terrain: Mapzen / AWS Open Data; source vertical datum is unverified.'


def fetch_json(url, value=None):
    body=None if value is None else encoded(value)
    request=urllib.request.Request(url,data=body,headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(request,timeout=60) as response:
        return json.load(response)


def search(bounds):
    result=fetch_json(API,dict(collections=['naip'],bbox=bounds,limit=100))
    items=result.get('features',[])
    if not items:
        raise ValueError('NAIP has no imagery for this selection. Choose supported US coverage.')
    if any(link.get('rel')=='next' for link in result.get('links',[])):
        raise ValueError('Provider search requires too many source records. Use a smaller region.')
    newest=max(int(item['properties']['naip:year']) for item in items)
    return [item for item in items if int(item['properties']['naip:year'])==newest]


def intersects(a,b):
    return a[0]<b[2] and a[2]>b[0] and a[1]<b[3] and a[3]>b[1]


def write_image(output, image):
    import io
    buffer=io.BytesIO(); image.save(buffer,format='PNG'); data=buffer.getvalue()
    sha=digest(data); name=sha+'.png'; (output/name).write_bytes(data)
    return dict(path=name,sha256=sha)


def raster_tile(items, sources, xyz):
    import rasterio
    from rasterio.vrt import WarpedVRT
    from rasterio.transform import from_bounds
    from rasterio.warp import transform_bounds
    bounds=tile_bounds(*xyz)
    transform=from_bounds(*transform_bounds('EPSG:4326','EPSG:3857',*bounds),512,512)
    rgba=np.zeros((512,512,4),dtype=np.uint8)
    for item,source in zip(items,sources):
        if not intersects(item['bbox'],bounds): continue
        with WarpedVRT(source,crs='EPSG:3857',transform=transform,width=512,height=512,
                       resampling=rasterio.enums.Resampling.bilinear,add_alpha=rasterio.enums.ColorInterp.alpha not in source.colorinterp) as vrt:
            rgb=vrt.read([1,2,3]); alpha=vrt.colorinterp.index(rasterio.enums.ColorInterp.alpha)+1
            valid=vrt.read(alpha)==255
            rgba[valid,:3]=np.moveaxis(rgb,0,-1)[valid]; rgba[valid,3]=255
    return Image.fromarray(rgba)


def build(plan, state, progress, known_items=None):
    import rasterio
    items=known_items if known_items is not None else search(plan['bounds'])
    source_identity=[dict(id=i['id'],datetime=i['properties'].get('datetime'),gsd=i['properties'].get('gsd'),url=i['assets']['image']['href']) for i in items]
    identity=dict(provider='planetary-computer-naip',sources=source_identity,selection=plan['requested'],imagery_tiles=plan['imagery_tiles'],processing='web-mercator-rgb-bilinear-512-opaque-mask/v2',terrain='mapzen-terrarium-z14')
    key=digest(encoded(identity)); output=state/'provider'/key; output.mkdir(parents=True,exist_ok=True)
    region_id='naip-'+key[:16]
    if (output/'map.json').is_file():
        return build_region(output,state,region_id,'NAIP · downloaded coverage')
    token=fetch_json('https://planetarycomputer.microsoft.com/api/sas/v1/token/naip')['token']
    tiles=[]
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.tif',GDAL_HTTP_MAX_RETRY=2,GDAL_HTTP_RETRY_DELAY=1),ExitStack() as stack:
        sources=[stack.enter_context(rasterio.open(item['assets']['image']['href']+'?'+token)) for item in items]
        for index,xyz in enumerate(plan['imagery_tiles']):
            image=raster_tile(items,sources,xyz)
            if image.getchannel('A').getextrema()[1]==0:
                continue
            tiles.append(dict(xyz=xyz,imagery=write_image(output,image)))
            progress(f'NAIP imagery {index+1}/{len(plan["imagery_tiles"])}')
    if not tiles: raise ValueError('The selected imagery contains no valid pixels')
    dem_tiles=sorted({(14,t['xyz'][1]>>(t['xyz'][0]-14),t['xyz'][2]>>(t['xyz'][0]-14)) for t in tiles})
    for index,xyz in enumerate(dem_tiles):
        z,x,y=xyz
        with urllib.request.urlopen(f'https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png',timeout=60) as response:
            image=Image.open(response).convert('RGBA'); image.load()
        if image.size!=(256,256): raise ValueError('Unexpected terrain tile dimensions')
        asset=write_image(output,image)
        same=next((t for t in tiles if t['xyz']==list(xyz)),None)
        if same is None: tiles.append(dict(xyz=list(xyz),elevation=asset))
        else: same['elevation']=asset
        progress(f'Terrain {index+1}/{len(dem_tiles)}')
    w,s,e,n=plan['bounds']
    manifest=dict(schema_version=2,release_id='naip-'+key,anchor_lat_lon=[(s+n)/2,(w+e)/2],elevation_datum='source_vertical_datum_unverified',attribution=ATTRIBUTION,tiles=tiles,provenance=identity)
    (output/'map.json').write_bytes(encoded(manifest))
    return build_region(output,state,region_id,'NAIP · downloaded coverage')
