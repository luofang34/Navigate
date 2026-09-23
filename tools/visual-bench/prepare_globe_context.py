"""Make a display-only world overview from public-domain Natural Earth land."""
import argparse,hashlib,json,math
from pathlib import Path
from PIL import Image,ImageDraw


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('geojson',type=Path)
    args=parser.parse_args()
    data=args.geojson.read_bytes(); source=json.loads(data)
    size=2048;image=Image.new('RGBA',(size,size),(30,57,73,255));draw=ImageDraw.Draw(image)
    def pixel(point):
        lon,lat=point[:2];lat=max(-85.05112878,min(85.05112878,lat))
        return ((lon+180)/360*size,(1-math.asinh(math.tan(math.radians(lat)))/math.pi)/2*size)
    for feature in source['features']:
        geometry=feature['geometry'];polygons=geometry['coordinates'] if geometry['type']=='MultiPolygon' else [geometry['coordinates']]
        for rings in polygons:
            for i,ring in enumerate(rings): draw.polygon([pixel(p) for p in ring],fill=(97,127,112,255) if i==0 else (30,57,73,255))
    for lon in range(-180,181,30):
        x=pixel([lon,0])[0];draw.line([(x,0),(x,size)],fill=(160,182,178,80),width=1)
    for lat in range(-60,61,30):
        y=pixel([0,lat])[1];draw.line([(0,y),(size,y)],fill=(160,182,178,80),width=1)
    image.putalpha(255)
    output=Path(__file__).resolve().parent/'webapp/context';output.mkdir(exist_ok=True)
    target=output/'earth.png';image.resize((512,512),Image.Resampling.LANCZOS).save(target)
    manifest=dict(attribution='Made with Natural Earth. Public domain. Display context only.',source='https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_land.geojson',source_sha256=hashlib.sha256(data).hexdigest(),sha256=hashlib.sha256(target.read_bytes()).hexdigest(),url='/context/earth.png')
    (output/'earth.json').write_text(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
