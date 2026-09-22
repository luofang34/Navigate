import {read,sha256} from './storage.js';
import {makeCanvas} from './observation.js';
import {localPosition} from './geography.js';
import {cropPlan} from './crop-plan.js';
export class ReferencePack {
  static async open(pack){const self=new ReferencePack();self.pack=pack;self.tiles=[];
    for(const tile of pack.tiles){const item={xyz:tile.xyz};for(const role of ['imagery','elevation']){const a=tile[role];if(!a)continue;const bytes=await read(`pilotage://chunks/${a.chunk}.bin`,a.offset,a.length);if(await sha256(bytes)!==a.sha256)throw Error('Reference tile checksum failed');const bitmap=await createImageBitmap(new Blob([bytes]));const c=makeCanvas();c.width=bitmap.width;c.height=bitmap.height;const ctx=c.getContext('2d',{willReadFrequently:true});ctx.drawImage(bitmap,0,0);bitmap.close();item[role]={canvas:role==='imagery'?c:undefined,pixels:role==='elevation'?ctx.getImageData(0,0,c.width,c.height).data:undefined}}
      self.tiles.push(item);
    }self.dem=self.tiles.filter(t=>t.elevation).sort((a,b)=>b.xyz[0]-a.xyz[0]);return self;
  }
  coordinate(z,x,y){return [Math.atan(Math.sinh(Math.PI*(1-2*y/2**z)))*180/Math.PI,x/2**z*360-180]}
  elevation(lat,lon){for(const t of this.dem){const [z,x,y]=t.xyz,n=2**z,px=((lon+180)/360*n-x)*256,py=((1-Math.asinh(Math.tan(lat*Math.PI/180))/Math.PI)/2*n-y)*256;if(px<0||py<0||px>=256||py>=256)continue;const i=(Math.floor(py)*256+Math.floor(px))*4,p=t.elevation.pixels;if(p[i+3]!==255)continue;return p[i]*256+p[i+1]+p[i+2]/256-32768}throw Error('The selected prior has no valid terrain elevation')}
  world(z,x,y){const [lat,lon]=this.coordinate(z,x,y);return localPosition(this.pack,lat,lon,this.elevation(lat,lon))}
  crops(prior,camera,scales=[1]){const key=JSON.stringify([prior,camera,scales]);if(this.cropCache?.key===key)return this.cropCache.value;const value=this.buildCrops(prior,camera,scales);this.cropCache={key,value};return value}
  buildCrops(prior,camera,scales){const z=Math.max(...this.tiles.filter(t=>t.imagery).map(t=>t.xyz[0]));const tiles=this.tiles.filter(t=>t.imagery&&t.xyz[0]===z);const x0=Math.min(...tiles.map(t=>t.xyz[1])),y0=Math.min(...tiles.map(t=>t.xyz[2])),x1=Math.max(...tiles.map(t=>t.xyz[1]))+1,y1=Math.max(...tiles.map(t=>t.xyz[2]))+1;
    const width=(x1-x0)*512,height=(y1-y0)*512;
    const scale=2*Math.PI*6371008.8*Math.cos(this.pack.anchor_lat_lon[0]*Math.PI/180),mpp=scale/(2**z*512),baseSize=Math.max(256,Math.round(1.5*prior.agl_m*Math.max(camera.width/camera.fx,camera.height/camera.fy)/mpp));
    const center=[((prior.longitude+180)/360*2**z-x0)*512,((1-Math.asinh(Math.tan(prior.latitude*Math.PI/180))/Math.PI)/2*2**z-y0)*512];
    const result=[];
    for(const {x,y,size} of cropPlan({width,height,baseSize,center,metresPerPixel:mpp,radius:prior.radius_m,scales})){
      const crop=makeCanvas();crop.width=crop.height=640;const cc=crop.getContext('2d',{willReadFrequently:true});for(const t of tiles){const tx=(t.xyz[1]-x0)*512,ty=(t.xyz[2]-y0)*512;if(tx+512<=x||ty+512<=y||tx>=x+size||ty>=y+size)continue;cc.drawImage(t.imagery.canvas,(tx-x)*640/size,(ty-y)*640/size,512*640/size,512*640/size);}const data=cc.getImageData(0,0,640,640).data;let valid=0;for(let i=3;i<data.length;i+=4)if(data[i]===255)valid++;if(valid<640*640*.35)continue;
      const alpha=new Uint8Array(640*640),pixels=new Uint8Array(640*640);for(let i=0;i<alpha.length;i++){alpha[i]=data[i*4+3];pixels[i]=(77*data[i*4]+150*data[i*4+1]+29*data[i*4+2])>>>8}
      result.push({key:`${this.pack.pack_id}/${x}/${y}/${size}`,image:{gray:pixels,valid:alpha,width:640,height:640},world:p=>{const mx=(x+(p[0]+.5)*size/640-.5)/512+x0,my=(y+(p[1]+.5)*size/640-.5)/512+y0;const ix=Math.max(0,Math.min(639,Math.round(p[0]))),iy=Math.max(0,Math.min(639,Math.round(p[1])));if(alpha[iy*640+ix]!==255)return null;try{return this.world(z,mx,my)}catch{return null}}});
    }
    if(!result.length)throw Error('No valid reference imagery overlaps this prior');if(result.length>160)throw Error('Prior requires too many browser search crops. Reduce the search radius or split the area.');return result;
  }
}
