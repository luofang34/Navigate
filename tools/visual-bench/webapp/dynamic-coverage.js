export function viewCoverage(pack,pose,height) {
  if(!pose||!pack||!Number.isFinite(height)||height>5000)return null;
  const r=6371008.8,[lat0,lon0]=pack.anchor_lat_lon.map(v=>v*Math.PI/180);
  const axes=[[Math.cos(lon0),0,-Math.sin(lon0)],[-Math.sin(lon0)*Math.sin(lat0),Math.cos(lat0),-Math.cos(lon0)*Math.sin(lat0)],[Math.sin(lon0)*Math.cos(lat0),Math.sin(lat0),Math.cos(lon0)*Math.cos(lat0)]];
  const local=[pose.position_enu_m[0],pose.position_enu_m[1],pose.position_enu_m[2]+r];
  const world=axes[0].map((_,i)=>axes.reduce((sum,a,j)=>sum+a[i]*local[j],0));
  const [x,y,z,w]=pose.eye_to_enu_xyzw,forward=[-2*(x*z+w*y),-2*(y*z-w*x),2*(x*x+y*y)-1];
  const ray=axes[0].map((_,i)=>axes.reduce((sum,a,j)=>sum+a[i]*forward[j],0));
  const b=world.reduce((sum,v,i)=>sum+v*ray[i],0),c=world.reduce((sum,v)=>sum+v*v,0)-r*r,disc=b*b-c;
  if(disc<0)return null;const distance=-b-Math.sqrt(disc);if(distance<0)return null;
  const hit=world.map((v,i)=>v+distance*ray[i]),latitude=Math.asin(hit[1]/r)*180/Math.PI,longitude=Math.atan2(hit[0],hit[2])*180/Math.PI;
  const radius=Math.max(100,Math.min(2000,height*1.2)),dy=radius/r*180/Math.PI,dx=dy/Math.cos(latitude*Math.PI/180);
  if(Math.abs(latitude)>75||Math.abs(longitude)+dx>=180)return null;
  return {bounds:[longitude-dx,latitude-dy,longitude+dx,latitude+dy],zoom:height<400?18:height<1600?17:height<3000?16:14};
}

export function covers(pack,selection) {
  const [w,s,e,n]=selection.bounds,z=selection.zoom,count=2**z;
  const position=(lon,lat)=>[(lon+180)/360*count,(1-Math.asinh(Math.tan(lat*Math.PI/180))/Math.PI)/2*count];
  const [x0,y0]=position(w,n),[x1,y1]=position(e,s);
  const imagery=pack.tiles.filter(t=>t.imagery);
  for(let y=Math.floor(y0);y<=Math.floor(y1);y++)for(let x=Math.floor(x0);x<=Math.floor(x1);x++) {
    // A complete finer tile set can cover a coarse display request.
    const level=Math.max(z,Math.min(...imagery.map(t=>t.xyz[0]))),scale=2**(level-z);
    for(let yy=y*scale;yy<(y+1)*scale;yy++)for(let xx=x*scale;xx<(x+1)*scale;xx++)if(!imagery.some(t=>t.xyz[0]===level&&t.xyz[1]===xx&&t.xyz[2]===yy))return false;
  }
  return imagery.length>0;
}

export class CoverageLoader {
  constructor({request,download,attach,installed,available,remember,status}){Object.assign(this,{request,download,attach,installed,available,remember,status});this.enabled=false;this.pending=null;this.failed=new Set()}
  async update(selection) {
    if(!selection||this.pending)return;
    const key=JSON.stringify(selection);if(this.failed.has(key))return;
    this.pending=key;
    try {
      const known=await this.installed();if(known.some(p=>covers(p,selection)))return;
      const cached=(await this.available?.()??[]).find(p=>covers(p,selection));
      if(cached){await this.download(cached,()=>{});await this.attach(cached);this.status('Viewed area loaded from offline storage.');return;}
      if(!this.enabled){this.status('Higher detail is not cached. Enable viewed-area downloads to fetch it.');return;}
      const plan=await this.request('/api/coverage-plan',selection);
      this.status(`Loading viewed area · ${plan.imagery_tiles.length} imagery tiles`);
      let job=await this.request('/api/coverage-download',selection);
      while(['queued','running'].includes(job.status)) {this.status(job.progress);await new Promise(r=>setTimeout(r,1000));job=await this.request('/api/downloads/'+job.id);}
      if(job.status!=='complete')throw Error(job.error||'Coverage download failed');
      const region=job.region;await this.download(region.manifest,(n,t)=>this.status(`Storing viewed area · ${(n/1048576).toFixed(1)} / ${(t/1048576).toFixed(1)} MB`));
      await this.remember(region);
      if(this.enabled)await this.attach(region.manifest);
      this.status('Viewed area stored. Select its region to use it for localization.');
    } catch(error) {this.failed.add(key);this.status(String(error));}
    finally {this.pending=null;}
  }
}
