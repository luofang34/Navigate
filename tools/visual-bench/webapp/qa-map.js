import {MapView,localPosition} from './map.js';
import {download,sha256} from './storage.js';
import {viewCoverage} from './dynamic-coverage.js';
const report={},canvas=document.getElementById('map'),status=document.getElementById('result');
const check=(name,value)=>{if(!value)throw Error(name);report[name]=true;status.textContent=JSON.stringify(report,null,2)};
const request=async(url,value)=>{const r=await fetch(url,value?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(value)}:{});if(!r.ok)throw Error(await r.text());return r.json()};
try {
  const catalog=await request('/api/catalog'),packages=[];for(const entry of catalog.filter(r=>r.id.startsWith('naip-')))packages.push(await request('/api/offline-plan',{region_id:entry.id}));
  const pack=packages.find(p=>Math.max(...p.tiles.filter(t=>t.imagery).map(t=>t.xyz[0]))===16);if(!pack)throw Error('Prepare zoom-16 coverage first');await download(pack,()=>{});
  const map=new MapView(canvas);await map.load(pack,{width:960,height:544,fx:690,fy:690,cx:479.5,cy:271.5});
  await map.setPose({position_enu_m:localPosition(pack,40.543925,-74.457595,116),eye_to_enu_xyzw:[.014,.045,.831,.554].map((v,_,a)=>v/Math.hypot(...a))});
  const base=JSON.stringify(map.base),before=await sha256(new Uint8Array(await map.preview.render(JSON.stringify(map.pose))));
  check('default globe and physical display resolution',canvas.dataset.projection==='globe'&&canvas.width>=1100);
  check('real loaded terrain sampled',JSON.parse(canvas.dataset.clearance).known);
  await map.move(2,-100000);let clearance=JSON.parse(canvas.dataset.clearance);check('large downward move stops above terrain',clearance.known&&clearance.agl>=19.999);
  await map.move(0,80);clearance=JSON.parse(canvas.dataset.clearance);check('lateral movement retains terrain clearance',clearance.known&&clearance.agl>=19.999);
  await map.reset();check('clearance does not rewrite selected pose',JSON.stringify(map.base)===base&&JSON.stringify(map.pose)===base);
  const detail=packages.find(p=>p.tiles.some(t=>t.imagery&&t.xyz[0]>=17));
  if(!detail)throw Error('Prepare zoom-17 or finer coverage in the Rust service first');
  await download(detail,(done,total)=>status.textContent='Detailed imagery '+Math.round(done/total*100)+'%',{activate:false});
  await map.addPack(detail);const after=await sha256(new Uint8Array(await map.preview.render(JSON.stringify(map.pose))));
  check('high-resolution data changes rendered imagery',before!==after);
  check('detail upload preserves selected pose',JSON.stringify(map.pose)===base&&map.pack_id===pack.pack_id);
  report.detail={pack_id:detail.pack_id,imagery_zoom:Math.max(...detail.tiles.filter(t=>t.imagery).map(t=>t.xyz[0])),bytes:detail.files.reduce((n,f)=>n+f.size,0),render_sha256:after};
  check('low altitude plans finer coverage',viewCoverage(pack,map.pose,map.height()).zoom===18);
  await map.draw();document.title='PASS map detail and clearance';
}catch(error){report.error=String(error);document.title='FAIL map detail and clearance'}
status.textContent=JSON.stringify(report,null,2);await fetch('/__test__/report',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(report)});
