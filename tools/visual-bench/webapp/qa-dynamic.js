import init,{Preview} from '/wasm/navigate_visual_preview.js';
import {MapView,localPosition} from './map.js';
import {CoverageLoader} from './dynamic-coverage.js';
import {download,read,get} from './storage.js';
const result={},check=(name,value)=>{if(!value)throw Error(name);result[name]=true};
try {
  const request=async(url,body)=>{const r=await fetch(url,body?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}:{});if(!r.ok)throw Error(await r.text());return r.json()};
  const catalog=await request('/api/catalog'),entry=catalog.find(r=>r.id.startsWith('naip-'));
  const first=await request('/api/offline-plan',{region_id:'new-jersey'}),second=await request('/api/offline-plan',{region_id:entry.id});
  await download(first,()=>{});const active=await get('state','active-pack');await download(second,()=>{},{activate:false});
  check('dynamic storage preserves active reference selection',await get('state','active-pack')===active);
  const camera={width:640,height:360,fx:459,fy:459,cx:319.5,cy:179.5},map=new MapView(document.getElementById('map'));
  await map.load(first,camera);await map.setPose({position_enu_m:localPosition(first,40.547,-74.45,800),eye_to_enu_xyzw:[0,0,0,1]});
  const pose=JSON.stringify(map.pose),anchor=JSON.stringify(map.pack.anchor_lat_lon);
  const loader=new CoverageLoader({request:async()=>{throw Error('cached coverage must not contact provider')},download:(p,cb)=>download(p,cb,{activate:false}),installed:async()=>[first],available:async()=>[second],attach:p=>map.addPack(p),remember:()=>{},status:()=>{}});
  loader.enabled=true;await loader.update({bounds:[-74.469,40.54,-74.46,40.55],zoom:16});
  check('cached coverage attaches to real wgpu renderer',map.displayPacks.size===2);
  check('coverage keeps camera pose',JSON.stringify(map.pose)===pose);
  check('coverage keeps observation reference identity',map.pack_id===first.pack_id&&JSON.stringify(map.pack.anchor_lat_lon)===anchor);
  await init();const reference=await Preview.create(JSON.stringify(first),JSON.stringify(camera),read,false);
  let rejected=false;try{await reference.add_display_package(JSON.stringify(second),read)}catch{rejected=true}finally{reference.free()}
  check('dynamic data cannot mutate a reference renderer',rejected);document.title='PASS dynamic coverage';
}catch(error){result.error=String(error);document.title='FAIL dynamic coverage';}
document.getElementById('result').textContent=JSON.stringify(result,null,2);
await fetch('/__test__/report',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(result)});
