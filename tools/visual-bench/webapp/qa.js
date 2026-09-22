import * as store from './storage.js';
import {MapView} from './map.js';
import {hypotheses} from './hypotheses.js';
const report=document.getElementById('report');const results=[];
function check(value,message){if(!value)throw Error(message);results.push({pass:message});report.textContent=JSON.stringify(results,null,2)}
async function run(){
  const bytes=new TextEncoder().encode('Navigate isolated storage fixture '+crypto.randomUUID());const sha=await store.sha256(bytes);
  const chunk={sha256:sha,size:bytes.length,url:'data:application/octet-stream;base64,'+btoa(String.fromCharCode(...bytes))};
  const pack={schema_version:1,pack_id:'qa-'+sha,files:[chunk]};const active=await store.get('state','active-pack');
  try{
    await store.download(pack,()=>{});check(await store.verifyChunk(chunk),'Streamed chunk is present and checksum-valid');
    const slice=await store.read(`pilotage://chunks/${sha}.bin`,3,7);check(new TextDecoder().decode(slice)===new TextDecoder().decode(bytes.slice(3,10)),'OPFS exact range read returns the requested bytes');
    let blocked=false;try{await store.read(`pilotage://chunks/${sha}.bin`,bytes.length,1)}catch{blocked=true}check(blocked,'Out-of-range OPFS read rejects');
    const originalFetch=window.fetch;let fetched=0;
    window.fetch=(...args)=>{fetched++;return originalFetch(...args)};
    try{await store.download(pack,()=>{})}finally{window.fetch=originalFetch}
    check(fetched===0,'A verified chunk is reused without downloading');
    const bad={...pack,pack_id:'bad-'+sha,files:[{...chunk,sha256:'0'.repeat(64)}]};blocked=false;
    try{await store.download(bad,()=>{})}catch{blocked=true}
    check(blocked&&!await store.get('packs',bad.pack_id),'Checksum failure does not commit a package');
    await store.put('state','active-pack',active);
    const id=new URLSearchParams(location.search).get('job');if(!/^[a-f0-9]{32}$/.test(id||''))throw Error('Supply a completed job ID');
    const job=await(await fetch('/api/jobs/'+id)).json();const observation=job.view.frames.find(f=>hypotheses(f).length);const frame=observation&&hypotheses(observation)[0];check(!!frame,'Local inference returned a geometrically accepted hypothesis');
    const packages=await store.all('packs');const saved=packages.find(p=>p.anchor_lat_lon&&JSON.stringify(p.anchor_lat_lon)===JSON.stringify(frame.anchor_lat_lon));
    check(saved&&await store.verifyPack(saved),'The camera preview package is complete in OPFS');
    const view=new MapView(document.getElementById('canvas'));await view.load(saved,job.view.camera);await view.setPose({position_enu_m:frame.position_enu_m,eye_to_enu_xyzw:frame.eye_to_enu_xyzw});
    const pixels=()=>view.canvas.getContext('2d').getImageData(0,0,view.canvas.width,view.canvas.height).data;
    const initial=await store.sha256(pixels());check(pixels().filter((v,i)=>i%4===3&&v>0).length>view.canvas.width*view.canvas.height*.5,'Rust/WASM renders nonempty terrain from the estimated camera');
    await view.move(1,20);check(await store.sha256(pixels())!==initial,'Moving the 3D camera changes rendered pixels');
    await view.reset();check(await store.sha256(pixels())===initial,'Reset restores the exact estimated camera view');
    const mission=await store.saveMission(job,saved);check((await store.queryBlob(mission,observation)).size>0,'Observation frames and pose are saved for offline replay');
    const corrupt=structuredClone(saved);corrupt.tiles.find(t=>t.imagery).imagery.sha256='0'.repeat(64);let denied=false;
    try{const broken=new MapView(document.createElement('canvas'));await broken.load(corrupt,job.view.camera)}catch{denied=true}
    check(denied,'Rust rejects a tile whose bytes do not match its manifest digest');
    results.push({complete:true,rgba_sha256:initial,pose:frame.position_enu_m});report.textContent=JSON.stringify(results,null,2);document.title='PASS · 12 browser checks';const summary=document.createElement('p');summary.textContent='PASS: storage, exact ranges, reuse, failed checksums, local pose, offline map, camera movement, camera reset, saved observation, and Rust tile validation.';document.body.prepend(summary);
  }finally{await store.put('state','active-pack',active)}
}
document.getElementById('run').onclick=()=>run().catch(e=>{results.push({error:String(e)});report.textContent=JSON.stringify(results,null,2)});
