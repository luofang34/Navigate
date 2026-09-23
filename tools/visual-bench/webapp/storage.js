import {assetUrl} from './asset-url.js';
const DB='navigate-visual-offline-v1';
function database(){return new Promise((resolve,reject)=>{const r=indexedDB.open(DB,1);r.onupgradeneeded=()=>{for(const name of ['packs','state','missions'])r.result.createObjectStore(name)};r.onsuccess=()=>resolve(r.result);r.onerror=()=>reject(r.error)})}
export async function get(store,key){const db=await database();try{return await new Promise((resolve,reject)=>{const r=db.transaction(store).objectStore(store).get(key);r.onsuccess=()=>resolve(r.result);r.onerror=()=>reject(r.error)})}finally{db.close()}}
export async function put(store,key,value){const db=await database();try{await new Promise((resolve,reject)=>{const tx=db.transaction(store,'readwrite');tx.objectStore(store).put(value,key);tx.oncomplete=resolve;tx.onerror=()=>reject(tx.error);tx.onabort=()=>reject(tx.error)})}finally{db.close()}}
export async function all(store){const db=await database();try{return await new Promise((resolve,reject)=>{const r=db.transaction(store).objectStore(store).getAll();r.onsuccess=()=>resolve(r.result);r.onerror=()=>reject(r.error)})}finally{db.close()}}
async function directory(path,create=false){let dir=await navigator.storage.getDirectory();for(const part of path.split('/'))dir=await dir.getDirectoryHandle(part,{create});return dir}
async function fileAt(path){const parts=path.split('/');const name=parts.pop();return (await directory(parts.join('/'))).getFileHandle(name)}
export async function read(uri,offset,length){if(!/^pilotage:\/\/chunks\/[a-f0-9]{64}\.bin$/.test(uri))throw Error('Invalid data URI');const file=await(await fileAt(uri.replace('pilotage://','pilotage/'))).getFile();if(offset+length>file.size)throw Error('OPFS range exceeds file');return file.slice(offset,offset+length).arrayBuffer()}
export async function sha256(bytes){return [...new Uint8Array(await crypto.subtle.digest('SHA-256',bytes))].map(b=>b.toString(16).padStart(2,'0')).join('')}
export async function verifyChunk(chunk){try{const f=await(await fileAt(`pilotage/chunks/${chunk.sha256}.bin`)).getFile();return f.size===chunk.size&&await sha256(await f.arrayBuffer())===chunk.sha256}catch(error){if(error.name==='NotFoundError')return false;throw error}}
export async function verifyPack(pack,progress=()=>{}){let count=0;for(const c of pack.files){if(!await verifyChunk(c))return false;progress(++count/pack.files.length)}return true}
export async function downloadFiles(files,progress){
  const dir=await directory('pilotage/chunks',true);let completed=0;const total=files.reduce((sum,c)=>sum+c.size,0);
  const missing=[];for(const c of files){if(await verifyChunk(c))completed+=c.size;else missing.push(c)}
  const quota=await navigator.storage.estimate();const need=missing.reduce((sum,c)=>sum+c.size,0)+8*1024*1024;
  if(quota.quota!==undefined&&quota.usage!==undefined&&quota.quota-quota.usage<need)throw Error('Insufficient browser storage for this package');
  progress(completed,total);
  for(const chunk of missing){
    const temp=await dir.getFileHandle(chunk.sha256+'.partial',{create:true});const writer=await temp.createWritable();
    try{const response=await fetch(assetUrl(chunk.url));if(!response.ok||!response.body)throw Error(`Download failed: ${response.status}`);
      const reader=response.body.getReader();let size=0;
      while(true){const {value,done}=await reader.read();if(done)break;size+=value.length;if(size>chunk.size){await reader.cancel();throw Error('Chunk exceeds declared size')}await writer.write(value);progress(completed+size,total)}
      await writer.close();const file=await temp.getFile();
      if(file.size!==chunk.size||await sha256(await file.arrayBuffer())!==chunk.sha256)throw Error('Chunk checksum failed');
      const target=await dir.getFileHandle(chunk.sha256+'.bin',{create:true});const output=await target.createWritable();
      try{await file.stream().pipeTo(output)}catch(error){await output.abort().catch(()=>{});throw error}
      await dir.removeEntry(chunk.sha256+'.partial');completed+=chunk.size;progress(completed,total);
    }catch(error){await writer.abort().catch(()=>{});await dir.removeEntry(chunk.sha256+'.partial').catch(()=>{});throw error}
  }
}
export async function download(pack,progress,{activate=true}={}){
  await downloadFiles(pack.files,progress);
  await put('packs',pack.pack_id,{...pack,saved_at:Date.now()});if(activate)await put('state','active-pack',pack.pack_id);
}
export async function saveMission(job,pack){
  const dir=await directory(`pilotage/missions/${job.id}`,true);
  for(const frame of job.view.frames){const response=await fetch(assetUrl(`/jobs/${job.id}/result/${frame.query}`));if(!response.ok)throw Error('Cannot save observation frame');const file=await dir.getFileHandle(frame.query,{create:true});await response.body.pipeTo(await file.createWritable())}
  const mission={id:job.id,pack_id:pack.pack_id,view:job.view,saved_at:Date.now()};await put('missions',job.id,mission);return mission;
}
export async function queryBlob(mission,frame){return (await(await fileAt(`pilotage/missions/${mission.id}/${frame.query}`)).getFile())}

export async function saveLocalMission(view,pack,frames){
  const id=crypto.randomUUID().replaceAll('-',''),dir=await directory(`pilotage/missions/${id}`,true);
  for(let i=0;i<view.frames.length;i++){const name=`frame-${i}.png`;view.frames[i].query=name;const file=await dir.getFileHandle(name,{create:true});await frames[i].stream().pipeTo(await file.createWritable())}
  const mission={id,pack_id:pack.pack_id,view,saved_at:Date.now()};await put('missions',id,mission);return mission;
}
