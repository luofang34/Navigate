import {cp,mkdir,readFile,readdir,writeFile,stat} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {resolve,dirname,relative} from 'node:path';
import {fileURLToPath} from 'node:url';
const webapp=resolve(dirname(fileURLToPath(import.meta.url)),'webapp');
const runtimeNames=new Set(['ort.webgpu.min.mjs','ort-wasm-simd-threaded.jsep.mjs','ort-wasm-simd-threaded.jsep.wasm','ort-wasm-simd-threaded.asyncify.mjs','ort-wasm-simd-threaded.asyncify.wasm','LICENSE','package-integrity.json']);
const digest=bytes=>createHash('sha256').update(bytes).digest('hex');
const hashPattern=/^[a-f0-9]{64}$/;
export async function exportSite(state,output,regions,source=webapp){
  if(!regions.length||regions.some(id=>!/^[-a-z0-9]+$/.test(id)))throw Error('Select explicit region IDs for public export');
  const destination=resolve(output);source=resolve(source);state=resolve(state);
  if(destination===source||destination.startsWith(source+'/')||destination===state||destination.startsWith(state+'/'))throw Error('Export outside source directories');
  await mkdir(destination,{recursive:true});if((await readdir(destination)).length)throw Error('Export requires an empty output directory');
  const models=JSON.parse(await readFile(resolve(source,'models/manifest.json'),'utf8'));
  if(!models.xfeat||Object.keys(models).some(name=>!['xfeat','lighterglue'].includes(name)))throw Error('Public export requires redistributable matcher models');
  const modelNames=new Set(['manifest.json','provenance.json']);
  for(const [name,entry] of Object.entries(models)){
    if(entry.url!==`/models/${name}.onnx`)throw Error('Invalid public model URL');
    const model=await readFile(resolve(source,`models/${name}.onnx`));
    if(digest(model)!==entry.sha256||model.length!==entry.size)throw Error(`Model checksum mismatch: ${name}`);
    modelNames.add(`${name}.onnx`);
  }
  const catalog=[],chunks=new Map(),packs=[];
  for(const id of [...new Set(regions)]){
    const item=JSON.parse(await readFile(resolve(state,`catalog/${id}.json`),'utf8'));
    if(item.id!==id||!hashPattern.test(item.pack_id))throw Error('Invalid region identity');
    const manifest=JSON.parse(await readFile(resolve(state,`packs/${item.pack_id}.json`),'utf8'));
    if(manifest.pack_id!==item.pack_id||manifest.region_id!==id)throw Error('Package identity mismatch');
    if(manifest.provenance?.provider!=='planetary-computer-naip')throw Error('Public export only supports the NAIP provider package');
    for(const file of manifest.files){
      if(!hashPattern.test(file.sha256)||file.url!==`/chunks/${file.sha256}.bin`||!Number.isSafeInteger(file.size)||file.size<=0)throw Error('Invalid public chunk record');
      chunks.set(file.sha256,file);
    }
    catalog.push({id:item.id,label:item.label,pack_id:item.pack_id,anchor_lat_lon:item.anchor_lat_lon,bounds:item.bounds,bytes:item.bytes,thumbnail:`/catalog/${id}.jpg`});
    packs.push(manifest);
  }
  for(const file of chunks.values()){
    const path=resolve(state,`chunks/${file.sha256}.bin`),bytes=await readFile(path);
    if(bytes.length!==file.size||digest(bytes)!==file.sha256)throw Error(`Chunk checksum mismatch: ${file.sha256}`);
  }
  await cp(source,destination,{recursive:true,filter:path=>{
    const parts=relative(source,path).split('/');if(!parts[0])return true;
    if(parts.some(part=>part.startsWith('.')||part.startsWith('qa')))return false;
    if(parts[0]==='models'&&parts.length>1)return parts.length===2&&modelNames.has(parts[1]);
    if(parts[0]==='runtime'&&parts.length>1)return parts.length===2&&runtimeNames.has(parts[1]);
    return true;
  }});
  for(const name of ['chunks','packs','catalog'])await mkdir(resolve(destination,name),{recursive:true});
  for(const file of chunks.values())await cp(resolve(state,`chunks/${file.sha256}.bin`),resolve(destination,`chunks/${file.sha256}.bin`));
  for(const manifest of packs)await writeFile(resolve(destination,`packs/${manifest.pack_id}.json`),JSON.stringify(manifest));
  for(const item of catalog)await cp(resolve(state,`catalog/${item.id}.jpg`),resolve(destination,`catalog/${item.id}.jpg`));
  await writeFile(resolve(destination,'catalog.json'),JSON.stringify(catalog));
  await writeFile(resolve(destination,'.nojekyll'),'');
  return {regions:catalog.length,chunks:chunks.size,package_bytes:[...chunks.values()].reduce((sum,f)=>sum+f.size,0)};
}
if(process.argv[1]&&resolve(process.argv[1])===fileURLToPath(import.meta.url)){
  const [state,output,...regions]=process.argv.slice(2);
  if(!state||!output)throw Error('Usage: node export_web.mjs package-state site-output region-id [region-id...]');
  console.info(JSON.stringify(await exportSite(state,output,regions)));
}
