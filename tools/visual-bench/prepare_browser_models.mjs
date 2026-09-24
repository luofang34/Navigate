import {execFileSync} from 'node:child_process';
import {readFileSync,writeFileSync,mkdirSync,mkdtempSync,rmSync,existsSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {dirname,resolve} from 'node:path';
import {tmpdir} from 'node:os';
import {fileURLToPath} from 'node:url';
const root=resolve(dirname(fileURLToPath(import.meta.url)),'webapp'),runtime=resolve(root,'runtime'),models=resolve(root,'models');
if(process.argv.length>2)throw Error('Usage: node prepare_browser_models.mjs');
mkdirSync(runtime,{recursive:true});mkdirSync(models,{recursive:true});
const temporary=mkdtempSync(resolve(tmpdir(),'navigate-runtime-'));
try {
  const [metadata]=JSON.parse(execFileSync('npm',['pack','onnxruntime-web@1.30.0','--ignore-scripts','--json','--pack-destination',temporary],{encoding:'utf8'}));
  const archive=resolve(temporary,metadata.filename);
  for(const name of ['ort.webgpu.min.mjs','ort-wasm-simd-threaded.jsep.mjs','ort-wasm-simd-threaded.jsep.wasm','ort-wasm-simd-threaded.asyncify.mjs','ort-wasm-simd-threaded.asyncify.wasm']) {
    writeFileSync(resolve(runtime,name),execFileSync('tar',['-xOf',archive,'package/dist/'+name],{maxBuffer:64*1024*1024}));
  }
  writeFileSync(resolve(runtime,'LICENSE'),readFileSync(resolve(root,'assets/ONNX-Runtime-LICENSE.txt')));
  writeFileSync(resolve(runtime,'package-integrity.json'),JSON.stringify(metadata,null,2));
} finally {rmSync(temporary,{recursive:true,force:true});}
const sha256='41336466bbbb09b701b815e33932c7da69cff6ab75a068230418468430482912',path=resolve(models,'xfeat.onnx');
const source='https://github.com/DavideCatto/XFeat-ONNX/releases/download/V1.0.0/xfeat.onnx';
let bytes=existsSync(path)?readFileSync(path):null;
if(!bytes||createHash('sha256').update(bytes).digest('hex')!==sha256){const response=await fetch(source);if(!response.ok)throw Error(`Model download failed: ${response.status}`);bytes=Buffer.from(await response.arrayBuffer())}
if(bytes.length!==2681450||createHash('sha256').update(bytes).digest('hex')!==sha256)throw Error('XFeat release checksum mismatch');
writeFileSync(path,bytes);
const gluePath=resolve(root,'../model-assets/lighterglue.onnx'),glue=readFileSync(gluePath),glueProvenance=JSON.parse(readFileSync(resolve(root,'../model-assets/lighterglue.json'),'utf8'));
if(glue.length!==glueProvenance.size||createHash('sha256').update(glue).digest('hex')!==glueProvenance.sha256)throw Error('LighterGlue asset checksum mismatch');
writeFileSync(resolve(models,'lighterglue.onnx'),glue);
const denseProvenance=JSON.parse(readFileSync(resolve(root,'../model-assets/loftr.json'),'utf8')),densePath=resolve(models,'loftr.onnx');
let dense=existsSync(densePath)?readFileSync(densePath):null;
if(!dense||createHash('sha256').update(dense).digest('hex')!==denseProvenance.sha256){const response=await fetch(denseProvenance.url);if(!response.ok)throw Error(`LoFTR asset download failed: ${response.status}`);dense=Buffer.from(await response.arrayBuffer())}
if(dense.length!==denseProvenance.size||createHash('sha256').update(dense).digest('hex')!==denseProvenance.sha256)throw Error('LoFTR asset checksum mismatch');
writeFileSync(densePath,dense);
writeFileSync(resolve(models,'manifest.json'),JSON.stringify({xfeat:{url:'/models/xfeat.onnx',size:bytes.length,sha256},lighterglue:{url:'/models/lighterglue.onnx',size:glue.length,sha256:glueProvenance.sha256},loftr:{url:'/models/loftr.onnx',size:dense.length,sha256:denseProvenance.sha256}},null,2));
writeFileSync(resolve(models,'provenance.json'),JSON.stringify({xfeat:{algorithm:'XFeat sparse features',license:'Apache-2.0',source,source_commit:'bc1acfa02489efc491d6f5891d07bea87f29ec19',sha256,modified:false},lighterglue:glueProvenance,loftr:denseProvenance},null,2));
