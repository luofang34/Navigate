import {execFileSync} from 'node:child_process';
import {readFileSync,writeFileSync,mkdirSync,copyFileSync,mkdtempSync,rmSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {dirname,resolve} from 'node:path';
import {tmpdir} from 'node:os';
import {fileURLToPath} from 'node:url';
const source=process.argv[2];
if(!source)throw Error('Usage: node prepare_browser_models.mjs /path/to/browser-ready-onnx');
const root=resolve(dirname(fileURLToPath(import.meta.url)),'webapp'),runtime=resolve(root,'runtime'),models=resolve(root,'models');
mkdirSync(runtime,{recursive:true});mkdirSync(models,{recursive:true});
const temporary=mkdtempSync(resolve(tmpdir(),'navigate-runtime-'));
try {
  const [metadata]=JSON.parse(execFileSync('npm',['pack','onnxruntime-web@1.30.0','--ignore-scripts','--json','--pack-destination',temporary],{encoding:'utf8'}));
  const archive=resolve(temporary,metadata.filename);
  for(const name of ['ort.webgpu.min.mjs','ort-wasm-simd-threaded.jsep.mjs','ort-wasm-simd-threaded.jsep.wasm','ort-wasm-simd-threaded.asyncify.mjs','ort-wasm-simd-threaded.asyncify.wasm']) {
    writeFileSync(resolve(runtime,name),execFileSync('tar',['-xOf',archive,'package/dist/'+name],{maxBuffer:64*1024*1024}));
  }
  writeFileSync(resolve(runtime,'package-integrity.json'),JSON.stringify(metadata,null,2));
} finally {rmSync(temporary,{recursive:true,force:true});}
const manifest={};
for(const name of ['superpoint','superglue']) {
  const input=resolve(source,name+'.onnx'),target=resolve(models,name+'.onnx'),bytes=readFileSync(input);
  if(input!==target)copyFileSync(input,target);
  manifest[name]={url:'/models/'+name+'.onnx',size:bytes.length,sha256:createHash('sha256').update(bytes).digest('hex')};
}
writeFileSync(resolve(models,'manifest.json'),JSON.stringify(manifest,null,2));
