import * as ort from './runtime/ort.webgpu.min.mjs';
let session;
function pairs(data,n){const row=new Int32Array(n).fill(-1),col=new Int32Array(n).fill(-1),rv=new Float32Array(n).fill(-Infinity),cv=new Float32Array(n).fill(-Infinity);for(let i=0;i<n;i++)for(let j=0;j<n;j++){const x=data[i*n+j];if(x>rv[i]){rv[i]=x;row[i]=j}if(x>cv[j]){cv[j]=x;col[j]=i}}const result=[];for(let i=0;i<n;i++)if(row[i]>=0&&col[row[i]]===i&&rv[i]>Math.log(.1))result.push(`${i}:${row[i]}`);return result}
const report={scope:'matcher model portability; not geographic search or accuracy',cases:[]};
try{
 ort.env.wasm.numThreads=1;ort.env.wasm.wasmPaths=new URL('./runtime/',import.meta.url).href;
 const adapter=await navigator.gpu?.requestAdapter({powerPreference:'high-performance'});if(!adapter)throw Error('WebGPU unavailable');ort.env.webgpu.adapter=adapter;
 report.adapter={vendor:adapter.info.vendor,architecture:adapter.info.architecture};
 postMessage({stage:'Loading redistributable matcher'});
 const model=new Uint8Array(await(await fetch('./model.onnx')).arrayBuffer());report.model_sha256=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',model)),v=>v.toString(16).padStart(2,'0')).join('');
 session=await ort.InferenceSession.create(model,{executionProviders:['webgpu','wasm'],graphOptimizationLevel:'all'});
 let dispatches=0;const device=ort.env.webgpu.device,create=device.createCommandEncoder.bind(device);device.createCommandEncoder=(...args)=>{const encoder=create(...args),begin=encoder.beginComputePass.bind(encoder);encoder.beginComputePass=(...a)=>{const pass=begin(...a),dispatch=pass.dispatchWorkgroups.bind(pass);pass.dispatchWorkgroups=(...x)=>{dispatches++;return dispatch(...x)};return pass};return encoder};
 for(const c of await(await fetch('./cases.json')).json()){
  postMessage({stage:c.id});const feeds={};for(const [key,f] of Object.entries(c.inputs))feeds[key]=new ort.Tensor('float32',new Float32Array(await(await fetch(f.path)).arrayBuffer()),f.shape);
  const expected=new Float32Array(await(await fetch(c.expected)).arrayBuffer());const start=performance.now(),out=await session.run(feeds),elapsed=performance.now()-start,actual=out.log_assignment.data;let maxLogError=0,maxScoreError=0,finite=true;
  for(let i=0;i<actual.length;i++){finite&&=Number.isFinite(actual[i]);maxLogError=Math.max(maxLogError,Math.abs(actual[i]-expected[i]));maxScoreError=Math.max(maxScoreError,Math.abs(Math.exp(actual[i])-Math.exp(expected[i])))}
  const n=c.inputs.keypoints0.shape[1],a=pairs(actual,n),e=pairs(expected,n),same=JSON.stringify(a)===JSON.stringify(e);
  const result={id:c.id,first_run_ms:elapsed,finite,max_log_error:maxLogError,max_score_error:maxScoreError,pairs:a.length,expected_pairs:e.length,identical_pairs:same,webgpu_dispatches:dispatches};report.cases.push(result);
  for(const v of Object.values(out))v.dispose();for(const v of Object.values(feeds))v.dispose();
  if(!finite||!same||maxScoreError>.001||dispatches===0)throw Error('Matcher parity or GPU dispatch check failed');
 }
 await session.release();report.complete=true;
}catch(e){report.error=String(e);report.complete=true}
await fetch('/report',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(report)});postMessage(report);
