import {LocalMatcher} from './inference/local.js';
import {gray} from './observation.js';
const report={};let matcher;
try{
 matcher=new LocalMatcher();await matcher.initialize(text=>postMessage({progress:text}));
 const bitmap=await createImageBitmap(await(await fetch('/models/test-input.png')).blob());
 for(const [width,height] of [[640,360],[640,640]]){
  const image=gray(bitmap,width,height),canvas=new OffscreenCanvas(width,height),ctx=canvas.getContext('2d');ctx.drawImage(image.canvas,24,16);const translated=gray(canvas,width,height);
  const f=await matcher.matcher.features(image,undefined,1536),g=await matcher.matcher.features(translated,undefined,1536),identity=await matcher.matcher.pairs(f,f),shift=await matcher.matcher.pairs(f,g);
  const errors=shift.map(p=>Math.hypot(p.query[0]-p.reference[0]-24,p.query[1]-p.reference[1]-16)).sort((a,b)=>a-b);
  report[`${width}x${height}`]={count:f.count,maxX:Math.max(...f.pixels.filter((_,i)=>i%2===0)),maxY:Math.max(...f.pixels.filter((_,i)=>i%2===1)),identity:identity.length,shift:shift.length,median:errors[Math.floor(errors.length/2)],examples:shift.slice(0,10)};
 }
 for(const result of Object.values(report))if(result.identity!==result.count||result.shift<100||result.median>2)throw Error('XFeat failed the identity or known translation check');report.execution=matcher.diagnostics();if(!report.execution.feature_gpu_dispatches||!report.execution.matching_gpu_dispatches)throw Error('XFeat GPU execution was not observed');
}catch(e){report.error=String(e)}finally{await matcher?.close();postMessage({report})}
