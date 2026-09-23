import {LearnedMatcher} from './learned.js';
let matcher;
self.onmessage=async({data})=>{try{
  let value;
  if(data.method==='initialize'){matcher=await LearnedMatcher.create(data.args[0]);value={identity:matcher.identity}}
  else if(data.method==='match'){const [a,b,keys]=data.args;const query=await matcher.features(b,keys.query);value=query.count<6?[]:await matcher.pairs(await matcher.features(a,keys.reference),query)}
  else throw Error('Unknown matcher request');
  self.postMessage({id:data.id,value});
}catch(error){self.postMessage({id:data.id,error:String(error)})}};
