export class BrowserPipeline {
  constructor(){
    this.worker=new Worker(new URL('./localization-worker.js',import.meta.url),{type:'module'});this.pending=new Map();this.next=0;
    this.worker.onmessage=({data})=>{const p=this.pending.get(data.id);if(!p)return;if(data.progress){p.progress(data.progress);return}this.pending.delete(data.id);data.error?p.reject(Error(data.error)):p.resolve(data.value)};
    this.worker.onerror=e=>this.fail(Error(e.message));
  }
  call(method,args,progress=()=>{}){return new Promise((resolve,reject)=>{const id=this.next=(this.next+1)>>>0;this.pending.set(id,{resolve,reject,progress});this.worker.postMessage({id,method,args})})}
  initialize(pack,camera,progress,options={}){return this.call('initialize',[pack,camera,options],progress)}
  beginSequence(){return this.call('beginSequence',[])}
  estimate(image,prior,sequence,progress){const {gray,width,height,time,requested_time_s,timing}=image;return this.call('estimate',[{gray,width,height,time,requested_time_s,timing},prior,sequence],progress)}
  fail(error){for(const p of this.pending.values())p.reject(error);this.pending.clear()}
  close(){this.worker.terminate();this.fail(new DOMException('Processing cancelled','AbortError'))}
}
