export class BrowserPipeline {
  constructor(){
    this.worker=new Worker(new URL('./localization-worker.js',import.meta.url),{type:'module'});this.pending=new Map();this.next=0;
    this.worker.onmessage=({data})=>{const p=this.pending.get(data.id);if(!p)return;if(data.progress){p.progress(data.progress);return}this.pending.delete(data.id);data.error?p.reject(Error(data.error)):p.resolve(data.value)};
    this.worker.onerror=e=>this.fail(Error(e.message));
  }
  call(method,args,progress=()=>{}){if(this.closed)return Promise.reject(new DOMException('Processing cancelled','AbortError'));return new Promise((resolve,reject)=>{const id=this.next=(this.next+1)>>>0;this.pending.set(id,{resolve,reject,progress});this.worker.postMessage({id,method,args})})}
  initialize(pack,camera,progress,options={}){return this.call('initialize',[pack,camera,options],progress)}
  beginSequence(options){return this.call('beginSequence',options?[options]:[])}
  finishSequence(){return this.call('finishSequence',[])}
  reconstructSequence(groups,progress){return this.call('reconstructSequence',[groups],progress)}
  observeScene(image,prior,sequence,expected,progress){return this.call('observeScene',[imageMessage(image),prior,sequence,expected],progress)}
  refineScenePaths(groups,reconstruction,frames,progress){return this.call('refineScenePaths',[groups,reconstruction,frames],progress)}
  refineScenes(groups,reconstruction,progress){return this.call('refineScenes',[groups,reconstruction],progress)}
  registerScene(plan,image,prior,progress){return this.call('registerScene',[plan,imageMessage(image),prior],progress)}
  estimate(image,prior,sequence,progress){const {gray,width,height,time,requested_time_s,timing}=image;return this.call('estimate',[{gray,width,height,time,requested_time_s,timing},prior,sequence],progress)}
  trackFrom(reference,image,prior,sequence,progress){return this.call('trackFrom',[{report:reference.report,image:imageMessage(reference.image)},imageMessage(image),prior,sequence],progress)}
  checkMotion(reference,observation,image,prior,candidates,progress){return this.call('checkMotion',[{report:reference.report,image:imageMessage(reference.image)},observation,imageMessage(image),prior,candidates],progress)}
  refineAt(observation,image,prior,candidates,progress){return this.call('refineAt',[observation,imageMessage(image),prior,candidates],progress)}
  fail(error){for(const p of this.pending.values())p.reject(error);this.pending.clear()}
  close(){this.closed=true;this.worker.terminate();this.fail(new DOMException('Processing cancelled','AbortError'))}
}

function imageMessage(image){const {gray,width,height,time,requested_time_s,timing}=image;return {gray,width,height,time,requested_time_s,timing}}
