export class BrowserPipeline {
  constructor(worker=new Worker(new URL('./localization-worker.js',import.meta.url),{type:'module'})){
    this.worker=worker;this.pending=new Map();this.next=0;
    this.worker.onmessage=({data})=>{const p=this.pending.get(data.id);if(!p)return;if(data.progress){p.progress(data.progress);return}this.pending.delete(data.id);data.error?p.reject(Error(data.error)):p.resolve(data.value)};
    this.worker.onerror=e=>this.stop(Error(e.message));
    this.worker.onmessageerror=()=>this.stop(Error('Visual worker response could not be decoded'));
  }
  call(method,args,progress=()=>{},transfer=[]){
    if(this.closed)return Promise.reject(new DOMException('Processing cancelled','AbortError'));
    // The renderer and matcher share observation state across asynchronous calls.
    if(this.pending.size)return Promise.reject(new DOMException('Visual worker is busy; submit current evidence when it completes','InvalidStateError'));
    return new Promise((resolve,reject)=>{
      const id=this.next=(this.next+1)>>>0;this.pending.set(id,{resolve,reject,progress});
      try{this.worker.postMessage({id,method,args},transfer)}catch(error){this.pending.delete(id);reject(error)}
    });
  }
  async initialize(pack,camera,progress,options={}){this.requiresOriginalImage=false;const ready=await this.call('initialize',[pack,camera,options],progress);this.requiresOriginalImage=ready?.original_pixels===true;return ready}
  beginSequence(options){return this.call('beginSequence',options?[options]:[])}
  finishSequence(){return this.call('finishSequence',[])}
  reconstructSequence(groups,progress){return this.call('reconstructSequence',[groups],progress)}
  observeScene(image,prior,sequence,expected,progress){return this.call('observeScene',[imageMessage(image),prior,sequence,expected],progress)}
  refineScenePaths(groups,reconstruction,frames,progress){return this.call('refineScenePaths',[groups,reconstruction,frames],progress)}
  refineScenes(groups,reconstruction,progress){return this.call('refineScenes',[groups,reconstruction],progress)}
  registerScene(plan,image,prior,progress){return this.call('registerScene',[plan,imageMessage(image),prior],progress)}
  estimate(image,prior,sequence,progress){const {gray,rgb,original,width,height,time,requested_time_s,timing}=image;return this.call('estimate',[{gray,rgb,original,width,height,time,requested_time_s,timing},prior,sequence],progress,original?.bitmap?[original.bitmap]:[]).finally(()=>original?.bitmap?.close?.())}
  trackFrom(reference,image,prior,sequence,progress){return this.call('trackFrom',[{report:reference.report,image:imageMessage(reference.image)},imageMessage(image),prior,sequence],progress)}
  checkMotion(reference,observation,image,prior,candidates,progress){return this.call('checkMotion',[{report:reference.report,image:imageMessage(reference.image)},observation,imageMessage(image),prior,candidates],progress)}
  refineAt(observation,image,prior,candidates,progress){return this.call('refineAt',[observation,imageMessage(image),prior,candidates],progress)}
  fail(error){for(const p of this.pending.values())p.reject(error);this.pending.clear()}
  stop(error){this.closed=true;this.worker.terminate();this.fail(error)}
  close(){this.stop(new DOMException('Processing cancelled','AbortError'))}
}

function imageMessage(image){const {gray,rgb,width,height,time,requested_time_s,timing}=image;return {gray,rgb,width,height,time,requested_time_s,timing}}
