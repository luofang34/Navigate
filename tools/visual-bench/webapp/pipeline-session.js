import {BrowserPipeline} from './browser-pipeline.js';

export class PipelineSession {
  constructor(create=()=>new BrowserPipeline()){this.create=create;this.current=null;this.key=null;this.ready=null}
  async acquire(pack,camera,options,progress){
    const key=JSON.stringify([pack.pack_id,camera,options]);
    if(this.current&&this.key===key)return this.ready;
    this.close();const pipeline=this.create();this.current=pipeline;this.key=key;
    this.ready=(async()=>{
      try {
        await pipeline.initialize(pack,camera,progress,options);
        if(this.current!==pipeline)throw new DOMException('Processing cancelled','AbortError');
        return pipeline;
      } catch(error){if(this.current===pipeline)this.close();throw error}
    })();
    return this.ready;
  }
  close(){this.current?.close();this.current=null;this.key=null;this.ready=null}
}
