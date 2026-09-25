// Image associations persist even when map-based camera tracking has no result.
export class SequenceImages {
  constructor({create,matcher,save,camera,maxFrames=64,overlapFrames=Math.min(8,maxFrames-1)}){
    if(!Number.isInteger(maxFrames)||maxFrames<2||maxFrames>129)throw Error('Invalid image group size');
    if(!Number.isInteger(overlapFrames)||overlapFrames<1||overlapFrames>=maxFrames)throw Error('Invalid image group overlap');
    Object.assign(this,{create,matcher,save,camera,maxFrames,overlapFrames});this.overlap=[];this.current=null;this.previous=null;this.sources=[];this.backends=new Set();this.groups=[];this.seen=new Set();this.lastMatches=null;
  }
  async observe(image,observation,sequence,progress){
    if(this.previous?.observation===observation)return this.lastMatches;
    if(this.seen.has(observation))throw Error('Repeated source observation in image sequence');
    if(image.width!==this.camera.width||image.height!==this.camera.height)throw Error('Image group calibration changed');
    if(this.sources.length>=this.maxFrames){await this.flush();this.start();for(const [index,item] of this.overlap.entries()){this.current.push(item.source.observation_sha256,index===0?'[]':item.pairs);this.sources.push(item.source);if(index>0&&item.backend)this.backends.add(item.backend)}}
    if(!this.current)this.start();
    const reference=this.previous;
    let match=null;
    if(reference){
      progress('Following image features…');
      const result=await this.matcher.matchImages(reference.image,image,{reference:reference.observation+'/query',query:observation+'/query',stage:'tracking',progress});
      match={reference:reference.observation,query:observation,result};
      if(result.backend_identity)this.backends.add(result.backend_identity);
    }
    const pairs=JSON.stringify(match?.result.pairs??[]);this.current.push(observation,pairs);
    const source={observation_sha256:observation,sequence,capture_time_ns:Math.round(image.time*1e9),requested_time_s:image.requested_time_s};
    this.sources.push(source);this.seen.add(observation);
    this.overlap.push({source,pairs,backend:match?.result.backend_identity});if(this.overlap.length>this.overlapFrames)this.overlap.shift();
    this.previous={observation,source,image:{gray:image.gray.slice(),width:image.width,height:image.height}};
    this.lastMatches=match;return match;
  }
  start(){this.current=this.create();this.sources=[];this.backends=new Set()}
  async flush(){
    if(this.current&&this.sources.length>=2){
      const graph=JSON.parse(this.current.snapshot());
      const group={camera:this.camera,observations:this.sources,matcher_identities:[...this.backends],graph,
        stage:'image_associations',geographic_acceptance:false,
        evidence_correlation:'unknown; adjacent groups share observations; reverse interpolation reuses pair proposals'};
      const stored=await this.save(JSON.stringify(group));this.groups.push({...stored,observations:this.sources.map(s=>s.observation_sha256)});
    }
    this.current?.free();this.current=null;this.sources=[];
  }
  async finish(){await this.flush();const groups=this.groups;this.close();return groups}
  close(){this.current?.free();this.current=null;this.previous=null;this.overlap=[];this.sources=[];this.backends.clear();this.groups=[];this.seen.clear();this.lastMatches=null}
}
