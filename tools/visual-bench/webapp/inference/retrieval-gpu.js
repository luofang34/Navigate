const matrix=`
struct Shape { n:u32, m:u32, index:u32 }
@group(0) @binding(0) var<storage,read> a:array<f32>;
@group(0) @binding(1) var<storage,read> b:array<f32>;
@group(0) @binding(2) var<storage,read_write> scores:array<f32>;
@group(0) @binding(3) var<uniform> shape:Shape;
var<workgroup> at:array<f32,256>;
var<workgroup> bt:array<f32,256>;
@compute @workgroup_size(16,16) fn main(@builtin(local_invocation_id) l:vec3<u32>,@builtin(global_invocation_id) g:vec3<u32>){
 var sum=0.0;
 for(var k=0u;k<256u;k+=16u){
  at[l.y*16u+l.x]=select(0.0,a[(k+l.y)*shape.n+g.x],g.x<shape.n);
  bt[l.y*16u+l.x]=select(0.0,b[(k+l.x)*shape.m+g.y],g.y<shape.m);
  workgroupBarrier();
  for(var d=0u;d<16u;d++){sum+=at[d*16u+l.x]*bt[l.y*16u+d];}
  workgroupBarrier();
 }
 if(g.x<shape.n && g.y<shape.m){scores[g.y*shape.n+g.x]=sum;}
}`;
const nearest=`
struct Shape { n:u32, m:u32, index:u32 }
@group(0) @binding(0) var<storage,read> scores:array<f32>;
@group(0) @binding(1) var<storage,read_write> best:array<vec4<f32>>;
@group(0) @binding(2) var<uniform> shape:Shape;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g:vec3<u32>){
 let id=g.x;if(id>=shape.n+shape.m){return;}
 let row=id<shape.m;let fixed=select(id-shape.m,id,row);let count=select(shape.m,shape.n,row);
 var first=-2.0;var second=-2.0;var index=0u;
 for(var j=0u;j<count;j++){let offset=select(j*shape.n+fixed,fixed*shape.n+j,row);let s=scores[offset];if(s>first){second=first;first=s;index=j;}else if(s>second){second=s;}}
 best[id]=vec4<f32>(f32(index),first,second,0.0);
}`;
const rank=`
struct Shape { n:u32, m:u32, index:u32 }
@group(0) @binding(0) var<storage,read> best:array<vec4<f32>>;
@group(0) @binding(1) var<storage,read_write> ranks:array<atomic<u32>>;
@group(0) @binding(2) var<uniform> shape:Shape;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g:vec3<u32>){
 let q=g.x;if(q>=shape.m){return;}let v=best[q];let r=u32(v.x);
 if(u32(best[shape.m+r].x)==q && v.y>0.65 && (1.0-v.y)<0.81*(1.0-v.z)){atomicAdd(&ranks[shape.index],1u);}
}`;
export class DescriptorRetrieval {
  static async create(device,dimensions=256,{minimumSimilarity=.65,maximumDistanceRatio=.81}={}){if(![64,256].includes(dimensions))throw Error('Unsupported descriptor width');const self=new DescriptorRetrieval();self.device=device;self.matchPolicy={minimumSimilarity,maximumDistanceRatio};self.uploads=new Map();self.dispatches=0;self.uploadCount=0;self.uploadBytes=0;
    self.matrix=await device.createComputePipelineAsync({layout:'auto',compute:{module:device.createShaderModule({code:matrix.replace('k<256u','k<'+dimensions+'u')}),entryPoint:'main'}});
    self.nearest=await device.createComputePipelineAsync({layout:'auto',compute:{module:device.createShaderModule({code:nearest}),entryPoint:'main'}});
    self.rank=await device.createComputePipelineAsync({layout:'auto',compute:{module:device.createShaderModule({code:rank.replace('v.y>0.65','v.y>'+minimumSimilarity.toFixed(6)).replace('<0.81*','<'+maximumDistanceRatio.toFixed(6)+'*')}),entryPoint:'main'}});
    const buffer=(size,usage)=>device.createBuffer({size,usage});self.shape=buffer(160*256,GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST);self.scores=buffer(4096*4096*4,GPUBufferUsage.STORAGE);self.best=buffer(8192*16,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC);self.ranks=buffer(160*4,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC);self.readback=buffer(160*4,GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ);self.pairReadback=buffer(8192*16,GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ);return self;
  }
  descriptors(f){if(this.uploads.has(f)){const buffer=this.uploads.get(f);this.uploads.delete(f);this.uploads.set(f,buffer);return buffer;}const buffer=this.device.createBuffer({size:f.descriptors.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});this.device.queue.writeBuffer(buffer,0,f.descriptors);this.uploadCount++;this.uploadBytes+=f.descriptors.byteLength;this.uploads.set(f,buffer);if(this.uploads.size>192){const [key,old]=this.uploads.entries().next().value;old.destroy();this.uploads.delete(key)}return buffer}
  async rankGrid(firsts,seconds,limit,progress=()=>{}){
    if(!Number.isSafeInteger(limit)||limit<0)throw Error('Invalid retrieval result limit');
    if(!limit||!firsts.length||!seconds.length)return [];
    const ranked=[];
    // A reference batch fits the GPU cache and stays resident across camera headings.
    for(let start=0;start<firsts.length;start+=160){
      const batch=firsts.slice(start,start+160);
      for(const [q,second] of seconds.entries()){
        progress(`Searching area ${Math.floor(start/160)+1}/${Math.ceil(firsts.length/160)} · view ${q+1}/${seconds.length}`);
        const scores=await this.rankMany(batch,second);
        if(scores.length!==batch.length||scores.some(score=>!Number.isFinite(score)))throw Error('Invalid retrieval scores');
        for(const [index,score] of scores.entries())ranked.push({reference_index:start+index,query_index:q,score});
      }
    }
    ranked.sort((a,b)=>b.score-a.score||a.query_index-b.query_index||a.reference_index-b.reference_index);
    return ranked.slice(0,limit).map(({reference_index,query_index})=>({reference_index,query_index}));
  }
  async rankMany(firsts,second){
    if(firsts.length>160)throw Error('Too many retrieval references');if(second.count<6)return firsts.map(()=>0);
    const d=this.device,shape=new Uint32Array(firsts.length*64);for(let i=0;i<firsts.length;i++)shape.set([firsts[i].count,second.count,i],i*64);d.queue.writeBuffer(this.shape,0,shape);
    const encoder=d.createCommandEncoder();encoder.clearBuffer(this.ranks);const pass=encoder.beginComputePass();
    const group=(pipeline,buffers,index)=>d.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:buffers.map((buffer,binding)=>({binding,resource:{buffer,...(buffer===this.shape?{offset:index*256,size:16}:{})}}))});
    for(const [i,first] of firsts.entries()){
      if(first.count<6)continue;
      pass.setPipeline(this.matrix);pass.setBindGroup(0,group(this.matrix,[this.descriptors(first),this.descriptors(second),this.scores,this.shape],i));pass.dispatchWorkgroups(Math.ceil(first.count/16),Math.ceil(second.count/16));
      pass.setPipeline(this.nearest);pass.setBindGroup(0,group(this.nearest,[this.scores,this.best,this.shape],i));pass.dispatchWorkgroups(Math.ceil((first.count+second.count)/64));
      pass.setPipeline(this.rank);pass.setBindGroup(0,group(this.rank,[this.best,this.ranks,this.shape],i));pass.dispatchWorkgroups(Math.ceil(second.count/64));this.dispatches+=3;
    }
    pass.end();encoder.copyBufferToBuffer(this.ranks,0,this.readback,0,firsts.length*4);d.queue.submit([encoder.finish()]);
    await this.readback.mapAsync(GPUMapMode.READ,0,firsts.length*4);const result=[...new Uint32Array(this.readback.getMappedRange(0,firsts.length*4))];this.readback.unmap();return result;
  }
  async matchPairs(first,second){
    if(first.count>4096||second.count>4096)throw Error('Too many matching features');
    const d=this.device;d.queue.writeBuffer(this.shape,0,new Uint32Array([first.count,second.count,0,0]));
    const encoder=d.createCommandEncoder(),pass=encoder.beginComputePass();
    const group=(pipeline,buffers)=>d.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:buffers.map((buffer,binding)=>({binding,resource:{buffer,...(buffer===this.shape?{size:16}:{})}}))});
    pass.setPipeline(this.matrix);pass.setBindGroup(0,group(this.matrix,[this.descriptors(first),this.descriptors(second),this.scores,this.shape]));pass.dispatchWorkgroups(Math.ceil(first.count/16),Math.ceil(second.count/16));
    pass.setPipeline(this.nearest);pass.setBindGroup(0,group(this.nearest,[this.scores,this.best,this.shape]));pass.dispatchWorkgroups(Math.ceil((first.count+second.count)/64));pass.end();
    const size=(first.count+second.count)*16;encoder.copyBufferToBuffer(this.best,0,this.pairReadback,0,size);d.queue.submit([encoder.finish()]);
    await this.pairReadback.mapAsync(GPUMapMode.READ,0,size);
    try {return mutualPairs(new Float32Array(this.pairReadback.getMappedRange(0,size)),first.count,second.count,this.matchPolicy)}finally{this.pairReadback.unmap()}
  }
  close(){for(const b of this.uploads.values())b.destroy();for(const b of [this.shape,this.scores,this.best,this.ranks,this.readback,this.pairReadback])b.destroy()}
}

export function mutualPairs(best,n,m,{minimumSimilarity=.65,maximumDistanceRatio=.81}={}){
  const pairs=[];
  for(let q=0;q<m;q++){
    const r=best[q*4],reverse=(m+r)*4;
    if(!Number.isInteger(r)||r<0||r>=n)continue;
    if(best[reverse]===q&&best[q*4+1]>minimumSimilarity&&(1-best[q*4+1])<maximumDistanceRatio*(1-best[q*4+2])&&(1-best[reverse+1])<maximumDistanceRatio*(1-best[reverse+2]))pairs.push([r,q]);
  }
  return pairs;
}
