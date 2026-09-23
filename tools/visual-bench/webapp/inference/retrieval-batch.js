const PACK_BYTES=16*1024*1024;
const BEST_ENTRIES=65536;

export function referenceBatchSize(features,dimensions){
  const largest=Math.max(1,...features.map(f=>f.count));
  if(![64,256].includes(dimensions)||!Number.isInteger(largest)||largest>4096)throw Error('Invalid retrieval descriptor shape');
  return Math.min(160,Math.floor(PACK_BYTES/(largest*dimensions*4)));
}

function validateFeatures(feature,dimensions){
  if(!Number.isInteger(feature.count)||feature.count<0||feature.count>4096||!(feature.descriptors instanceof Float32Array)||feature.descriptors.length!==feature.count*dimensions)throw Error('Invalid retrieval descriptor shape');
}

export function packDescriptors(features,dimensions){
  if(!features.length||features.length>referenceBatchSize(features,dimensions))throw Error('Invalid retrieval reference batch');
  for(const feature of features)validateFeatures(feature,dimensions);
  const stride=Math.max(1,...features.map(f=>f.count));
  const data=new Float32Array(features.length*dimensions*stride),counts=new Uint32Array(features.length);
  for(const [index,feature] of features.entries()){
    counts[index]=feature.count;
    for(let channel=0;channel<dimensions;channel++)data.set(feature.descriptors.subarray(channel*feature.count,(channel+1)*feature.count),(index*dimensions+channel)*stride);
  }
  return {data,counts,stride};
}

function shaders(dimensions,policy){
  const shape='struct Shape { n:u32, m:u32, index:u32 }';
  return [
`${shape}
@group(0) @binding(0) var<storage,read> a:array<f32>;
@group(0) @binding(1) var<storage,read> b:array<f32>;
@group(0) @binding(2) var<storage,read_write> scores:array<f32>;
@group(0) @binding(3) var<uniform> shape:Shape;
@group(0) @binding(4) var<storage,read> counts:array<u32>;
var<workgroup> at:array<f32,256>;
var<workgroup> bt:array<f32,256>;
@compute @workgroup_size(16,16) fn main(@builtin(local_invocation_id) l:vec3<u32>,@builtin(global_invocation_id) g:vec3<u32>){
 let reference=shape.index+g.z;let n=counts[reference];var sum=0.0;
 for(var k=0u;k<${dimensions}u;k+=16u){
  at[l.y*16u+l.x]=select(0.0,a[(reference*${dimensions}u+k+l.y)*shape.n+g.x],g.x<n);
  bt[l.y*16u+l.x]=select(0.0,b[(k+l.x)*shape.m+g.y],g.y<shape.m);
  workgroupBarrier();
  for(var d=0u;d<16u;d++){sum+=at[d*16u+l.x]*bt[l.y*16u+d];}
  workgroupBarrier();
 }
 if(g.x<n && g.y<shape.m){scores[(g.z*shape.m+g.y)*shape.n+g.x]=sum;}
}`,
`${shape}
@group(0) @binding(0) var<storage,read> scores:array<f32>;
@group(0) @binding(1) var<storage,read_write> best:array<vec4<f32>>;
@group(0) @binding(2) var<uniform> shape:Shape;
@group(0) @binding(3) var<storage,read> counts:array<u32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g:vec3<u32>){
 let n=counts[shape.index+g.y];let id=g.x;if(n<6u || id>=n+shape.m){return;}
 let row=id<shape.m;let fixed=select(id-shape.m,id,row);let count=select(shape.m,n,row);
 var first=-2.0;var second=-2.0;var index=0u;
 for(var j=0u;j<count;j++){let offset=g.y*shape.m*shape.n+select(j*shape.n+fixed,fixed*shape.n+j,row);let s=scores[offset];if(s>first){second=first;first=s;index=j;}else if(s>second){second=s;}}
 best[g.y*(shape.n+shape.m)+id]=vec4<f32>(f32(index),first,second,0.0);
}`,
`${shape}
@group(0) @binding(0) var<storage,read> best:array<vec4<f32>>;
@group(0) @binding(1) var<storage,read_write> ranks:array<atomic<u32>>;
@group(0) @binding(2) var<uniform> shape:Shape;
@group(0) @binding(3) var<storage,read> counts:array<u32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g:vec3<u32>){
 let q=g.x;let reference=shape.index+g.y;if(counts[reference]<6u || q>=shape.m){return;}
 let base=g.y*(shape.n+shape.m);let v=best[base+q];let r=u32(v.x);
 if(u32(best[base+shape.m+r].x)==q && v.y>${policy.minimumSimilarity.toFixed(6)} && (1.0-v.y)<${policy.maximumDistanceRatio.toFixed(6)}*(1.0-v.z)){atomicAdd(&ranks[reference],1u);}
}`];
}

export class BatchedDescriptorRetrieval {
  static async create(owner,dimensions){
    const self=new BatchedDescriptorRetrieval();self.owner=owner;self.dimensions=dimensions;
    self.pipelines=await Promise.all(shaders(dimensions,owner.matchPolicy).map(code=>owner.device.createComputePipelineAsync({layout:'auto',compute:{module:owner.device.createShaderModule({code}),entryPoint:'main'}})));
    self.best=owner.device.createBuffer({size:BEST_ENTRIES*16,usage:GPUBufferUsage.STORAGE});
    return self;
  }
  prepare(features){
    const {data,counts,stride}=packDescriptors(features,this.dimensions),d=this.owner.device;
    const descriptors=d.createBuffer({size:data.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
    const sizes=d.createBuffer({size:counts.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
    d.queue.writeBuffer(descriptors,0,data);d.queue.writeBuffer(sizes,0,counts);
    this.owner.uploadCount++;this.owner.uploadBytes+=data.byteLength;
    return {descriptors,sizes,stride,count:features.length,close(){descriptors.destroy();sizes.destroy()}};
  }
  async rank(batch,query){
    validateFeatures(query,this.dimensions);
    if(query.count<6)return Array(batch.count).fill(0);
    const o=this.owner,d=o.device,n=batch.stride,m=query.count;
    // Each group reuses bounded score and nearest-neighbour scratch storage.
    const capacity=Math.min(batch.count,Math.floor(o.scores.size/(n*m*4)),Math.floor(BEST_ENTRIES/(n+m)));
    if(capacity<1)throw Error('Retrieval scratch capacity is insufficient');
    const shape=new Uint32Array(Math.ceil(batch.count/capacity)*64);
    for(let start=0,index=0;start<batch.count;start+=capacity,index++)shape.set([n,m,start],index*64);
    d.queue.writeBuffer(o.shape,0,shape);
    const queryBuffer=o.descriptors(query),encoder=d.createCommandEncoder();encoder.clearBuffer(o.ranks);const pass=encoder.beginComputePass();
    const bind=(pipeline,buffers,index)=>d.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:buffers.map((buffer,binding)=>({binding,resource:{buffer,...(buffer===o.shape?{offset:index*256,size:16}:{})}}))});
    const [matrix,nearest,rank]=this.pipelines;
    for(let start=0,index=0;start<batch.count;start+=capacity,index++){
      const count=Math.min(capacity,batch.count-start);
      pass.setPipeline(matrix);pass.setBindGroup(0,bind(matrix,[batch.descriptors,queryBuffer,o.scores,o.shape,batch.sizes],index));pass.dispatchWorkgroups(Math.ceil(n/16),Math.ceil(m/16),count);
      pass.setPipeline(nearest);pass.setBindGroup(0,bind(nearest,[o.scores,this.best,o.shape,batch.sizes],index));pass.dispatchWorkgroups(Math.ceil((n+m)/64),count);
      pass.setPipeline(rank);pass.setBindGroup(0,bind(rank,[this.best,o.ranks,o.shape,batch.sizes],index));pass.dispatchWorkgroups(Math.ceil(m/64),count);o.dispatches+=3;
    }
    pass.end();encoder.copyBufferToBuffer(o.ranks,0,o.readback,0,batch.count*4);d.queue.submit([encoder.finish()]);
    await o.readback.mapAsync(GPUMapMode.READ,0,batch.count*4);
    try{return [...new Uint32Array(o.readback.getMappedRange(0,batch.count*4))]}finally{o.readback.unmap()}
  }
  close(){this.best.destroy()}
}
