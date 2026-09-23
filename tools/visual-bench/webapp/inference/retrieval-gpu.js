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
  static async create(device){const self=new DescriptorRetrieval();self.device=device;self.uploads=new Map();self.dispatches=0;
    self.matrix=await device.createComputePipelineAsync({layout:'auto',compute:{module:device.createShaderModule({code:matrix}),entryPoint:'main'}});
    self.nearest=await device.createComputePipelineAsync({layout:'auto',compute:{module:device.createShaderModule({code:nearest}),entryPoint:'main'}});
    self.rank=await device.createComputePipelineAsync({layout:'auto',compute:{module:device.createShaderModule({code:rank}),entryPoint:'main'}});
    const buffer=(size,usage)=>device.createBuffer({size,usage});self.shape=buffer(160*256,GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST);self.scores=buffer(512*512*4,GPUBufferUsage.STORAGE);self.best=buffer(1024*16,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC);self.ranks=buffer(160*4,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC);self.readback=buffer(160*4,GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ);return self;
  }
  descriptors(f){if(this.uploads.has(f))return this.uploads.get(f);const buffer=this.device.createBuffer({size:f.descriptors.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});this.device.queue.writeBuffer(buffer,0,f.descriptors);this.uploads.set(f,buffer);if(this.uploads.size>192){const [key,old]=this.uploads.entries().next().value;old.destroy();this.uploads.delete(key)}return buffer}
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
  close(){for(const b of this.uploads.values())b.destroy();for(const b of [this.shape,this.scores,this.best,this.ranks,this.readback])b.destroy()}
}
