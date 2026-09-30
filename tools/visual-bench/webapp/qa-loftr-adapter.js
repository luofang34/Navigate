import * as ort from './runtime/ort.webgpu.min.mjs';
export function denseInput(image){
 const width=640,height=480,scale=Math.min(width/image.width,height/image.height),x=(width-image.width*scale)/2,y=(height-image.height*scale)/2;
 const data=new Float32Array(width*height);let min=255,max=0;for(const value of image.gray){min=Math.min(min,value);max=Math.max(max,value)}
 for(let py=0;py<height;py++)for(let px=0;px<width;px++){
  const sx=(px-x+.5)/scale-.5,sy=(py-y+.5)/scale-.5;if(sx<-.5||sy<-.5||sx>=image.width-.5||sy>=image.height-.5)continue;
  const left=Math.floor(sx),top=Math.floor(sy),fx=sx-left,fy=sy-top;let value=0;
  for(let dy=0;dy<2;dy++)for(let dx=0;dx<2;dx++)value+=image.gray[Math.max(0,Math.min(image.height-1,top+dy))*image.width+Math.max(0,Math.min(image.width-1,left+dx))]*(dx?fx:1-fx)*(dy?fy:1-fy)/255;
  data[py*width+px]=value;
 }
 return {data,flat:max-min<2,pixel:p=>[(p[0]-x-(scale-1)/2)/scale,(p[1]-y-(scale-1)/2)/scale]};
}
function supported(image,p){
 const [x,y]=p;if(!Number.isFinite(x)||!Number.isFinite(y)||x<4||y<4||x>=image.width-4||y>=image.height-4)return false;
 if(image.valid)for(let dy=-4;dy<=4;dy++)for(let dx=-4;dx<=4;dx++)if(image.valid[(Math.round(y)+dy)*image.width+Math.round(x)+dx]!==255)return false;return true;
}
export class DenseMatcher {
 static async create(bytes,tracker,metrics){const self=new DenseMatcher();self.tracker=tracker;self.metrics=metrics;self.session=await ort.InferenceSession.create(bytes,{executionProviders:['webgpu','wasm'],graphOptimizationLevel:'all'});return self}
 async match(reference,query){
  const a=denseInput(reference),b=denseInput(query);if(a.flat||b.flat)return [];
  const feeds={image0:new ort.Tensor('float32',a.data,[1,1,480,640]),image1:new ort.Tensor('float32',b.data,[1,1,480,640])};let result;
  this.tracker.phase('matching');this.metrics.match_runs++;
  try{
   result=await this.session.run(feeds);const {keypoints0,keypoints1,confidence}=result,pairs=[];
   if(keypoints0.data.length!==confidence.data.length*2||keypoints1.data.length!==keypoints0.data.length)throw Error('Invalid dense matcher output shape');
   for(let i=0;i<confidence.data.length;i++){
    const r=a.pixel(keypoints0.data.subarray(i*2,i*2+2)),q=b.pixel(keypoints1.data.subarray(i*2,i*2+2));
    if(confidence.data[i]>.2&&supported(reference,r)&&supported(query,q))pairs.push({reference:r,query:q});
   }
   this.metrics.correspondences_max=Math.max(this.metrics.correspondences_max||0,pairs.length);return pairs;
  }finally{for(const t of Object.values(feeds))t.dispose();if(result)for(const t of Object.values(result))t.dispose()}
 }
 close(){return this.session.release()}
}
