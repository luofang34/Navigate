const f32=Math.fround;
export function detectorInput(image,width=640,height=360){
 const {gray,width:iw,height:ih}=image;if(gray.length!==iw*ih||Math.min(iw,ih)<16)throw Error('Invalid detector image');
 const scale=Math.min(width/iw,height/ih),ox=(width-iw*scale)/2,oy=(height-ih*scale)/2,data=new Float32Array(width*height);
 for(let y=0;y<height;y++)for(let x=0;x<width;x++){
  const sx=(x-ox+.5)/scale-.5,sy=(y-oy+.5)/scale-.5;
  if(sx<-.5||sy<-.5||sx>=iw-.5||sy>=ih-.5)continue;
  const left=Math.floor(sx),top=Math.floor(sy),fx=sx-left,fy=sy-top;let value=0;
  for(let dy=0;dy<2;dy++)for(let dx=0;dx<2;dx++)value+=gray[Math.max(0,Math.min(ih-1,top+dy))*iw+Math.max(0,Math.min(iw-1,left+dx))]*(dx?fx:1-fx)*(dy?fy:1-fy)/255;
  data[y*width+x]=value;
 }
 return {data,width,height,pixel:([x,y])=>[(x-ox-(scale-1)/2)/scale,(y-oy-(scale-1)/2)/scale]};
}
function maximum(values,width,height){
 const horizontal=new Float32Array(values.length),result=new Float32Array(values.length);
 for(let y=0;y<height;y++)for(let x=0;x<width;x++){let v=0;for(let k=Math.max(0,x-4);k<=Math.min(width-1,x+4);k++)v=Math.max(v,values[y*width+k]);horizontal[y*width+x]=v}
 for(let y=0;y<height;y++)for(let x=0;x<width;x++){let v=0;for(let k=Math.max(0,y-4);k<=Math.min(height-1,y+4);k++)v=Math.max(v,horizontal[k*width+x]);result[y*width+x]=v}
 return result;
}
export function suppress(heat,width,height){
 const pooled=maximum(heat,width,height),keep=heat.map((v,i)=>Number(v===pooled[i]));
 for(let round=0;round<2;round++){
  const mask=maximum(keep,width,height),remaining=heat.map((v,i)=>mask[i]>0?0:v),pooled=maximum(remaining,width,height);
  for(let i=0;i<keep.length;i++)keep[i]||=Number(mask[i]===0&&remaining[i]===pooled[i]);
 }
 return heat.map((v,i)=>keep[i]?v:0);
}
export function decodeDetector(output,image,input,limit=1024){
 const {width,height}=input,gw=width/8,gh=height/8,cells=gw*gh,logits=output.scores.data,descriptors=output.descriptors.data;
 if(JSON.stringify(output.scores.dims)!==JSON.stringify([1,65,gh,gw])||JSON.stringify(output.descriptors.dims)!==JSON.stringify([1,256,gh,gw])||logits.some(v=>!Number.isFinite(v))||descriptors.some(v=>!Number.isFinite(v)))throw Error('Invalid SuperPoint output');
 const heat=new Float32Array(width*height);
 for(let cell=0;cell<cells;cell++){
  let max=-Infinity,sum=0;for(let c=0;c<65;c++)max=Math.max(max,logits[c*cells+cell]);
  const exp=new Float32Array(65);for(let c=0;c<65;c++){exp[c]=Math.exp(f32(logits[c*cells+cell]-max));sum=f32(sum+exp[c])}
  for(let c=0;c<64;c++)heat[(Math.floor(cell/gw)*8+Math.floor(c/8))*width+cell%gw*8+c%8]=exp[c]/sum;
 }
 const nms=suppress(heat,width,height),selected=[];
 for(let i=0;i<nms.length;i++){
  const x=i%width,y=Math.floor(i/width),pixel=input.pixel([x,y]);
  if(nms[i]>.0005&&pixel[0]>=4&&pixel[1]>=4&&pixel[0]<image.width-4&&pixel[1]<image.height-4&&(!image.valid||image.valid[Math.round(pixel[1])*image.width+Math.round(pixel[0])]===255))selected.push({x,y,pixel,score:nms[i]});
 }
 selected.sort((a,b)=>b.score-a.score);selected.length=Math.min(selected.length,limit);
 const points=new Float32Array(selected.length*2),dense=new Float32Array(selected.length*256);
 for(const [i,p] of selected.entries()){
  points.set([p.x,p.y],i*2);const sx=(p.x-3.5)/(width-4.5)*(gw-1),sy=(p.y-3.5)/(height-4.5)*(gh-1),left=Math.floor(sx),top=Math.floor(sy),fx=sx-left,fy=sy-top;
  let norm=0;
  for(let c=0;c<256;c++){
   let value=0;for(let dy=0;dy<2;dy++)for(let dx=0;dx<2;dx++){const x=left+dx,y=top+dy;if(x>=0&&x<gw&&y>=0&&y<gh)value+=descriptors[c*cells+y*gw+x]*(dx?fx:1-fx)*(dy?fy:1-fy)}
   value=f32(value);dense[i*256+c]=value;norm=f32(norm+f32(value*value));
  }
  norm=f32(Math.sqrt(norm));if(!Number.isFinite(norm)||norm<=0)throw Error('Invalid SuperPoint descriptor');
  for(let c=0;c<256;c++)dense[i*256+c]/=norm;
 }
 return {pixels:selected.map(p=>p.pixel),points,descriptors:dense};
}
export function decodeAssignments(indices,first,second){
 if(indices.length!==first.pixels.length)throw Error('LightGlue output count differs from detector');
 const used=new Set(),pairs=[];
 for(let i=0;i<indices.length;i++){
  const j=Number(indices[i]);if(j===-1)continue;
  if(!Number.isSafeInteger(j)||j<0||j>=second.pixels.length||used.has(j))throw Error('Invalid LightGlue assignment');
  used.add(j);pairs.push({reference:first.pixels[i],query:second.pixels[j]});
 }
 return pairs;
}
