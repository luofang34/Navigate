export function decodeXFeat(result,image,limit=512,transform={scale:1,x:0,y:0}){
  const {keypoints,descriptors,scores}=result,{width,height,valid}=image;
  if(keypoints.dims.length!==2||keypoints.dims[1]!==2||descriptors.dims[1]!==64||keypoints.dims[0]!==descriptors.dims[0]||scores.data.length!==keypoints.dims[0])throw Error('Invalid XFeat outputs');
  const selected=[];
  for(let i=0;i<scores.data.length;i++){
    const x=(Number(keypoints.data[i*2])-transform.x)/transform.scale,y=(Number(keypoints.data[i*2+1])-transform.y)/transform.scale,score=scores.data[i];
    if(!Number.isFinite(x)||!Number.isFinite(y)||!Number.isFinite(score)||score<=0||x<4||y<4||x>=width-4||y>=height-4)continue;
    let supported=true;
    if(valid)for(let dy=-4;dy<=4&&supported;dy++)for(let dx=-4;dx<=4;dx++)if(valid[(Math.round(y)+dy)*width+Math.round(x)+dx]!==255){supported=false;break}
    if(supported)selected.push({i,x,y,score});
  }
  selected.sort((a,b)=>b.score-a.score);selected.length=Math.min(selected.length,limit);
  const count=selected.length,pixels=new Float32Array(count*2),confidence=new Float32Array(count),data=new Float32Array(count*64);
  for(const [j,p] of selected.entries()){
    pixels.set([p.x,p.y],j*2);confidence[j]=p.score;let norm=0;
    for(let c=0;c<64;c++){const value=descriptors.data[p.i*64+c];if(!Number.isFinite(value))throw Error('Non-finite XFeat descriptor');norm+=value*value;data[c*count+j]=value}
    if(norm<=0)throw Error('Empty XFeat descriptor');
    for(let c=0;c<64;c++)data[c*count+j]/=Math.sqrt(norm);
  }
  return {pixels,scores:confidence,descriptors:data,width,height,count};
}

export function xfeatInput(image){
  // This release's traced resize and output coordinates use an 800 by 600 image.
  const width=800,height=600,scale=Math.min(width/image.width,height/image.height),x=(width-image.width*scale)/2,y=(height-image.height*scale)/2;
  const plane=width*height,data=new Float32Array(plane*3);
  for(let py=0;py<height;py++)for(let px=0;px<width;px++){
    const sx=(px-x+.5)/scale-.5,sy=(py-y+.5)/scale-.5;
    if(sx<-.5||sy<-.5||sx>=image.width-.5||sy>=image.height-.5)continue;
    const left=Math.floor(sx),top=Math.floor(sy),fx=sx-left,fy=sy-top;let value=0;
    for(let dy=0;dy<2;dy++)for(let dx=0;dx<2;dx++)value+=image.gray[Math.max(0,Math.min(image.height-1,top+dy))*image.width+Math.max(0,Math.min(image.width-1,left+dx))]*(dx?fx:1-fx)*(dy?fy:1-fy)/255;
    const i=py*width+px;data[i]=data[plane+i]=data[plane*2+i]=value;
  }
  return {data,width,height,transform:{scale,x:x+(scale-1)/2,y:y+(scale-1)/2}};
}

export function emptyXFeat(width,height){return {pixels:new Float32Array(),scores:new Float32Array(),descriptors:new Float32Array(),width,height,count:0}}
export function isEmptyXFeatOutput(error){return String(error).includes("Name:'/Where'")&&String(error).includes('Condition Shape: {1,0,2}, X Shape: {1,0}, Y Shape: {}')}
