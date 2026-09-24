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
export function supported(image,p){
 const [x,y]=p;if(!Number.isFinite(x)||!Number.isFinite(y)||x<4||y<4||x>=image.width-4||y>=image.height-4)return false;
 if(image.valid)for(let dy=-4;dy<=4;dy++)for(let dx=-4;dx<=4;dx++)if(image.valid[(Math.round(y)+dy)*image.width+Math.round(x)+dx]!==255)return false;return true;
}
export function quarterTurn(image,turn){
 if(![1,2,3].includes(turn))throw Error('Invalid matcher quarter turn');
 const width=turn===2?image.width:image.height,height=turn===2?image.height:image.width;
 const gray=new Uint8Array(width*height),valid=image.valid?new Uint8Array(width*height):undefined;
 const unrotate=([x,y])=>turn===1?[y,image.height-1-x]:turn===2?[image.width-1-x,image.height-1-y]:[image.width-1-y,x];
 for(let y=0;y<height;y++)for(let x=0;x<width;x++){const [sx,sy]=unrotate([x,y]),source=sy*image.width+sx,target=y*width+x;gray[target]=image.gray[source];if(valid)valid[target]=image.valid[source]}
 return {width,height,gray,valid,unrotate};
}
