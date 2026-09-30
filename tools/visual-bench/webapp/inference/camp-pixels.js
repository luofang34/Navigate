export function validateCampImage(image){
 const {rgb,width,height}=image;
 if(!Number.isInteger(width)||!Number.isInteger(height)||Math.min(width,height)<1||Math.max(width,height)>8192||!(rgb instanceof Uint8Array)||rgb.length!==width*height*3)throw Error('Invalid CAMP RGB image');
}
export function campPixels(image,size=384){
 validateCampImage(image);if(!Number.isInteger(size)||size<1||size>384)throw Error('Invalid CAMP output size');
 const {rgb,width,height}=image;
 const plane=size*size,out=new Float32Array(plane*3);
 for(let y=0;y<size;y++)for(let x=0;x<size;x++){
  const sx=(x+.5)*width/size-.5,sy=(y+.5)*height/size-.5,left=Math.floor(sx),top=Math.floor(sy),fx=sx-left,fy=sy-top;
  for(let c=0;c<3;c++){
   let value=0;
   for(let dy=0;dy<2;dy++)for(let dx=0;dx<2;dx++)value+=rgb[(Math.max(0,Math.min(height-1,top+dy))*width+Math.max(0,Math.min(width-1,left+dx)))*3+c]*(dx?fx:1-fx)*(dy?fy:1-fy);
   out[c*plane+y*size+x]=Math.round(value);
  }
 }
 return out;
}
