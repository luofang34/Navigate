export function cameraForImage(sourceWidth,sourceHeight,fov,longEdge=640){
  if(![sourceWidth,sourceHeight,fov].every(Number.isFinite)||Math.min(sourceWidth,sourceHeight)<=0||fov<=10||fov>=170)throw Error('Invalid source dimensions or sensor field of view');
  if(![640,960,1280].includes(longEdge))throw Error('Unsupported matching resolution');
  const longest=Math.max(sourceWidth,sourceHeight),width=Math.max(64,Math.round(sourceWidth/longest*longEdge/8)*8),height=Math.max(64,Math.round(sourceHeight/longest*longEdge/8)*8);
  const focal=longest*1.25/(2*Math.tan(fov*Math.PI/360));
  return {width,height,fx:focal*width/sourceWidth,fy:focal*height/sourceHeight,cx:(width-1)/2,cy:(height-1)/2};
}
