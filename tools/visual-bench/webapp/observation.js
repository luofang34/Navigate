export const makeCanvas=()=>typeof document==='undefined'?new OffscreenCanvas(1,1):document.createElement('canvas');
const canvas=makeCanvas;
export function gray(source,width,height){const c=canvas();c.width=width;c.height=height;const ctx=c.getContext('2d',{willReadFrequently:true});ctx.drawImage(source,0,0,width,height);const rgba=ctx.getImageData(0,0,width,height).data,pixels=new Uint8Array(width*height),valid=new Uint8Array(width*height);for(let i=0;i<pixels.length;i++){pixels[i]=(77*rgba[4*i]+150*rgba[4*i+1]+29*rgba[4*i+2])>>>8;valid[i]=rgba[4*i+3]}return {gray:pixels,valid,width,height,canvas:c}}
function mediaEvent(video,event){return new Promise((resolve,reject)=>{const done=()=>{cleanup();resolve()},error=()=>{cleanup();reject(Error('This browser cannot decode the video. Use MP4/H.264 or a PNG frame.'))},cleanup=()=>{video.removeEventListener(event,done);video.removeEventListener('error',error)};video.addEventListener(event,done,{once:true});video.addEventListener('error',error,{once:true})})}
export async function openInput(file,video){const url=URL.createObjectURL(file);if(file.type.startsWith('image/')){const bitmap=await createImageBitmap(file);return {type:'image',source:bitmap,close(){bitmap.close();URL.revokeObjectURL(url)}}}
  const ready=mediaEvent(video,'loadedmetadata');video.preload='auto';video.src=url;video.load();await ready;
  if(!Number.isFinite(video.duration)||video.duration<=0)throw Error('Video duration is unavailable');
  return {type:'video',source:video,duration:video.duration,close(){video.removeAttribute('src');video.load();URL.revokeObjectURL(url)}};
}
export async function frameAt(input,time,camera){let actual=time,timing='still image';if(input.type==='video'){
    const video=input.source;video.pause();const changed=Math.abs(video.currentTime-time)>1e-6;
    if(changed){const seek=mediaEvent(video,'seeked');video.currentTime=time;await seek}
    if(video.readyState<2)await mediaEvent(video,'loadeddata');
    actual=video.currentTime;timing='browser media seek time; decoded frame PTS is not independently verified';
  }
  const image=gray(input.source,camera.width,camera.height);const blob=await new Promise(resolve=>image.canvas.toBlob(resolve,'image/png'));if(!blob)throw Error('Could not encode the observation frame');return {...image,blob,time:actual,requested_time_s:time,timing};
}
export function rotationGeometry(width,height,angle){
  const radians=angle*Math.PI/180,c=Math.cos(radians),s=Math.sin(radians);
  const padded=v=>Math.ceil((v-1e-9)/8)*8,targetWidth=padded(Math.abs(c)*width+Math.abs(s)*height),targetHeight=padded(Math.abs(s)*width+Math.abs(c)*height);
  return {width:targetWidth,height:targetHeight,radians,unrotate(p){const x=p[0]+.5-targetWidth/2,y=p[1]+.5-targetHeight/2;return [c*x-s*y+(width-1)/2,s*x+c*y+(height-1)/2]}};
}
export function rotate(image,angle){const transform=rotationGeometry(image.width,image.height,angle),{width,height,radians}=transform;
  const target=canvas();target.width=width;target.height=height;const ctx=target.getContext('2d');ctx.translate(width/2,height/2);ctx.rotate(-radians);ctx.drawImage(image.canvas,-image.width/2,-image.height/2);
  return {...gray(target,width,height),unrotate:transform.unrotate};
}
