// Cameras such as DJI drones and phones store the recording position as an ISO 6709 string in a
// `©xyz` box inside `moov`. DJI also stores the attitude at the start of recording as text boxes:
// aircraft yaw `©fyw`, gimbal yaw relative to the aircraft `©gyw` and gimbal pitch `©gpt`, in
// degrees. Only box headers and the `moov` box are read, so a large video file is never loaded whole.
const MAX_MOOV_BYTES=64*1024*1024;

async function bytes(file,start,end){return new Uint8Array(await file.slice(start,end).arrayBuffer())}

async function moovBox(file){
 for(let offset=0;offset+8<=file.size;){
  const header=await bytes(file,offset,offset+16),view=new DataView(header.buffer);
  let size=view.getUint32(0);const type=String.fromCharCode(...header.subarray(4,8));let body=8;
  if(size===1){if(header.length<16)return null;size=Number(view.getBigUint64(8));body=16}else if(size===0)size=file.size-offset;
  if(size<body)return null;
  if(type==='moov')return size-body>MAX_MOOV_BYTES?null:bytes(file,offset+body,offset+size);
  offset+=size;
 }
 return null;
}

export function parseIso6709(text){
 const match=/^([+-]\d{1,2}(?:\.\d+)?)([+-]\d{1,3}(?:\.\d+)?)([+-]\d+(?:\.\d+)?)?/.exec(text.trim());
 if(!match)return null;
 const latitude=+match[1],longitude=+match[2],altitude=match[3]===undefined?undefined:+match[3];
 if(!(Math.abs(latitude)<=90&&Math.abs(longitude)<=180)||(latitude===0&&longitude===0))return null;
 return {latitude,longitude,...(Number.isFinite(altitude)?{altitude_m:altitude}:{})};
}

export function findLocation(moov){
 const tag=[0xa9,0x78,0x79,0x7a];
 for(let i=4;i+8<=moov.length;i++){
  if(moov[i]!==tag[0]||moov[i+1]!==tag[1]||moov[i+2]!==tag[2]||moov[i+3]!==tag[3])continue;
  const size=new DataView(moov.buffer,moov.byteOffset+i-4,4).getUint32(0),end=Math.min(moov.length,i-4+size);
  // The box holds a 16-bit string length and a 16-bit language code before the text.
  const length=new DataView(moov.buffer,moov.byteOffset+i+4,2).getUint16(0),text=new TextDecoder().decode(moov.subarray(i+8,Math.min(end,i+8+length)));
  const location=parseIso6709(text);if(location)return location;
 }
 return null;
}

function textBox(moov,name){
 const tag=[0xa9,...name].map(c=>typeof c==='number'?c:c.charCodeAt(0));
 for(let i=4;i+8<=moov.length;i++){
  if(tag.some((value,k)=>moov[i+k]!==value))continue;
  const length=new DataView(moov.buffer,moov.byteOffset+i+4,2).getUint16(0);
  return new TextDecoder().decode(moov.subarray(i+8,i+8+length)).replace(/\0+$/,'');
 }
 return null;
}

// The position, plus the camera heading (direction of the image top) and gimbal pitch when present.
export function videoMetadata(moov){
 const location=findLocation(moov);if(!location)return null;
 const number=name=>{const value=Number.parseFloat(textBox(moov,name)??'');return Number.isFinite(value)?value:null};
 const yaw=number('fyw'),gimbalYaw=number('gyw'),pitch=number('gpt');
 return {...location,...(yaw!==null?{heading_deg:(((yaw+(gimbalYaw??0))%360)+360)%360}:{}),...(pitch!==null?{gimbal_pitch_deg:pitch}:{})};
}

export async function videoLocation(file){
 try{const moov=await moovBox(file);return moov?videoMetadata(moov):null}catch{return null}
}
