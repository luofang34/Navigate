// The caller verifies the file hash before it commits the downloaded bytes.
export async function* streamFile(url,size,{fetcher=fetch,rangeBytes=4*1024*1024}={}){
 if(!Number.isSafeInteger(size)||size<=0||!Number.isSafeInteger(rangeBytes)||rangeBytes<=0)throw Error('Invalid download size');
 for(let start=0;start<size;){
  const end=Math.min(size,start+rangeBytes)-1,ranged=size>rangeBytes;
  const response=await fetcher(url,{cache:'no-store',...(ranged?{headers:{Range:`bytes=${start}-${end}`}}:{})});
  const full=response.status===200&&start===0;
  const partial=ranged&&response.status===206&&response.headers.get('Content-Range')===`bytes ${start}-${end}/${size}`;
  if(!response.body||(!full&&!partial)){await response.body?.cancel();throw Error(`Invalid download response: ${response.status} at byte ${start}`)}
  const expected=full?size:end-start+1;
  yield* readBody(response.body,expected);
  start+=expected;
 }
}

async function* readBody(body,expected){
 const reader=body.getReader();let size=0,complete=false;
 try{
  while(true){
   const {value,done}=await reader.read();
   if(done){complete=true;break}
   size+=value.byteLength;if(size>expected)throw Error('Download exceeds expected response size');
   yield value;
  }
  if(size!==expected)throw Error(`Truncated download: ${size} of ${expected} bytes`);
 }finally{
  try{if(!complete)await reader.cancel()}finally{reader.releaseLock()}
 }
}
