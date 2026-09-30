import {consumeOriginalPixels} from './observation.js';
self.onmessage=async({data:{id,method,args}})=>{try{
 if(method==='initialize'){self.postMessage({id,value:{original_pixels:true}});return}
 if(method!=='estimate')throw Error('Unexpected capture probe operation');
 const image=args[0];self.postMessage({id,progress:'Reading original camera pixels in the worker'});consumeOriginalPixels(image);
 const hash=[...new Uint8Array(await crypto.subtle.digest('SHA-256',image.original.rgb))].map(v=>v.toString(16).padStart(2,'0')).join('');
 self.postMessage({id,value:{original_rgb_sha256:hash,original_dimensions:[image.original.width,image.original.height],geometric_dimensions:[image.width,image.height]}});
}catch(error){self.postMessage({id,error:String(error)})}finally{if(method==='estimate')args[0]?.original?.bitmap?.close()}};
