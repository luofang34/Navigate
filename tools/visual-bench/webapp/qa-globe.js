import {Preview} from './wasm/navigate_visual_preview.js';
import {assetUrl} from './asset-url.js';
import {read} from './storage.js';

export async function checkGlobeContext(pack,camera,pose){
  const preview=await Preview.create(JSON.stringify(pack),JSON.stringify(camera),read,true);
  try{
    const context=await(await fetch(assetUrl('context/earth.json'))).json();
    preview.set_globe_context(new Uint8Array(await(await fetch(assetUrl(context.url))).arrayBuffer()));
    const rgba=await preview.render(JSON.stringify(pose));let land=0,pixels=0;
    for(let y=Math.floor(camera.height*.4);y<camera.height*.6;y++)for(let x=Math.floor(camera.width*.4);x<camera.width*.6;x++){
      const i=(y*camera.width+x)*4,[red,green,blue]=rgba.subarray(i,i+3);
      if(red>50&&green>red+10&&green>blue+5)land++;
      pixels++;
    }
    return land>pixels*.25;
  }finally{preview.free()}
}
