import {downloadFiles,read,sha256} from '../storage.js';
import {CampIndex} from './camp-index.js';
import {CampRetriever} from './camp.js';
export async function openCampRetriever(manifest,shared,progress){
 const {model,descriptors}=manifest.files;
 if(model.sha256!==manifest.index.model_sha256)throw Error('CAMP index uses a different model');
 await downloadFiles([model,descriptors],(n,total)=>progress(`Preparing image search ${(n/1048576).toFixed(1)} / ${(total/1048576).toFixed(1)} MB`));
 const load=async file=>{const bytes=await read(`pilotage://chunks/${file.sha256}.bin`,0,file.size);if(await sha256(bytes)!==file.sha256)throw Error('CAMP stored file checksum differs');return new Uint8Array(bytes)};
 const data=await load(descriptors);
 if(data.byteLength%4)throw Error('Incomplete CAMP descriptor value');
 const values=new Float32Array(data.byteLength/4),view=new DataView(data.buffer,data.byteOffset,data.byteLength);for(let i=0;i<values.length;i++)values[i]=view.getFloat32(i*4,true);
 progress('Preparing reference search…');return CampRetriever.create(await load(model),new CampIndex(manifest.index,values),shared);
}
