import {assetUrl} from './asset-url.js';
import {downloadFiles,read,sha256,get,put} from './storage.js';
import {IndexedReferenceSearch} from './indexed-reference-search.js';
import {openCampRetriever} from './inference/camp-storage.js';
export async function openReferenceSearch(pack,progress,shared){
 const key=`reference-index/${pack.pack_id}`;let manifest,response;try{response=await fetch(assetUrl(`retrieval/${pack.pack_id}.json`))}catch(error){manifest=await get('state',key);if(!manifest)throw error}
 if(response){if(response.status===404)return null;if(!response.ok)throw Error(`Reference index request failed: ${response.status}`);manifest=await response.json()}if(manifest.index.map_manifest_sha256!==pack.pack_id||manifest.index.map_release!==pack.release_id)throw Error('Reference index uses different map data');
 const file=manifest.files.catalog;if(file.sha256!==manifest.index.catalog_manifest_sha256)throw Error('Reference catalog identity differs');
 await downloadFiles([file],()=>progress('Reading reference index…'));const bytes=await read(`pilotage://chunks/${file.sha256}.bin`,0,file.size);if(await sha256(bytes)!==file.sha256)throw Error('Stored reference catalog checksum differs');
 const catalog=JSON.parse(new TextDecoder().decode(bytes));if(catalog.release!==pack.release_id)throw Error('Reference catalog uses different imagery');
 if(catalog.gallery.length!==manifest.index.ids.length||catalog.gallery.some((row,i)=>row.id!==manifest.index.ids[i]))throw Error('Reference catalog rows differ from the feature index');
 const retriever=await openCampRetriever(manifest,shared,progress);
 try{const search=new IndexedReferenceSearch(retriever,catalog.gallery,manifest.scale_order_m);await put('state',key,manifest);return search}catch(error){try{await retriever.close()}finally{throw error}}
}
