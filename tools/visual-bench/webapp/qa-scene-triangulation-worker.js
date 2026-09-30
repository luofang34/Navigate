import init,{triangulate_scene_points} from './wasm/navigate_visual_preview.js';
import {sha256} from './storage.js';
self.onmessage=async()=>{try{
 const manifest=await(await fetch('http://127.0.0.1:61678/triangulation-input-manifest.json')).json(),sources={};
 for(const [key,file] of Object.entries(manifest)){self.postMessage({progress:`Loading verified ${key}…`});const bytes=await(await fetch(file.url)).arrayBuffer();if(bytes.byteLength!==file.size||await sha256(bytes)!==file.sha256)throw Error('Source checksum changed');sources[key]=JSON.parse(new TextDecoder().decode(bytes))}
 const camera=JSON.stringify(sources.graph.camera),graph=JSON.stringify(sources.graph.graph),scene=JSON.stringify({...sources.refinement.result.scene,points:[]});
 await init();self.postMessage({progress:'Triangulating conditional scene points…'});const start=performance.now(),raw=triangulate_scene_points(camera,graph,scene),elapsed_ms=performance.now()-start,result=JSON.parse(raw);
 if(result.geographic_acceptance!==false||result.scene.cameras.length!==sources.refinement.result.scene.cameras.length||result.scene.cameras.some((c,i)=>{const source=sources.refinement.result.scene.cameras[i];return c.observation_sha256!==source.observation_sha256||c.fixed!==source.fixed||c.position_scene_units.some((v,j)=>v!==source.position_scene_units[j])||c.eye_to_scene_xyzw.some((v,j)=>v!==source.eye_to_scene_xyzw[j])}))throw Error('Triangulation changed a camera');
 const report={...result,stage:'browser_scene_triangulation',graph_sha256:manifest.graph.sha256,refinement_sha256:manifest.refinement.sha256,elapsed_ms,timing_scope:'WASM call including input decoding and output encoding',execution:'Rust/WASM in a dedicated browser worker'};
 await fetch('http://127.0.0.1:61678/result',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(report)});
 self.postMessage({result:{...report,scene:undefined,unresolved_feature_ids:undefined,points:result.scene.points.length,unresolved:result.unresolved_feature_ids.length}});
}catch(error){self.postMessage({error:String(error)+'\n'+error.stack})}};
