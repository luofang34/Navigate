import assert from 'node:assert/strict';
import {mkdtemp,mkdir,writeFile,readFile,readdir,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {createHash} from 'node:crypto';
import {exportSite} from '../export_web.mjs';
const root=await mkdtemp(join(tmpdir(),'vnav-export-test-')),source=join(root,'webapp'),state=join(root,'state');
const digest=b=>createHash('sha256').update(b).digest('hex'),data=Buffer.from('public imagery'),sha=digest(data),model=Buffer.from('public model'),id='a'.repeat(64);
try{
 for(const name of ['webapp/models','webapp/runtime','state/chunks','state/packs','state/catalog'])await mkdir(join(root,name),{recursive:true});
 await writeFile(join(source,'index.html'),'public shell');await writeFile(join(source,'models/xfeat.onnx'),model);
 await writeFile(join(source,'models/manifest.json'),JSON.stringify({xfeat:{sha256:digest(model),size:model.length}}));
 for(const name of ['models/superglue.onnx','models/test-input.png','qa-private.js'])await writeFile(join(source,name),'must not publish');
 await writeFile(join(state,'chunks',sha+'.bin'),data);await writeFile(join(state,'chunks','unused.bin'),'private');
 await writeFile(join(state,'catalog/demo.jpg'),'thumbnail');await writeFile(join(state,'catalog/demo.json'),JSON.stringify({id:'demo',pack_id:id,manifest:{private_path:'/private/file'}}));
 await writeFile(join(state,'catalog/private.json'),JSON.stringify({id:'private'}));
 await writeFile(join(state,'packs',id+'.json'),JSON.stringify({pack_id:id,region_id:'demo',provenance:{provider:'planetary-computer-naip'},files:[{sha256:sha,size:data.length,url:`/chunks/${sha}.bin`}]}));
 await assert.rejects(exportSite(state,join(root,'no-selection'),[],source),/explicit region/);
 const output=join(root,'site'),summary=await exportSite(state,output,['demo'],source);
 assert.deepEqual(summary,{regions:1,chunks:1,package_bytes:data.length});
 assert.deepEqual((await readdir(join(output,'models'))).sort(),['manifest.json','xfeat.onnx']);
 assert.deepEqual(await readdir(join(output,'chunks')),[sha+'.bin']);assert.equal((await readdir(output)).includes('qa-private.js'),false);
 const catalog=JSON.parse(await readFile(join(output,'catalog.json')));assert.equal(catalog.length,1);assert.equal(catalog[0].manifest,undefined);
 await assert.rejects(exportSite(state,output,['demo'],source),/empty output/);
 await writeFile(join(state,'chunks',sha+'.bin'),'tampered');await assert.rejects(exportSite(state,join(root,'tampered'),['demo'],source),/checksum mismatch/);
 console.info('Public export selection, model allowlist, private fixture exclusion and checksum checks passed');
}finally{await rm(root,{recursive:true,force:true})}
