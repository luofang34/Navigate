import assert from 'node:assert/strict';
import {readFileSync,mkdtempSync,mkdirSync,writeFileSync,rmSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {spawnSync} from 'node:child_process';
const workflow=readFileSync(new URL('../../../.github/workflows/vnav-pages.yml',import.meta.url),'utf8');
assert.ok(workflow.includes("if: github.ref == 'refs/heads/main' && github.event_name != 'pull_request'"));
const block=workflow.match(/name: Read the renderer revision[\s\S]*?run: \|\n((?: {10}[^\n]+\n)+)/);
assert.ok(block,'Pages must read the checked renderer revision');
const command=block[1].split('\n').map(line=>line.slice(10)).join('\n');
assert.ok(workflow.includes('ref: ${{ steps.renderer.outputs.revision }}'));
const root=mkdtempSync(join(tmpdir(),'navigate-renderer-pin-'));
try{
 mkdirSync(join(root,'tools/visual-bench'),{recursive:true});
 const pin=join(root,'tools/visual-bench/MAPLIBRE_REVISION'),output=join(root,'output');
 const run=()=>spawnSync('bash',['-e','-c',command],{cwd:root,env:{...process.env,GITHUB_OUTPUT:output},encoding:'utf8'});
 writeFileSync(pin,'  '+ 'a'.repeat(40)+'\n');assert.equal(run().status,0);
 assert.equal(readFileSync(output,'utf8'),'revision='+ 'a'.repeat(40)+'\n');
 for(const invalid of ['main','a'.repeat(39),'a'.repeat(40)+';true']){
  writeFileSync(pin,invalid);writeFileSync(output,'');assert.notEqual(run().status,0);assert.equal(readFileSync(output,'utf8'),'');
 }
 rmSync(pin);assert.notEqual(run().status,0);
 console.log('Renderer pin: valid immutable revision accepted; invalid and absent pins rejected');
}finally{rmSync(root,{recursive:true,force:true})}
