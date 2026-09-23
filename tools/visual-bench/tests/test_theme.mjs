import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
const source=readFileSync(new URL('../webapp/theme.js',import.meta.url),'utf8');
function page(stored,dark,blocked=false){
 const document={documentElement:{dataset:{}},addEventListener:(name,fn)=>document[name]=fn,getElementById:()=>select};
 const select={value:'',addEventListener:(name,fn)=>select[name]=fn};
 const media={matches:dark,addEventListener:(name,fn)=>media[name]=fn};
 const storage={getItem:()=>{if(blocked)throw Error('Storage unavailable');return stored},setItem:(key,value)=>{if(blocked)throw Error('Storage unavailable');stored=value}};
 vm.runInNewContext(source,{document,localStorage:storage,matchMedia:()=>media});document.DOMContentLoaded();
 return {document,select,media,stored:()=>stored,theme:()=>document.documentElement.dataset.theme};
}
const system=page(null,true);assert.equal(system.theme(),'dark');system.media.matches=false;system.media.change();assert.equal(system.theme(),'light');
system.select.value='dark';system.select.change();assert.equal(system.theme(),'dark');assert.equal(system.stored(),'dark');system.media.change();assert.equal(system.theme(),'dark');
assert.equal(page(system.stored(),false).theme(),'dark');assert.equal(page('light',true).theme(),'light');
const blocked=page(null,true,true);blocked.select.value='light';blocked.select.change();assert.equal(blocked.theme(),'light');
assert.equal(page('invalid',true).select.value,'system');
console.log('PASS: system theme, manual override, reload, OS changes, unavailable storage');
