import {cp,mkdir,readFile,readdir,writeFile} from 'node:fs/promises';
import {resolve,dirname} from 'node:path';
import {fileURLToPath} from 'node:url';
const [state,output]=process.argv.slice(2);
if(!state||!output)throw Error('Usage: node export_web.mjs /path/to/package-state /path/to/site-output');
const destination=resolve(output),source=resolve(dirname(fileURLToPath(import.meta.url)),'webapp');
if(destination===source||destination.startsWith(source+'/'))throw Error('Export outside the source webapp directory');
await mkdir(destination,{recursive:true});
await cp(source,destination,{recursive:true,dereference:true,filter:path=>!/(?:\/qa[^/]*|\/models\/test-[^/]*)$/.test(path)});
for(const name of ['chunks','packs','catalog'])await cp(resolve(state,name),resolve(destination,name),{recursive:true});
const catalog=[];
for(const name of await readdir(resolve(state,'catalog')))if(name.endsWith('.json')){const {manifest,...region}=JSON.parse(await readFile(resolve(state,'catalog',name),'utf8'));catalog.push(region);}
await writeFile(resolve(destination,'catalog.json'),JSON.stringify(catalog));
