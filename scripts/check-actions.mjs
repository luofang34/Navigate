import {readFile,readdir} from 'node:fs/promises';
import {resolve} from 'node:path';
import {pathToFileURL} from 'node:url';

export function actionUses(text){
  return [...text.matchAll(/^\s*(?:-\s*)?uses:\s*["']?([^\s"'#]+)["']?/gm)].map(match=>match[1]);
}

export async function checkActions(workflows,load){
  const checked=new Set();
  async function visit(action,chain){
    if(checked.has(action))return;
    if(!/^[\w.-]+\/[\w./-]+@[a-f0-9]{40}$/.test(action))throw Error(`Action must use an immutable revision: ${action}`);
    checked.add(action);
    const text=await load(action),runtime=text.match(/^\s*using:\s*["']?([\w]+)["']?/m)?.[1];
    const path=[...chain,action];
    if(runtime==='composite'){
      for(const child of actionUses(text))await visit(child,path);
    }else if(runtime!=='node24'&&runtime!=='docker'){
      throw Error(`Unsupported action runtime ${runtime}: ${path.join(' -> ')}`);
    }
  }
  for(const text of workflows)for(const action of actionUses(text))await visit(action,[]);
  return [...checked];
}

async function loadAction(action){
  const [location,ref]=action.split('@'),[owner,repo,...directory]=location.split('/');
  const base=`https://raw.githubusercontent.com/${owner}/${repo}/${ref}/${directory.length?directory.join('/')+'/':''}`;
  for(const name of ['action.yml','action.yaml']){
    const response=await fetch(base+name,{signal:AbortSignal.timeout(30_000)});
    if(response.status===404)continue;
    if(!response.ok)throw Error(`Read ${action}: HTTP ${response.status}`);
    return response.text();
  }
  throw Error(`Missing action metadata: ${action}`);
}

if(process.argv[1]&&import.meta.url===pathToFileURL(resolve(process.argv[1])).href){
  const directory=new URL('../.github/workflows/',import.meta.url);
  const files=(await readdir(directory)).filter(name=>/\.ya?ml$/.test(name));
  const workflows=await Promise.all(files.map(name=>readFile(new URL(name,directory),'utf8')));
  const actions=await checkActions(workflows,loadAction);
  console.log(`Verified ${actions.length} direct and transitive actions: Node.js 24, composite, or Docker.`);
}
