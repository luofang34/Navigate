import http from 'node:http';
import {access,readFile,writeFile} from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const [data,runtime,model,report]=process.argv.slice(2);
if(!data||!runtime||!model||!report)throw Error('Usage: node serve.mjs INPUT_DIRECTORY ORT_WEB_DIRECTORY MODEL.onnx NEW_REPORT.json');
try{await access(report);throw Error('Report already exists')}catch(error){if(error.code!=='ENOENT')throw error}
const page=path.dirname(fileURLToPath(import.meta.url));
const server=http.createServer(async(req,res)=>{
  try{
    if(req.url==='/report'&&req.method==='POST'){
      let text='';for await(const b of req){text+=b;if(text.length>100000)throw Error('Report too large')}
      await writeFile(report,JSON.stringify(JSON.parse(text),null,2),{flag:'wx'});res.end('{}');return;
    }
    const name=new URL(req.url,'http://localhost').pathname;
    let file;
    if(name==='/model.onnx')file=model;
    else if(name.startsWith('/runtime/'))file=path.join(runtime,path.basename(name));
    else if(name==='/'||name==='/index.html'||name==='/worker.js')file=path.join(page,name==='/worker.js'?'worker.js':'index.html');
    else if(name==='/cases.json'||/^\/[a-zA-Z0-9_-]+\.f32$/.test(name))file=path.join(data,path.basename(name));
    else throw Error('Unknown test resource');
    const body=await readFile(file);
    const type=file.endsWith('.html')?'text/html':/\.(mjs|js)$/.test(file)?'text/javascript':file.endsWith('.json')?'application/json':file.endsWith('.wasm')?'application/wasm':'application/octet-stream';
    res.writeHead(200,{'Content-Type':type,'Cross-Origin-Opener-Policy':'same-origin','Cross-Origin-Embedder-Policy':'require-corp','Cache-Control':'no-store'}).end(body);
  }catch(error){res.writeHead(400).end(String(error))}
});
server.listen(0,'127.0.0.1',()=>console.info(JSON.stringify({url:`http://127.0.0.1:${server.address().port}`,pid:process.pid})));
