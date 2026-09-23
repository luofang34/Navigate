import http from 'node:http';
import {writeFile} from 'node:fs/promises';
const [upstream,result,port='0']=process.argv.slice(2);
if(!upstream||!result)throw Error('Usage: node browser_server.mjs http://127.0.0.1:service-port /path/to/report.json [port]');
const target=new URL(upstream);
if(target.hostname!=='127.0.0.1')throw Error('Test upstream must use loopback');
const server=http.createServer(async(request,response)=>{
  try {
    if(request.url==='/__test__/report'&&request.method==='POST') {
      let text='';for await(const chunk of request){text+=chunk;if(text.length>1048576)throw Error('Test report is too large');}
      await writeFile(result,JSON.stringify(JSON.parse(text),null,2));response.writeHead(200,{'Content-Type':'application/json'}).end('{}');return;
    }
    const headers={...request.headers,host:target.host};if(headers.origin)headers.origin=target.origin;
    const proxy=http.request({hostname:target.hostname,port:target.port,path:request.url,method:request.method,headers},remote=>{response.writeHead(remote.statusCode,remote.headers);remote.pipe(response)});
    proxy.on('error',error=>response.writeHead(502).end(String(error)));request.pipe(proxy);
  } catch(error){response.writeHead(400).end(String(error));}
});
server.listen(Number(port),'127.0.0.1',()=>console.info(JSON.stringify({url:`http://127.0.0.1:${server.address().port}`,pid:process.pid})));
