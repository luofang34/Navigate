const worker=new Worker('./qa-xfeat-worker.js',{type:'module'});
worker.onmessage=async({data})=>{document.querySelector('pre').textContent=JSON.stringify(data,null,2);if(data.report){document.title=data.report.error?'FAIL XFeat':'XFeat diagnostics';await fetch('/__test__/report',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data.report)});worker.terminate()}};
worker.onerror=e=>document.querySelector('pre').textContent=e.message;
