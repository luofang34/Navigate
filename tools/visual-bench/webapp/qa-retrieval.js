import {checkGpuRetrieval,checkBatchedGpuRetrieval} from './qa-gpu.js';
const report={test:'batched-gpu-retrieval',started_at:new Date().toISOString()};
try {
 report.known_correspondences=await checkGpuRetrieval();
 report.cases=await checkBatchedGpuRetrieval();report.passed=true;
}catch(error){report.error=String(error);report.passed=false}
report.finished_at=new Date().toISOString();document.title=report.passed?'PASS GPU retrieval parity':'FAIL GPU retrieval parity';document.querySelector('pre').textContent=JSON.stringify(report,null,2);
await fetch('/__test__/report',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(report)});
