export async function emitDatasetReport(report,phase,output,submit=fetch){
 report.phase=phase;
 const cases=report.cases.map(({acquisition_trace,retrieval_trace,hypotheses,...summary})=>({...summary,trace_comparisons:acquisition_trace?.length,retrieval_pairs:retrieval_trace?.ranks.length,evaluated_hypotheses:hypotheses?.length,accepted_candidate_ids:hypotheses?.filter(h=>h.accepted).map(h=>h.candidate_id)}));
 output.textContent=JSON.stringify({...report,cases},null,2);
 const response=await submit('/__test__/report',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(report)});
 if(!response.ok)throw Error('Diagnostic export failed: '+response.status);
}
