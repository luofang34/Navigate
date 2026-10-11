"""Record completed matching evidence used to select fallback queries."""
import json

from photo_testset import digest


def resolved_inputs(runs):
    resolved=set();inputs={}
    for run in runs:
        reports={}
        for path in sorted(run.glob('q-*.json')):
            result=json.loads(path.read_text())
            candidate=result.get('candidate')
            evidence={'report_sha256':digest(path)}
            if (result.get('status')=='candidate_requires_review' and candidate
                    and candidate.get('score',{}).get('geometry_accepted') is True):
                state=run/result['id']/'candidate.npz'
                evidence['state_sha256']=digest(state)
                resolved.add(result['id'])
            reports[path.name]=evidence
        inputs[str(run.resolve())]={'run_sha256':digest(run/'run.json'),'reports':reports}
    return resolved,inputs
