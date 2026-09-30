import assert from 'node:assert/strict';
import {distinctPoses,TemporalSearch} from '../webapp/temporal-search.js';

const level=[0,0,0,1],turned=[0,0,Math.sin(Math.PI/8),Math.cos(Math.PI/8)];
const h=(id,position,support,attitude=level)=>({candidate_id:id,accepted:true,position_enu_m:position,eye_to_enu_xyzw:attitude,spatial_support:support});
const cluster=[h(0,[509,795,118],114),h(1,[511,799,117],74),h(2,[510,796,112],132),h(3,[504,800,115],118)];
assert.deepEqual(distinctPoses(cluster).map(c=>c.candidate_id),[2],'near-identical poses keep the best supported one');
assert.deepEqual(distinctPoses([...cluster,h(4,[700,795,118],50)]).map(c=>c.candidate_id),[2,4],'a distant alternative stays tracked');
assert.deepEqual(distinctPoses([h(0,[0,0,100],10),h(1,[2,0,100],5,turned)]).map(c=>c.candidate_id),[0,1],'a different heading at the same place stays tracked');

const temporal=new TemporalSearch();
temporal.remember({observation_sha256:'a',sequence:0,capture_time_ns:0,decision:'unresolved',candidate_hypotheses:cluster},null);
assert.equal(temporal.previous.candidates.length,1,'the next frame tracks one seed per distinct camera pose');
console.info('Near-identical hypotheses merge before tracking; distinct alternatives remain');
