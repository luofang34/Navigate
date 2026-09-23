import test from 'node:test';
import assert from 'node:assert/strict';
import {checkActions} from './check-actions.mjs';
const action=name=>`actions/${name}@${'a'.repeat(40)}`;
const uses=name=>`- uses: ${action(name)} # pinned`;
test('follows composite dependencies and rejects Node.js 20 with its chain',async()=>{
 const metadata={pages:`runs:\n  using: composite\n  steps:\n    ${uses('upload')}`,upload:"runs:\n  using: 'node20'"};
 await assert.rejects(checkActions([uses('pages')],async key=>metadata[key.split('/')[1].split('@')[0]]),/node20: actions\/pages@.* -> actions\/upload@/);
});
test('accepts Node.js 24 and visits a shared dependency once',async()=>{
 const seen=[];
 const result=await checkActions([uses('pages'),uses('upload')],async key=>{seen.push(key);return key===action('pages')?`using: composite\n${uses('upload')}`:'using: node24';});
 assert.deepEqual(result,[action('pages'),action('upload')]);assert.equal(seen.length,2);
});
test('rejects mutable references and unknown runtimes',async()=>{
 await assert.rejects(checkActions(['uses: actions/checkout@v7'],async()=>''),/immutable/);
 await assert.rejects(checkActions([uses('empty')],async()=>''),/Unsupported action runtime/);
});
test('retains metadata download failures',async()=>{
 await assert.rejects(checkActions([uses('pages')],async()=>{throw Error('HTTP 503')}),/HTTP 503/);
});
