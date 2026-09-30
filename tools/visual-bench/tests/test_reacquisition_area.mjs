import assert from 'node:assert/strict';
import {LocalizationPipeline,searchBudget} from '../webapp/localization.js';
import {localPosition} from '../webapp/geography.js';

const pipeline=new LocalizationPipeline({},{});
pipeline.pack={anchor_lat_lon:[40.5,-74.4]};pipeline.references={elevation:()=>20};
const prior={latitude:40.5,longitude:-74.4,radius_m:500,agl_m:110};
const seed={position_enu_m:[...localPosition(pipeline.pack,40.501,-74.399,0).slice(0,2),80]};

pipeline.temporal.previous={capture_time_ns:10e9};
assert.deepEqual(pipeline.reacquisitionArea(prior,[],12e9),prior,'without a supported pose the whole prior is searched');
const near=pipeline.reacquisitionArea(prior,[seed],12e9);
assert.equal(near.radius_m,90,'40 m plus 25 m/s for two seconds');
assert.ok(Math.abs(near.latitude-40.501)<1e-9&&Math.abs(near.longitude+74.399)<1e-9,'centred on the last pose');
assert.equal(near.agl_m,110,'the crop plan keeps the prior height');
assert.deepEqual(pipeline.reacquisitionArea(prior,[seed],60e9),prior,'a search that reaches the prior radius uses the whole prior');
const balanced={shortlist:160,candidates:8};
assert.equal(searchBudget(balanced,596,596),160,'the whole prior keeps the full budget');
assert.equal(searchBudget(balanced,319,596),86,'a bounded search keeps the pairs per crop of the whole prior');
assert.equal(searchBudget(balanced,40,596),24,'a small area still fills the refined candidate set');
assert.equal(searchBudget(balanced,319,undefined),160,'without a search of the whole prior the full budget applies');
console.info('Re-acquisition searches around the last supported pose');
