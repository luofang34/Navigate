import assert from 'node:assert/strict';
import {LocalizationPipeline,searchBudget,cameraHeading} from '../webapp/localization.js';
import {localPosition} from '../webapp/geography.js';

const pipeline=new LocalizationPipeline({},{});
pipeline.pack={anchor_lat_lon:[40.5,-74.4]};pipeline.references={elevation:()=>20};
const prior={latitude:40.5,longitude:-74.4,radius_m:500,agl_m:110};
const seed={position_enu_m:[...localPosition(pipeline.pack,40.501,-74.399,0).slice(0,2),80],eye_to_enu_xyzw:[0,0,Math.sin(Math.PI/4),Math.cos(Math.PI/4)]};


assert.deepEqual(pipeline.reacquisitionArea(prior,12e9),prior,'without a supported pose the whole prior is searched');
pipeline.temporal.lastPose={...seed,capture_time_ns:10e9};
const near=pipeline.reacquisitionArea({...prior,heading_deg:90},11e9);
assert.equal(near.radius_m,65,'40 m plus 25 m/s for one second');
assert.ok(Math.abs(near.latitude-40.501)<1e-9&&Math.abs(near.longitude+74.399)<1e-9,'centred on the last pose');
assert.equal(near.agl_m,110,'the crop plan keeps the prior height');
assert.ok(Math.abs(near.heading_deg-270)<1e-9,'the heading of the last supported pose replaces the starting heading');
assert.equal(near.heading_tolerance_deg,90,'the heading tolerance grows by 60 degrees per second');
assert.equal(pipeline.reacquisitionArea({...prior,heading_deg:90},13e9).heading_deg,undefined,'after a few seconds every rotation is searched');
assert.ok(Math.abs(cameraHeading([0,0,0,1]))<1e-9,'an unrotated eye frame has its image top to the north');
const late=pipeline.reacquisitionArea({...prior,heading_deg:90},60e9);
assert.equal(late.radius_m,500,'a search never covers more than the prior');assert.equal(late.heading_deg,undefined,'after a long loss neither heading is trusted');
const balanced={shortlist:160,candidates:8};
assert.equal(searchBudget(balanced,596,596),160,'the whole prior keeps the full budget');
assert.equal(searchBudget(balanced,319,596),86,'a bounded search keeps the pairs per crop of the whole prior');
assert.equal(searchBudget(balanced,40,596),24,'a small area still fills the refined candidate set');
assert.equal(searchBudget(balanced,319,undefined),160,'without a search of the whole prior the full budget applies');
console.info('Re-acquisition searches around the last supported pose');
