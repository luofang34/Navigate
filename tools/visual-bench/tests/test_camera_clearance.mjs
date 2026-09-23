import assert from 'node:assert/strict';
import {constrainCamera,minimumClearance,globeLocation} from '../webapp/camera-clearance.js';
import {toGlobePose} from '../webapp/geography.js';
const pack={anchor_lat_lon:[40.54,-74.45]};
for(const [east,north] of [[0,0],[3000,9000],[-20000,5000]]){
  const proposed=toGlobePose(pack,{position_enu_m:[east,north,50],eye_to_enu_xyzw:[0,0,0,1]}),original=structuredClone(proposed);
  const result=constrainCamera(pack,proposed,()=>125,20),location=globeLocation(pack,result.pose),before=globeLocation(pack,proposed);
  assert.ok(Math.abs(location.altitude-145)<1e-6);assert.ok(Math.abs(location.latitude-before.latitude)<1e-9);assert.ok(Math.abs(location.longitude-before.longitude)<1e-9);
  assert.equal(result.known,true);assert.ok(Math.abs(result.agl-20)<1e-6);assert.deepEqual(proposed,original,'estimated pose is never changed');
  const missing=constrainCamera(pack,proposed,()=>undefined);assert.equal(missing.known,false);assert.equal(missing.agl,null);assert.ok(missing.altitude>=9999.99,'missing terrain is not zero elevation');
}
assert.equal(minimumClearance(110),20);assert.equal(minimumClearance(800),80);assert.equal(minimumClearance(NaN),20);assert.equal(minimumClearance(10000),100);
