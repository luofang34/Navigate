import assert from 'node:assert/strict';
import {cropPlan} from '../webapp/crop-plan.js';
import {matchingOptions,headingAngles} from '../webapp/matching-options.js';
import {cameraForImage} from '../webapp/calibration.js';
import {rotationGeometry} from '../webapp/observation.js';
const fast=matchingOptions('fast'),detail=matchingOptions('balanced');
const a=cameraForImage(3840,2160,82.1,fast.longEdge),b=cameraForImage(3840,2160,82.1,detail.longEdge);
assert.ok(b.width>a.width);assert.ok(Math.abs(b.fx/a.fx-b.width/a.width)<1e-12);assert.ok(Math.abs(b.fy/a.fy-b.height/a.height)<1e-12);
const plans=cropPlan({width:1024,height:1024,baseSize:1500,center:[512,512],metresPerPixel:.5,radius:500,scales:detail.scales});
assert.equal(new Set(plans.map(p=>p.scale)).size,5);
assert.ok(plans.some(p=>p.x<0&&p.y<0),'a crop larger than the package uses a masked border');
assert.ok(plans.length<=160);

const headings=headingAngles(detail.headings);
for(let angle=0;angle<360;angle++){
  const error=Math.min(...headings.map(h=>Math.abs(((angle-h+540)%360)-180)));
  assert.ok(error<=2.5,'the standard search covers headings within two and a half degrees');
}
for(const mode of ['balanced','detailed']){
  const dense=headingAngles(matchingOptions(mode).headings);
  for(const angle of [...headingAngles(fast.headings),...headingAngles(36)])assert.ok(dense.includes(angle),'denser profiles retain coarse and ten-degree orientations');
}
for(const count of [0,3,73,NaN,4.5])assert.throws(()=>headingAngles(count),/heading search/);
for(const angle of [0,10,45,90,180,270,350]){
  const transform=rotationGeometry(640,360,angle);
  const center=transform.unrotate([(transform.width-1)/2,(transform.height-1)/2]);
  assert.deepEqual(center,[319.5,179.5],'rotation preserves the pixel-centre convention');
}
const quarterTurn=rotationGeometry(8,16,90);assert.equal(quarterTurn.width,16);assert.equal(quarterTurn.height,8);
for(const [target,expected] of [[[0,7],[0,0]],[[15,0],[7,15]]]){
  const actual=quarterTurn.unrotate(target);assert.ok(actual.every((v,i)=>Math.abs(v-expected[i])<1e-10),'rotated corner pixels map to the source corners');
}
console.info('Wide scale and heading search with correct rotated pixel centres passed');
