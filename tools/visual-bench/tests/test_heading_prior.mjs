import assert from 'node:assert/strict';
import {searchAngles,headingAngles} from '../webapp/matching-options.js';

assert.deepEqual(searchAngles(72),headingAngles(72),'without a heading every rotation is searched');
assert.deepEqual(searchAngles(72,260),[70,75,80,85,90,95,100,105,110,115,120,125,130],'a 260 degree heading searches rotations within 30 degrees of 100');
assert.deepEqual(searchAngles(72,10,10),[0,340,345,350,355],'the rotation window wraps through north');
assert.deepEqual(searchAngles(72,90,180),headingAngles(72),'a tolerance of half a turn searches every rotation');
console.info('A camera heading limits the area search to nearby image rotations');
