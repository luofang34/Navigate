import assert from 'node:assert/strict';
import {displayGap} from '../webapp/pose-playback.js';
import {orderedInsert} from '../webapp/realtime-video.js';

const at=seconds=>({capture_time_ns:seconds*1e9});
assert.equal(displayGap([0,.27,.54,.81].map(at),.2).toFixed(2),'0.27','real-time samples connect at their own spacing');
assert.equal(displayGap([0,.27,2.27,5.27].map(at),.2).toFixed(2),'3.00','catch-up hops stay connected at their spacing');
assert.equal(displayGap([0,.2,.4].map(at),.5),.5,'a longer configured period is kept');
assert.equal(displayGap([0,30].map(at),.2),4,'a long gap never connects across more than one catch-up hop');
assert.equal(displayGap([],.2),.2);
const frames=[at(1),at(3)],blobs=['a','c'];
assert.equal(orderedInsert(frames,blobs,at(2),'b'),1,'a retried catch-up frame is placed in time order');
assert.deepEqual(blobs,['a','b','c']);
console.info('Real-time samples are ordered and connect at their observed spacing');
