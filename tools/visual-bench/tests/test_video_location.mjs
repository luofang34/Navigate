import assert from 'node:assert/strict';
import {videoLocation,parseIso6709} from '../webapp/video-location.js';

const box=(type,...parts)=>{const body=Buffer.concat(parts),head=Buffer.alloc(8);head.writeUInt32BE(8+body.length);head.write(type,4,'latin1');return Buffer.concat([head,body])};
const text=(name,value)=>{const bytes=Buffer.from(value+'\0'),head=Buffer.alloc(4);head.writeUInt16BE(bytes.length);head.writeUInt16BE(0x15c7,2);return box('\xa9'+name,head,bytes)};
const xyz=value=>text('xyz',value);
const file=(...boxes)=>new Blob([Buffer.concat(boxes)]);

const dji=file(box('ftyp',Buffer.from('isom')),box('mdat',Buffer.alloc(4096)),box('moov',box('mvhd',Buffer.alloc(100)),box('udta',xyz('+40.5442-74.4564'))));
assert.deepEqual(await videoLocation(dji),{latitude:40.5442,longitude:-74.4564},'the recording position comes from the moov box after the media data');
const attitude=file(box('ftyp',Buffer.from('isom')),box('mdat',Buffer.alloc(64)),box('moov',box('udta',xyz('+40.5442-74.4564'),text('fyw','-110.00'),text('gyw','-3.40'),text('gpt','-90.00'))));
const found=await videoLocation(attitude);
assert.ok(Math.abs(found.heading_deg-246.6)<1e-9,'the camera heading is the aircraft yaw plus the gimbal yaw');
assert.equal(found.gimbal_pitch_deg,-90,'the gimbal pitch is reported');
assert.deepEqual(parseIso6709('+35.6762+139.6503+040.000/'),{latitude:35.6762,longitude:139.6503,altitude_m:40},'an ISO 6709 altitude is kept');
assert.equal(await videoLocation(file(box('ftyp',Buffer.from('isom')),box('moov',box('mvhd',Buffer.alloc(8))))),null,'a video without a location has no prior');
assert.equal(await videoLocation(file(box('moov',box('udta',xyz('+00.0000+000.0000'))))),null,'a zero location is treated as missing');
assert.equal(await videoLocation(new Blob([Buffer.from('not a video')])),null,'an unreadable file has no prior');
console.info('Video files supply their recorded position as the search prior');
