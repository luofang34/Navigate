import assert from 'node:assert/strict';
import {videoLocation,parseIso6709} from '../webapp/video-location.js';

const box=(type,...parts)=>{const body=Buffer.concat(parts),head=Buffer.alloc(8);head.writeUInt32BE(8+body.length);head.write(type,4,'latin1');return Buffer.concat([head,body])};
const xyz=text=>{const value=Buffer.from(text),head=Buffer.alloc(4);head.writeUInt16BE(value.length);head.writeUInt16BE(0x15c7,2);return box('\xa9xyz',head,value)};
const file=(...boxes)=>new Blob([Buffer.concat(boxes)]);

const dji=file(box('ftyp',Buffer.from('isom')),box('mdat',Buffer.alloc(4096)),box('moov',box('mvhd',Buffer.alloc(100)),box('udta',xyz('+40.5442-74.4564'))));
assert.deepEqual(await videoLocation(dji),{latitude:40.5442,longitude:-74.4564},'the recording position comes from the moov box after the media data');
assert.deepEqual(parseIso6709('+35.6762+139.6503+040.000/'),{latitude:35.6762,longitude:139.6503,altitude_m:40},'an ISO 6709 altitude is kept');
assert.equal(await videoLocation(file(box('ftyp',Buffer.from('isom')),box('moov',box('mvhd',Buffer.alloc(8))))),null,'a video without a location has no prior');
assert.equal(await videoLocation(file(box('moov',box('udta',xyz('+00.0000+000.0000'))))),null,'a zero location is treated as missing');
assert.equal(await videoLocation(new Blob([Buffer.from('not a video')])),null,'an unreadable file has no prior');
console.info('Video files supply their recorded position as the search prior');
