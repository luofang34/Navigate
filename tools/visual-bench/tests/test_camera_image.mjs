import assert from 'node:assert/strict';
import {openInput,originalPixels,consumeOriginalPixels} from '../webapp/observation.js';
import {decodeCameraImage} from '../webapp/camera-image.js';
const encoded={name:'camera.png'},rawRgb=[35,49,24],displayRgb=[23,38,11];
const decoder=async(source,options)=>{assert.strictEqual(source,encoded);return options?.colorSpaceConversion==='none'?rawRgb:displayRgb};
assert.deepEqual(await decodeCameraImage(encoded,decoder),rawRgb,'camera samples bypass display color conversion');
const failure=Error('invalid camera image');await assert.rejects(decodeCameraImage(encoded,async()=>{throw failure}),error=>error===failure);
console.info('Camera decode requests raw RGB samples and preserves decoder errors');

const originalDecoder=globalThis.createImageBitmap,originalUrl=URL.createObjectURL;
let closed=0;
try{
 URL.createObjectURL=()=>{throw Error('Image decoding must not allocate a video URL')};
 globalThis.createImageBitmap=async(_source,options)=>{assert.equal(options.colorSpaceConversion,'none');return {close(){closed++}}};
 const input=await openInput({type:'image/png'});assert.equal(input.type,'image');input.close();assert.equal(closed,1);
 globalThis.createImageBitmap=async()=>{throw failure};await assert.rejects(openInput({type:'image/png'}),error=>error===failure);
}finally{globalThis.createImageBitmap=originalDecoder;URL.createObjectURL=originalUrl}
console.info('Photo uploads release decoded bitmaps and do not leak object URLs on decoder failure');

const originalCanvas=globalThis.OffscreenCanvas,draws=[];
try{
 globalThis.OffscreenCanvas=class {constructor(width,height){this.width=width;this.height=height}getContext(){return {drawImage:(source,_x,_y,width,height)=>draws.push({source,width,height}),getImageData:()=>({data:Uint8ClampedArray.from({length:this.width*this.height*4},(_,i)=>[10,20,30,255][i%4])})}}};
 const source={width:4,height:3},pixels=originalPixels(source);assert.deepEqual(draws,[{source,width:4,height:3}]);assert.equal(pixels.rgb.length,36);assert.deepEqual([...pixels.rgb.subarray(0,6)],[10,20,30,10,20,30]);assert.equal(pixels.width,4);assert.equal(pixels.height,3);
 assert.throws(()=>originalPixels({width:0,height:3}),/dimensions/);
 let released=0;const bitmap={width:4,height:3,close(){released++}},observation={original:{bitmap}};consumeOriginalPixels(observation);assert.equal(released,1);assert.equal(observation.original.rgb.length,36);assert.equal('bitmap' in observation.original,false);
 const invalid={original:{bitmap:{width:0,height:1,close(){released++}}}};assert.throws(()=>consumeOriginalPixels(invalid),/dimensions/);assert.equal(released,2,'failed original-pixel reads release the bitmap');
}finally{globalThis.OffscreenCanvas=originalCanvas}
