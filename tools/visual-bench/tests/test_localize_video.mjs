import assert from 'node:assert/strict';
import {localizeVideo} from '../webapp/localize-video.js';

let now=0,presented=0;
const video={videoWidth:1920,videoHeight:1080,ended:false,currentTime:0,async play(){},pause(){},addEventListener(){},removeEventListener(){},
 requestVideoFrameCallback(fn){setTimeout(()=>{if(video.ended)return;presented++;video.currentTime=presented/30;fn(now,{mediaTime:presented/30,presentedFrames:presented})},0)}};
const calls=[],stop=new AbortController();let closed=false;
const pipeline={async initialize(pack,camera){calls.push(['initialize',pack.pack_id,camera.width])},async prepareReferences(prior){calls.push(['prepare',prior.latitude])},
 async beginSequence(){calls.push(['begin'])},async finishSequence(){calls.push(['finish']);return [{group:'last'}]},close(){closed=true},
 async estimate(_observation,_prior,index){now+=250;if(index>=4)stop.abort();return index===2?{decision:'search_deferred',candidate_hypotheses:[]}:{decision:'relative_tracking',candidate_hypotheses:[{tracking_supported:true}]}}};
const results=[];
const summary=await localizeVideo({pack:{pack_id:'area'},source:video,prior:{latitude:40.5442,longitude:-74.4564,radius_m:500,agl_m:110},signal:stop.signal,
 createPipeline:()=>pipeline,capture:(_video,_camera,time)=>({gray:new Uint8Array(4),width:2,height:2,time}),onResult:(result,time)=>results.push([result.decision,time])});
assert.deepEqual(calls.map(c=>c[0]),['initialize','prepare','begin','finish'],'the pipeline loads models and reference features before the first frame and stores the last image group after a stop');
assert.deepEqual(summary.image_track_groups,[{group:'last'}],'the stored image groups reach the caller');
assert.equal(calls[0][2],960,'the camera model follows the source size and matching detail');
assert.equal(summary.processed,5);assert.equal(results.length,4,'deferred frames are not reported');
assert.ok(closed,'the pipeline is released when the run ends');
console.info('A live stream or video element can be localized through one library call');
