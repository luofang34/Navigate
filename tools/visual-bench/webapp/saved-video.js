import {replaySeekTime} from './video-timing.js';
// These checks associate playback with saved samples. They do not validate poses.
export async function checkSavedVideo(mission,{decode,savedPixels}){
 const frames=mission.view.frames;
 if(!frames.length||frames.some(f=>f.timing_scope==='still image'))throw Error('This observation is not a video sequence.');
 const indices=[...new Set([0,Math.floor(frames.length/2),frames.length-1])];
 for(const index of indices){
  const frame=frames[index],image=await decode(replaySeekTime(frame)),saved=await savedPixels(frame);
  if(!Number.isFinite(image.time)||!Number.isFinite(frame.capture_time_ns)||!matchingTime(frame,image)||image.gray.length!==saved.length||image.gray.some((value,i)=>value!==saved[i]))throw Error('This video does not match the saved samples. The saved track is unchanged.');
 }
 return indices.length;
}

function matchingTime(frame,image){
 const fallback='browser media seek time; decoded frame PTS is unavailable';
 const storedSeek=frame.timing_scope===fallback,decodedSeek=image.timing===fallback;
 const stored=frame.capture_time_ns/1e9,requested=replaySeekTime(frame);
 // A seek timestamp identifies the request. Exact pixels associate its decoded frame.
 if(storedSeek&&Math.abs(stored-requested)>1e-5)return false;
 if(decodedSeek&&Math.abs(image.time-requested)>1e-5)return false;
 return storedSeek||decodedSeek||Math.abs(image.time-stored)<=1e-5;
}
