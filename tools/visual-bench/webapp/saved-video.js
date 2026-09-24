import {replaySeekTime} from './video-timing.js';
// These checks associate playback with saved samples. They do not validate poses.
export async function checkSavedVideo(mission,{decode,savedPixels}){
 const frames=mission.view.frames;
 if(!frames.length||frames.some(f=>f.timing_scope==='still image'))throw Error('This observation is not a video sequence.');
 const indices=[...new Set([0,Math.floor(frames.length/2),frames.length-1])];
 for(const index of indices){
  const frame=frames[index],image=await decode(replaySeekTime(frame)),saved=await savedPixels(frame);
  if(!Number.isFinite(image.time)||!Number.isFinite(frame.capture_time_ns)||Math.abs(image.time-frame.capture_time_ns/1e9)>1e-5||image.gray.length!==saved.length||image.gray.some((value,i)=>value!==saved[i]))throw Error('This video does not match the saved samples. The saved track is unchanged.');
 }
 return indices.length;
}
