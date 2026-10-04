import {BrowserPipeline} from './browser-pipeline.js';
import {runRealtime,loadedMetadata} from './realtime-video.js';
import {cameraForImage} from './calibration.js';
import {matchingOptions} from './matching-options.js';

// Library entry point: localize a live camera stream or a playing video element against an area
// package that is already stored offline. `prior` uses the app's prior fields (latitude, longitude,
// radius_m, agl_m, and an optional heading_deg). Each estimate with a pose or a search result reaches
// `onResult`; deferred frames that only wait for tracking to recover are not reported. `capture`
// replaces the default canvas grab when a host supplies its own grayscale frames. The returned summary
// includes the stored image track groups.
export async function localizeVideo({pack,source,prior,quality='balanced',fovDeg=82.1,signal,capture,onResult=()=>{},onProgress=()=>{},createPipeline=()=>new BrowserPipeline()}){
  const video=typeof MediaStream!=='undefined'&&source instanceof MediaStream?streamVideo(source):source;
  if(!video.videoWidth)await loadedMetadata(video,'The video source cannot be decoded');
  const options=matchingOptions(quality),camera=cameraForImage(video.videoWidth,video.videoHeight,fovDeg,options.longEdge),pipeline=createPipeline();
  try{
    await pipeline.initialize(pack,camera,onProgress,options);
    await pipeline.prepareReferences(prior,onProgress);
    await pipeline.beginSequence();
    const run=await runRealtime({video,camera,signal,...(capture?{capture}:{}),estimate:async(observation,index)=>{
      const result=await pipeline.estimate(observation,prior,index,onProgress);
      if(result.decision!=='search_deferred')onResult(result,observation.time);
      return result;
    }});
    // A stop or the end of the source still stores the partial last image group.
    return {...run.summary,image_track_groups:await pipeline.finishSequence()};
  }finally{pipeline.close()}
}

function streamVideo(stream){
  const video=document.createElement('video');video.muted=true;video.playsInline=true;video.srcObject=stream;return video;
}
