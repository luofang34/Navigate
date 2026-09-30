import {LocalizationPipeline} from './localization.js';
import {LocalMatcher} from './inference/local.js';
import {ResearchLightGlue} from './qa-lightglue-adapter.js';
import {openReferenceSearch} from './reference-search-loading.js';

export async function createAcquisitionPipeline(pack,camera,options,progress,{
 createPublic=options=>new LocalMatcher(options),createResearch=()=>new ResearchLightGlue(),
 createPipeline=(matcher,options)=>new LocalizationPipeline(matcher,options),openSearch=openReferenceSearch,
}={}){
 const direct=options.referenceSearch&&options.traceMatching==='lightglue'&&!options.researchHybrid;
 const matcher=direct?createResearch():createPublic(options),pipeline=createPipeline(matcher,options);
 try{
  await pipeline.initialize(pack,camera,progress);
  if(!direct&&(options.traceMatching==='lightglue'||options.researchHybrid)){
   const research=createResearch(),baseClose=matcher.close.bind(matcher);
   matcher.close=()=>closeAdapters([research,{close:baseClose}]);
   await research.initialize(progress,matcher.matcher);
   const baseMatch=matcher.matchImages.bind(matcher);
   matcher.matchImages=(reference,query,keys={})=>options.researchHybrid&&keys.stage!=='refinement'?baseMatch(reference,query,keys):research.matchImages(reference,query,keys);
   matcher.diagnostics=()=>({...research.diagnostics(),matcher_policy:options.researchHybrid?'public proposals and tracking; research refinement':'research matching'});
  }
  if(options.referenceSearch){
   pipeline.referenceSearch=await openSearch(pack,progress,direct?matcher:matcher.matcher);
   if(!pipeline.referenceSearch)throw Error('No prepared reference index for this map');
  }
  return pipeline;
 }catch(error){
  let cleanupError;try{await closeAdapters([pipeline.referenceSearch,matcher])}catch(failure){cleanupError=failure}
  pipeline.renderer?.free();
  if(cleanupError)throw new AggregateError([error,cleanupError],'Acquisition initialization and cleanup failed');
  throw error;
 }
}

async function closeAdapters(adapters){
 const outcomes=await Promise.allSettled(adapters.filter(Boolean).map(adapter=>Promise.resolve().then(()=>adapter.close()))),failures=outcomes.filter(result=>result.status==='rejected').map(result=>result.reason);
 if(failures.length)throw new AggregateError(failures,'Could not release acquisition adapters');
}
