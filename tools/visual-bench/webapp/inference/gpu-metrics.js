// Count dispatches on this adapter's device, including ORT's native WASM WebGPU path.
export function instrumentDevice(device,metrics){
  let phase='setup';const create=device.createCommandEncoder.bind(device);
  device.createCommandEncoder=(...args)=>{const encoder=create(...args),begin=encoder.beginComputePass.bind(encoder);
    encoder.beginComputePass=(...args)=>{const pass=begin(...args),dispatch=pass.dispatchWorkgroups.bind(pass),indirect=pass.dispatchWorkgroupsIndirect.bind(pass);
      const count=()=>{const key=phase+'_gpu_dispatches';metrics[key]=(metrics[key]||0)+1};
      pass.dispatchWorkgroups=(...args)=>{count();return dispatch(...args)};
      pass.dispatchWorkgroupsIndirect=(...args)=>{count();return indirect(...args)};return pass};return encoder};
  return {phase(name){phase=name},restore(){device.createCommandEncoder=create}};
}
