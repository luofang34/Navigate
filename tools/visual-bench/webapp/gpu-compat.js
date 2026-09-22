// The fork's wgpu binding names a device limit removed from the WebGPU API.
// Keep this translation local to this WASM module's device requests.
export function requestDeviceCompatible(adapter, descriptor) {
  const limits = descriptor.requiredLimits;
  if (limits && 'maxInterStageShaderComponents' in limits &&
      !('maxInterStageShaderComponents' in adapter.limits)) {
    const requiredLimits = {...limits};
    delete requiredLimits.maxInterStageShaderComponents;
    requiredLimits.maxInterStageShaderVariables = adapter.limits.maxInterStageShaderVariables;
    return adapter.requestDevice({...descriptor, requiredLimits});
  }
  return adapter.requestDevice(descriptor);
}
